"""LLM proposal provider: prompt construction, JSON parsing, and full transcript recording.

`LLMProposalProvider(complete)` takes any callable `complete(prompt: str) -> str`.  The OpenAI-backed callable is
proposals/openai_provider.py::OpenAICompletion (used by the GPT-5.5 experiments).  If the callable exposes
`last_call` (a metadata dict: returned model, request id, usage, retries, cache hit ...) it is copied into the
transcript.  Every returned hypothesis must follow the JSON schema of proposals/templates.py; the learner
validates each one and reports back which were accepted / rejected (record_validation).

Failure policy: if a response cannot be parsed as a JSON list of hypotheses, the provider re-asks once with a
format reminder (recorded); if that also fails, LLMParseError is raised and the run stops.  There is never a
fallback to deterministic templates.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Callable, Dict, List, Optional, Sequence

from environment.trajectories import Trajectory
from norms.hypotheses import Hypothesis

from .base import ProposalProvider, render_label

ATOM_SYNTAX = {
    "coords": "at(x,y) -- the agent is at cell (x,y); x = column (0 = left), y = row (0 = top); "
              "N decreases y, S increases y, E increases x, W decreases x",
    "cart_possession": "hasCart -- the agent is holding/pushing a shopping cart",
    "cart_location": "cartAt(x,y) -- the cart is parked (or stationed) at cell (x,y)",
    "hand": "handFull -- the agent carries an item by hand",
    "geometry": "free, adjShelf (some 4-neighbour of the agent's cell is a shelf), shelfN / shelfS / shelfE / shelfW "
                "(the neighbouring cell in that direction is a shelf); numeric features freeWidthH / freeWidthV "
                "(length of the horizontal / vertical run of free cells through the agent's cell), usable only "
                "inside comparisons such as {\"le\": [\"freeWidthH\", 2]}",
    "actions_coarse": "actions N, S, E, W, MOVE (any of the four moves), INTERACT (any interaction with an object: "
                      "the different kinds of interaction are NOT distinguished), EXIT (leave the store; always the "
                      "last action)",
    "actions_primitive": "actions PICKUP_CART, LEAVE_CART (put the cart down on the current cell), PICKUP_ITEM "
                         "(take an item from an adjacent shelf); each of these is also an INTERACT",
}

SCHEMA = """Output format: return ONLY a JSON list (no prose, no code fences). Each element:
{"id": short_name, "text": one-sentence natural-language statement of the norm,
 "abstraction": {"state_properties": {name: FORMULA, ...},
                 "events": {name: {"pre": FORMULA, "post": FORMULA or null}, ...}},
 "norm": {"type": "prohibition", "condition": FORMULA}
       | {"type": "obligation", "trigger": FORMULA or "START", "discharge": FORMULA}}
FORMULA := atom | name of a property/event defined in this hypothesis | {"not": F} | {"and": [F, ...]}
         | {"or": [F, ...]} | {"le"|"lt"|"ge"|"gt"|"eq": [numeric_feature, integer]}
Semantics.  A trajectory is a sequence of steps (state s_k, action a_k).  A step's label contains the atoms true in
s_k and the action a_k.  A state property is a formula over state atoms (no actions).  An event
{"pre": P, "post": Q} occurs at step k when P holds for the label of step k and Q holds in the next state s_{k+1}
(post may be null).  A prohibition counts one violation for every step whose label (including events occurring at
that step) satisfies the condition.  An obligation becomes pending at any step satisfying the trigger ("START" =
pending from the beginning) and is fulfilled by a later (or the same) step satisfying the discharge; it can be
triggered again; it is violated if it is still pending when the trajectory ends."""


class LLMParseError(RuntimeError):
    pass


def _strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t


def parse_hypotheses(text: str) -> Optional[List[dict]]:
    """Return the list of hypothesis objects, or None if the text is not parseable."""
    t = _strip_fences(text)
    candidates = [t]
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            obj = obj.get("hypotheses", obj.get("norms"))
        if isinstance(obj, list) and all(isinstance(h, dict) for h in obj):
            return obj
    return None


class LLMProposalProvider(ProposalProvider):
    name = "llm"
    max_level = 1

    def __init__(self, complete: Callable[[str], str], n_hypotheses: int = 8, max_parse_retries: int = 1,
                 run_info: Optional[Dict] = None):
        self.complete = complete
        self.n = n_hypotheses
        self.max_parse_retries = max_parse_retries
        self.run_info = dict(run_info or {})
        self.transcript: List[dict] = []

    # ------------------------------------------------------------------ prompts
    def _prompt(self, spec, demos: Sequence[Trajectory], level: int, refine: bool,
                current: Sequence[Hypothesis] = ()) -> str:
        d = spec.domain
        fams = d.vocab_levels[level]
        vocab = "\n".join(f"- {ATOM_SYNTAX[f]}" for f in fams)
        demo_txt = "\n\n".join(
            f"Demonstration {i + 1} (required items: {', '.join(t.task.items) or 'none'}):\n" +
            "\n".join(render_label(d, t, fams)) for i, t in enumerate(demos))
        head = ("Agents in a supermarket gridworld must collect their required items and leave through the exit. "
                "Every action costs the same, so demonstrators prefer short trajectories -- unless an unknown "
                "social norm makes them behave otherwise. Your task is to propose candidate norms (obligations or "
                "prohibitions) that could explain the demonstrated behaviour.\n"
                "Each demonstration is listed step by step as  `k: {atoms true in the state} action`.\n")
        if refine:
            prev_fams = d.vocab_levels[max(0, level - 1)]
            new = [f for f in fams if f not in prev_fams]
            cur = "\n".join(f"- {h.describe()}" for h in current) or "- (none)"
            task = ("The current hypotheses, expressed in a coarser representation, do not adequately explain the "
                    "latest demonstration:\n" + cur + "\n\nThe demonstrations are now shown in a lower-level "
                    "representation that exposes distinctions hidden before (newly available: " + ", ".join(new) +
                    "). Identify the hidden distinctions and use them to define higher-level state properties and/or "
                    "events (in \"abstraction\"), possibly hierarchically, and propose norms expressed over them. "
                    "Norms may also use the primitive atoms directly.")
        else:
            task = ("Propose norms using ONLY the atoms listed above. Do NOT define any new state properties or "
                    "events: \"abstraction\" must be {}.")
        return (f"{head}\nAvailable atoms:\n{vocab}\n\n{demo_txt}\n\n{task}\n"
                f"Propose {self.n} diverse candidate hypotheses.\n\n{SCHEMA}")

    # ------------------------------------------------------------------ calls
    def _ask(self, prompt: str, kind: str, level: int, n_demos: int) -> List[dict]:
        attempts = []
        p = prompt
        for attempt in range(self.max_parse_retries + 1):
            raw = self.complete(p)
            meta = getattr(self.complete, "last_call", None)
            parsed = parse_hypotheses(raw)
            attempts.append({"attempt": attempt, "prompt": p, "raw_response": raw, "parsed": parsed,
                             "call": {k: v for k, v in (meta or {}).items() if k != "output_text"}})
            if parsed is not None:
                break
            p = (prompt + "\n\nIMPORTANT: your previous answer could not be parsed. Reply with ONLY the JSON list, "
                          "nothing else.")
        entry = {"index": len(self.transcript), "kind": kind, "level": level, "n_demos_shown": n_demos,
                 "timestamp": _dt.datetime.now().isoformat(timespec="seconds"), **self.run_info,
                 "prompt": prompt, "raw_response": attempts[-1]["raw_response"], "parsed": attempts[-1]["parsed"],
                 "attempts": attempts, "accepted": None, "rejected": None}
        self.transcript.append(entry)
        if parsed is None:
            entry["error"] = "unparseable response after retries"
            raise LLMParseError(f"{kind} response could not be parsed after {len(attempts)} attempts")
        out = []
        for i, h in enumerate(parsed):
            h = dict(h)
            h["id"] = f"g{entry['index']}_{h.get('id', i)}"
            h["level"] = level
            h["template"] = f"llm_{kind}"
            out.append(h)
        return out

    def propose_initial(self, spec, demos):
        return self._ask(self._prompt(spec, demos[:1], 0, refine=False), "initial", 0, 1)

    def refine(self, spec, demos, current: Sequence[Hypothesis], level: int):
        return self._ask(self._prompt(spec, demos, level, refine=True, current=current), "refinement", level,
                         len(demos))

    def record_validation(self, accepted: List[dict], rejected: List[dict]) -> None:
        """Called by the learner after validating the most recent call's proposals."""
        if self.transcript:
            self.transcript[-1]["accepted"] = accepted
            self.transcript[-1]["rejected"] = rejected
