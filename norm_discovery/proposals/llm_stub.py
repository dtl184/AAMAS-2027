"""Optional LLM proposal provider (interface + prompt construction + JSON parsing).

Not used by any of the reported experiments (they use DeterministicProposalProvider so that they are
reproducible without paid API calls).  To use it, pass a callable `complete(prompt: str) -> str`
(e.g. a thin wrapper around an LLM API client) and select `--provider llm` in your own runner.
Each returned hypothesis must be a JSON object of the same schema as proposals/templates.py; invalid
or non-executable proposals are dropped by the learner's validation step.
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Sequence

from environment.trajectories import Trajectory
from norms.hypotheses import Hypothesis

from .base import ProposalProvider, render_label, vocabulary_description

SCHEMA = """Return a JSON list. Each element:
{"id": str, "text": str,
 "abstraction": {"state_properties": {name: FORMULA},
                 "events": {name: {"pre": FORMULA, "post": FORMULA or null}}},
 "norm": {"type": "prohibition", "condition": FORMULA}
       | {"type": "obligation", "trigger": FORMULA or "START", "discharge": FORMULA}}
FORMULA := atom | name of a defined property/event | {"not": F} | {"and": [F,...]} | {"or": [F,...]}
         | {"le"|"lt"|"ge"|"gt"|"eq": [numeric_feature, int]}
Events are patterns pre . post over consecutive trajectory labels (post refers to the next state)."""


class LLMProposalProvider(ProposalProvider):
    name = "llm"
    max_level = 1

    def __init__(self, complete: Callable[[str], str], n_hypotheses: int = 8):
        self.complete = complete
        self.n = n_hypotheses
        self.transcript: List[dict] = []

    def _prompt(self, spec, demos: Sequence[Trajectory], level: int, refine: bool) -> str:
        fams = spec.domain.vocab_levels[level]
        demo_txt = "\n\n".join(f"Demonstration {i + 1}:\n" + "\n".join(render_label(spec.domain, t, fams))
                               for i, t in enumerate(demos))
        task = ("The current representation could not explain the latest demonstration. Use the newly exposed "
                "vocabulary to define higher-level state properties and/or events, and propose norms over them."
                if refine else "Propose candidate norms using ONLY the given vocabulary; do not define new concepts.")
        return (f"Agents in a supermarket gridworld follow an unknown social norm.\n"
                f"Available vocabulary:\n{vocabulary_description(fams)}\n\n{demo_txt}\n\n{task}\n"
                f"Propose {self.n} diverse hypotheses (an obligation or a prohibition each).\n{SCHEMA}")

    def _parse(self, text: str, level: int) -> List[dict]:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        out = []
        for i, h in enumerate(items):
            if isinstance(h, dict) and "norm" in h:
                h.setdefault("id", f"llm_{level}_{i}")
                h["level"] = level
                h.setdefault("template", "llm")
                out.append(h)
        return out

    def propose_initial(self, spec, demos):
        p = self._prompt(spec, demos[:1], 0, refine=False)
        r = self.complete(p)
        self.transcript.append({"prompt": p, "response": r})
        return self._parse(r, 0)

    def refine(self, spec, demos, current: Sequence[Hypothesis], level: int):
        p = self._prompt(spec, demos, level, refine=True)
        r = self.complete(p)
        self.transcript.append({"prompt": p, "response": r})
        return self._parse(r, level)
