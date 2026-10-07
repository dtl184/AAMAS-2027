"""Norms N and normative hypotheses h = (alpha, N).

Norm types (each compiled to a small deterministic monitor so that both the
violation count V_h(tau) of a trajectory and the partition function can be
computed exactly):

  prohibition(condition)
      one violation for every transition whose label satisfies `condition`.
  obligation(trigger, discharge)
      after a transition satisfying `trigger`, a later transition satisfying
      `discharge` must occur before the trajectory ends.  Monitor state m in {0,1}
      (obligation pending).  Update on each transition:
          m' = (m or trigger) and not discharge
      (a discharge on the same transition as the trigger satisfies it).  The
      obligation is re-triggerable.  One violation is counted on the terminal
      transition if m' = 1.  trigger = "START" means the obligation is pending
      from the first step (an unconditional obligation).

Description length L(N) = 1 (norm type) + size of every formula it uses
("START" counts 1).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from abstractions.base import Abstraction, Batch, atom_family, formula_size, formula_str


@dataclass
class Norm:
    kind: str                      # "prohibition" | "obligation"
    condition: object = None       # prohibition
    trigger: object = None         # obligation ("START" or formula)
    discharge: object = None       # obligation

    def to_dict(self) -> dict:
        if self.kind == "prohibition":
            return {"type": "prohibition", "condition": self.condition}
        return {"type": "obligation", "trigger": self.trigger, "discharge": self.discharge}

    @classmethod
    def from_dict(cls, d: dict) -> "Norm":
        if d["type"] == "prohibition":
            return cls("prohibition", condition=d["condition"])
        if d["type"] == "obligation":
            return cls("obligation", trigger=d["trigger"], discharge=d["discharge"])
        raise ValueError(d)

    def formulas(self) -> List[object]:
        if self.kind == "prohibition":
            return [self.condition]
        return ([] if self.trigger == "START" else [self.trigger]) + [self.discharge]

    def description_length(self, coding: str = "nodes") -> int:
        if self.kind == "prohibition":
            return 1 + formula_size(self.condition, coding)
        return 1 + (1 if self.trigger == "START" else formula_size(self.trigger, coding)) + \
            formula_size(self.discharge, coding)

    def describe(self) -> str:
        if self.kind == "prohibition":
            return f"FORBIDDEN: {formula_str(self.condition)}"
        trig = "start" if self.trigger == "START" else formula_str(self.trigger)
        return f"OBLIGED: after {trig}, eventually {formula_str(self.discharge)} (before exit)"

    # ---------------------------------------------------------------- monitor
    def monitor(self, abstraction: Abstraction, batch: Batch):
        """Return (K, init, nxt, viol): nxt[m], viol[m] are arrays over the batch's transitions."""
        if self.kind == "prohibition":
            c = abstraction.eval(self.condition, batch)
            return 1, 0, [np.zeros(batch.n, dtype=np.int64)], [c.astype(np.int64)]
        dis = abstraction.eval(self.discharge, batch)
        if self.trigger == "START":
            trig = np.zeros(batch.n, dtype=bool)
            init = 1
        else:
            trig = abstraction.eval(self.trigger, batch)
            init = 0
        nxt0 = (trig & ~dis)
        nxt1 = ~dis
        term = batch.terminal
        return 2, init, [nxt0.astype(np.int64), nxt1.astype(np.int64)], \
            [(term & nxt0).astype(np.int64), (term & nxt1).astype(np.int64)]


@dataclass
class Hypothesis:
    hid: str
    abstraction: Abstraction
    norm: Norm
    level: int = 0
    text: str = ""
    template: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.hid, "level": self.level, "template": self.template, "text": self.text,
                "abstraction": self.abstraction.to_dict(), "norm": self.norm.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(hid=d["id"], abstraction=Abstraction.from_dict(d.get("abstraction")),
                   norm=Norm.from_dict(d["norm"]), level=int(d.get("level", 0)), text=d.get("text", ""),
                   template=d.get("template", ""))

    @property
    def key(self) -> str:
        """Canonical content key (abstraction + norm); identical content => identical hypothesis."""
        return json.dumps({"a": self.abstraction.to_dict(), "n": self.norm.to_dict()}, sort_keys=True)

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.key.encode()).hexdigest()[:12]

    def L_alpha(self, coding: str = "nodes") -> int:
        return self.abstraction.description_length(coding)

    def L_norm(self, coding: str = "nodes") -> int:
        return self.norm.description_length(coding)

    def describe(self) -> str:
        defs = "; ".join(self.abstraction.describe())
        return f"[{self.hid}] {self.norm.describe()}" + (f"  where {defs}" if defs else "")

    # ---------------------------------------------------------------- validation
    def primitive_atoms(self) -> set:
        out = set()
        for f in self.norm.formulas():
            out |= self.abstraction.primitive_atoms(f, allow_events=True, allow_actions=True)
        # definitions that are not referenced still have to be legal
        for name, f in self.abstraction.props.items():
            out |= self.abstraction.primitive_atoms(f, allow_events=False, allow_actions=False)
        for name, (pre, post) in self.abstraction.events.items():
            out |= self.abstraction.primitive_atoms(pre, allow_events=False, allow_actions=True)
            if post is not None:
                out |= self.abstraction.primitive_atoms(post, allow_events=False, allow_actions=False)
        return out

    def validate(self, allowed_families: Sequence[str]) -> None:
        """Check that the hypothesis is executable over the allowed low-level vocabulary."""
        bad = sorted(a for a in self.primitive_atoms() if atom_family(a) not in allowed_families)
        if bad:
            raise ValueError(f"hypothesis {self.hid} uses atoms outside vocabulary {allowed_families}: {bad}")
