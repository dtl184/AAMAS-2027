"""Normative abstractions alpha = (alpha_S, alpha_A) and the formula language.

Formulas (JSON-serialisable) are built from
  * primitive atoms (strings), e.g. "at(3,1)", "hasCart", "INTERACT", "shelfE"
  * names of abstract state properties / events defined by the abstraction
  * {"not": f}, {"and": [f, ...]}, {"or": [f, ...]}
  * numeric comparisons over numeric geometric features:
        {"le": ["freeWidthH", 1]}   (also "lt", "ge", "gt", "eq")

Labels are evaluated on *transitions* (s_k, a_k, s_{k+1}).  The transition label
is  {p : s_k |= p} U {a_k} U {events e completing on this transition}.  An event
is a regular expression of length <= 2 over trajectory labels, written
pre . post, where `pre` is a formula over the label l_k (state atoms of s_k and the
action a_k) and `post` (optional) is a formula over the state atoms of s_{k+1}.
For the terminal EXIT transition, s_{k+1} is the terminal state (identical to s_k,
EXIT does not change the state).  This is Eq. (13) of the paper restricted to
patterns of length <= 2, which suffices for every abstraction used here (see README).

Primitive atoms are grouped into *families*; each vocabulary level of a domain
allows a set of families (abstractions/vocabulary is how 'hidden' low-level
information is withheld from the learner until refinement).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from environment.gridworld import GEOM_BOOL, GEOM_NUM
from environment.shopping_domain import (ACT_EXIT, ACT_ITEM0, ACT_LEAVE_CART, ACT_PICKUP_CART, CART_HELD,
                                         CART_STATION, Graph, ShoppingDomain)

# ---------------------------------------------------------------------------------------------
# primitive vocabulary
COARSE_ACTIONS = ("N", "S", "E", "W", "MOVE", "INTERACT", "EXIT")
PRIMITIVE_ACTIONS = ("PICKUP_CART", "LEAVE_CART", "PICKUP_ITEM")
ACTION_ATOMS = set(COARSE_ACTIONS) | set(PRIMITIVE_ACTIONS)
CMP_OPS = ("le", "lt", "ge", "gt", "eq")
_CELL_RE = re.compile(r"^(at|cartAt)\((-?\d+),(-?\d+)\)$")

FAMILIES = {
    "coords": "agent coordinates at(x,y)",
    "cart_possession": "hasCart (agent holds the cart)",
    "cart_location": "cartAt(x,y) (cart parked / stationed at a cell)",
    "hand": "handFull (an item is carried by hand)",
    "geometry": "generic geometric relations of the agent's cell: free, adjShelf, shelfN/S/E/W, "
                "freeWidthH, freeWidthV",
    "actions_coarse": "N, S, E, W, MOVE, INTERACT (any object interaction), EXIT",
    "actions_primitive": "PICKUP_CART, LEAVE_CART, PICKUP_ITEM",
}


def atom_family(name: str) -> Optional[str]:
    m = _CELL_RE.match(name)
    if m:
        return "coords" if m.group(1) == "at" else "cart_location"
    if name == "hasCart":
        return "cart_possession"
    if name == "handFull":
        return "hand"
    if name in GEOM_BOOL or name in GEOM_NUM:
        return "geometry"
    if name in COARSE_ACTIONS:
        return "actions_coarse"
    if name in PRIMITIVE_ACTIONS:
        return "actions_primitive"
    return None


DL_CODINGS = ("nodes", "tokens")


def formula_size(f, coding: str = "nodes") -> int:
    """Description length of a formula.

    coding="nodes"  (primary): one unit per atom, per operator, and 3 per numeric comparison
                    (operator, feature, constant).
    coding="tokens" (sensitivity analysis): as "nodes", but a cell atom at(x,y) / cartAt(x,y) costs 3
                    (predicate + two integer constants), i.e. integer constants are counted
                    consistently everywhere.
    """
    if isinstance(f, str):
        return 3 if (coding == "tokens" and _CELL_RE.match(f)) else 1
    (op, arg), = f.items()
    if op == "not":
        return 1 + formula_size(arg, coding)
    if op in ("and", "or"):
        return 1 + sum(formula_size(x, coding) for x in arg)
    if op in CMP_OPS:
        return 3
    raise ValueError(f"bad formula {f}")


def formula_names(f) -> Set[str]:
    if isinstance(f, str):
        return {f}
    (op, arg), = f.items()
    if op == "not":
        return formula_names(arg)
    if op in ("and", "or"):
        out = set()
        for x in arg:
            out |= formula_names(x)
        return out
    if op in CMP_OPS:
        return {arg[0]}
    raise ValueError(f"bad formula {f}")


def formula_str(f) -> str:
    if isinstance(f, str):
        return f
    (op, arg), = f.items()
    if op == "not":
        return "¬" + formula_str(arg)
    if op == "and":
        return "(" + " ∧ ".join(formula_str(x) for x in arg) + ")"
    if op == "or":
        return "(" + " ∨ ".join(formula_str(x) for x in arg) + ")"
    sym = {"le": "≤", "lt": "<", "ge": "≥", "gt": ">", "eq": "="}[op]
    return f"{arg[0]}{sym}{arg[1]}"


# ---------------------------------------------------------------------------------------------
class Batch:
    """A batch of transitions (s, a, s') on which atoms / formulas are evaluated (vectorised)."""

    def __init__(self, domain: ShoppingDomain, src: Dict[str, np.ndarray], act: np.ndarray,
                 dst: Dict[str, np.ndarray], terminal: np.ndarray):
        self.domain = domain
        self.src, self.dst, self.act, self.terminal = src, dst, act, terminal
        self.n = len(act)
        self._cache: Dict = {}

    @classmethod
    def from_graph(cls, domain: ShoppingDomain, g: Graph) -> "Batch":
        if g._batch is None:
            s, d = g.E_src, g.E_dst
            term = d < 0
            dd = np.where(term, s, d)
            src = {"pos": g.S_pos[s], "cart": g.S_cart[s], "hand": g.S_hand[s]}
            dst = {"pos": g.S_pos[dd], "cart": g.S_cart[dd], "hand": g.S_hand[dd]}
            g._batch = cls(domain, src, g.E_act, dst, term)
        return g._batch

    @classmethod
    def from_states(cls, domain: ShoppingDomain, states, actions) -> "Batch":
        """states: s_0..s_n (s_n terminal state), actions a_0..a_{n-1}."""
        S = np.array(states, dtype=np.int64)
        n = len(actions)
        src = {"pos": S[:n, 0], "cart": S[:n, 1], "hand": S[:n, 3]}
        dst = {"pos": S[1:n + 1, 0], "cart": S[1:n + 1, 1], "hand": S[1:n + 1, 3]}
        act = np.array(actions, dtype=np.int64)
        return cls(domain, src, act, dst, act == ACT_EXIT)

    # -------------------------------------------------------------- primitive atoms
    def state_atom(self, name: str, which: str) -> np.ndarray:
        st = self.src if which == "pre" else self.dst
        L = self.domain.layout
        m = _CELL_RE.match(name)
        if m:
            cell = (int(m.group(2)), int(m.group(3)))
            idx = L.cell_index.get(cell, -999)
            if m.group(1) == "at":
                return st["pos"] == idx
            out = st["cart"] == idx
            if cell == L.cart_station:
                out = out | (st["cart"] == CART_STATION)
            return out
        if name == "hasCart":
            return st["cart"] == CART_HELD
        if name == "handFull":
            return st["hand"] == 1
        if name in GEOM_BOOL:
            return L.geometry[name][st["pos"]]
        raise KeyError(f"unknown state atom {name}")

    def numeric(self, name: str, which: str) -> np.ndarray:
        st = self.src if which == "pre" else self.dst
        if name in GEOM_NUM:
            return self.domain.layout.geometry[name][st["pos"]]
        raise KeyError(f"unknown numeric feature {name}")

    def action_atom(self, name: str) -> np.ndarray:
        a = self.act
        if name in ("N", "S", "E", "W"):
            return a == ("N", "S", "E", "W").index(name)
        if name == "MOVE":
            return a <= 3
        if name == "EXIT":
            return a == ACT_EXIT
        if name == "INTERACT":
            return (a == ACT_PICKUP_CART) | (a == ACT_LEAVE_CART) | (a >= ACT_ITEM0)
        if name == "PICKUP_CART":
            return a == ACT_PICKUP_CART
        if name == "LEAVE_CART":
            return a == ACT_LEAVE_CART
        if name == "PICKUP_ITEM":
            return a >= ACT_ITEM0
        raise KeyError(name)


# ---------------------------------------------------------------------------------------------
@dataclass
class Abstraction:
    """alpha = (alpha_S, alpha_A): named state properties and events (length<=2 patterns)."""
    props: Dict[str, object] = field(default_factory=dict)              # name -> formula
    events: Dict[str, Tuple[object, Optional[object]]] = field(default_factory=dict)  # name -> (pre, post)

    def to_dict(self) -> dict:
        return {"state_properties": {k: self.props[k] for k in sorted(self.props)},
                "events": {k: {"pre": self.events[k][0], "post": self.events[k][1]} for k in sorted(self.events)}}

    @classmethod
    def from_dict(cls, d: dict) -> "Abstraction":
        d = d or {}
        props = dict(d.get("state_properties", {}))
        events = {k: (v["pre"], v.get("post")) for k, v in d.get("events", {}).items()}
        return cls(props=props, events=events)

    def description_length(self, coding: str = "nodes") -> int:
        """L(alpha): total formula size of all definitions (alpha_0 primitives cost 0)."""
        L = sum(formula_size(f, coding) for f in self.props.values())
        for pre, post in self.events.values():
            L += formula_size(pre, coding) + (formula_size(post, coding) if post is not None else 0)
        return L

    def describe(self) -> List[str]:
        out = [f"{k} := {formula_str(v)}" for k, v in sorted(self.props.items())]
        for k, (pre, post) in sorted(self.events.items()):
            out.append(f"{k} := {formula_str(pre)}" + (f" · {formula_str(post)}" if post is not None else ""))
        return out

    # -------------------------------------------------------------- validation
    def primitive_atoms(self, f, allow_events: bool, allow_actions: bool, _seen=None) -> Set[str]:
        """Primitive atoms used by formula f (resolving defined names); raises on illegal references."""
        out = set()
        for name in formula_names(f):
            if name in self.props:
                out |= self.primitive_atoms(self.props[name], allow_events=False, allow_actions=False)
            elif name in self.events:
                if not allow_events:
                    raise ValueError(f"event {name} used where only state formulas are allowed")
                pre, post = self.events[name]
                out |= self.primitive_atoms(pre, allow_events=False, allow_actions=True)
                if post is not None:
                    out |= self.primitive_atoms(post, allow_events=False, allow_actions=False)
            else:
                fam = atom_family(name)
                if fam is None:
                    raise ValueError(f"undefined name {name!r}")
                if fam.startswith("actions") and not allow_actions:
                    raise ValueError(f"action atom {name} used in a state formula")
                out.add(name)
        return out

    # -------------------------------------------------------------- evaluation
    def cache_key(self) -> str:
        k = getattr(self, "_ckey", None)
        if k is None:
            k = json.dumps(self.to_dict(), sort_keys=True)
            self._ckey = k
        return k

    def eval(self, f, batch: Batch, which: str = "pre") -> np.ndarray:
        key = (self.cache_key(), json.dumps(f, sort_keys=True), which)
        hit = batch._cache.get(key)
        if hit is not None:
            return hit
        if isinstance(f, str):
            if f in self.props:
                r = self.eval(self.props[f], batch, which)
            elif f in self.events:
                assert which == "pre"
                pre, post = self.events[f]
                r = self.eval(pre, batch, "pre")
                if post is not None:
                    r = r & self.eval(post, batch, "post")
            elif f in ACTION_ATOMS:
                assert which == "pre"
                r = batch.action_atom(f)
            else:
                r = batch.state_atom(f, which)
        else:
            (op, arg), = f.items()
            if op == "not":
                r = ~self.eval(arg, batch, which)
            elif op == "and":
                r = np.ones(batch.n, dtype=bool)
                for x in arg:
                    r = r & self.eval(x, batch, which)
            elif op == "or":
                r = np.zeros(batch.n, dtype=bool)
                for x in arg:
                    r = r | self.eval(x, batch, which)
            elif op in CMP_OPS:
                v = batch.numeric(arg[0], which)
                k = arg[1]
                r = {"le": v <= k, "lt": v < k, "ge": v >= k, "gt": v > k, "eq": v == k}[op]
            else:
                raise ValueError(f)
        r = np.asarray(r, dtype=bool)
        batch._cache[key] = r
        return r
