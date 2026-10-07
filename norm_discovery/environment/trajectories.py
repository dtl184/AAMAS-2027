"""Trajectories, validity checking, scripted construction, and planners."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gridworld import DIR_ORDER, DIRS
from .shopping_domain import (ACT_EXIT, ACT_LEAVE_CART, ACT_PICKUP_CART, CART_HELD, MOVE_CODES, ShoppingDomain,
                              State, Task)


@dataclass
class Trajectory:
    task: Task
    actions: List[int]
    name: str = ""
    label: Optional[bool] = None          # ground-truth "violating" label (held-out sets)
    meta: Dict = field(default_factory=dict)
    _cache: Dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def cost(self) -> float:
        return float(len(self.actions))   # every primitive action costs 1

    def states(self, domain: ShoppingDomain) -> List[State]:
        """s_0 .. s_n; s_n is the terminal state (EXIT does not change the state)."""
        key = ("states", domain.name)
        if key not in self._cache:
            s = self.task.start
            out = [s]
            for a in self.actions:
                nxt = domain.step(s, a, self.task)
                s = s if nxt is None else nxt
                out.append(s)
            self._cache[key] = out
        return self._cache[key]

    def to_dict(self, domain: ShoppingDomain) -> dict:
        return {"name": self.name, "items": list(self.task.items), "start": list(self.task.start),
                "actions": [domain.action_name(a) for a in self.actions], "label_violating": self.label,
                "length": len(self.actions), "meta": self.meta}

    @classmethod
    def from_dict(cls, domain: ShoppingDomain, d: dict) -> "Trajectory":
        task = domain.make_task(d["items"], tuple(d["start"]))
        return cls(task=task, actions=[domain.action_code(a) for a in d["actions"]], name=d.get("name", ""),
                   label=d.get("label_violating"), meta=d.get("meta", {}))

    def pretty(self, domain: ShoppingDomain) -> str:
        parts, run = [], []
        for a in self.actions:
            n = domain.action_name(a)
            if n in ("N", "S", "E", "W"):
                run.append(n)
            else:
                if run:
                    parts.append("".join(run))
                    run = []
                parts.append(n)
        if run:
            parts.append("".join(run))
        return " ".join(parts)


def is_valid(domain: ShoppingDomain, traj: Trajectory) -> Tuple[bool, str]:
    """A trajectory is valid iff every action is available and it ends with (exactly one) EXIT."""
    s = traj.task.start
    for k, a in enumerate(traj.actions):
        if not domain.is_available(s, a, traj.task):
            return False, f"action {k} ({domain.action_name(a)}) unavailable in {domain.describe_state(s, traj.task)}"
        nxt = domain.step(s, a, traj.task)
        if nxt is None:
            if k != len(traj.actions) - 1:
                return False, "EXIT before end of trajectory"
            return True, "ok"
        s = nxt
    return False, "trajectory does not end with EXIT"


# ------------------------------------------------------------------------------------------- scripts
class ScriptBuilder:
    """Build trajectories from high-level steps (goto / interactions) with shortest-path navigation."""

    def __init__(self, domain: ShoppingDomain, task: Task):
        self.d, self.task = domain, task
        self.s: State = task.start
        self.actions: List[int] = []

    def _apply(self, a: int):
        nxt = self.d.step(self.s, a, self.task)
        self.actions.append(a)
        if nxt is not None:
            self.s = nxt
        return self

    def goto(self, cell, avoid: Sequence = ()):
        L = self.d.layout
        start, goal = self.d.layout.free_cells[self.s[0]], tuple(cell)
        avoid = {tuple(c) for c in avoid}
        prev = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                break
            for dname in DIR_ORDER:
                dx, dy = DIRS[dname]
                n = (c[0] + dx, c[1] + dy)
                if L.is_free(n) and n not in prev and (n not in avoid or n == goal):
                    prev[n] = (c, dname)
                    q.append(n)
        if goal not in prev:
            raise ValueError(f"no path to {goal}")
        path = []
        c = goal
        while prev[c] is not None:
            c, dname = prev[c][0], prev[c][1]
            path.append(dname)
        for dname in reversed(path):
            self._apply(MOVE_CODES[dname])
        return self

    def pickup_cart(self):
        return self._apply(ACT_PICKUP_CART)

    def leave_cart(self):
        return self._apply(ACT_LEAVE_CART)

    def pick(self, item: str, from_cell=None, avoid: Sequence = ()):
        if from_cell is None:
            cands = self.d.layout.item_access_cells(item)
            from_cell = min(cands, key=lambda c: (abs(c[0] - self.d.cell(self.s[0])[0]) +
                                                  abs(c[1] - self.d.cell(self.s[0])[1]), c))
        self.goto(from_cell, avoid=avoid)
        return self._apply(self.d.item_code[item])

    def exit(self, avoid: Sequence = ()):
        self.goto(self.d.layout.exit, avoid=avoid)
        return self._apply(ACT_EXIT)

    def build(self, name: str = "", label: Optional[bool] = None, **meta) -> Trajectory:
        t = Trajectory(task=self.task, actions=list(self.actions), name=name, label=label, meta=dict(meta))
        ok, msg = is_valid(self.d, t)
        if not ok:
            raise ValueError(f"scripted trajectory {name} invalid: {msg}")
        return t


# ------------------------------------------------------------------------------------------- planning
def _product(domain: ShoppingDomain, task: Task, hyp=None):
    """Product of the task graph with the hypothesis monitor (K=1 if hyp is None)."""
    from abstractions.base import Batch  # local import to avoid cycles
    g = domain.graph(task)
    if hyp is None:
        K, init = 1, 0
        nxt = [np.zeros(g.n_edges, dtype=np.int64)]
        viol = [np.zeros(g.n_edges, dtype=np.int64)]
    else:
        batch = Batch.from_graph(domain, g)
        K, init, nxt, viol = hyp.norm.monitor(hyp.abstraction, batch)
    src, dst, act, vv = [], [], [], []
    for m in range(K):
        src.append(g.E_src * K + m)
        dst.append(np.where(g.E_dst >= 0, g.E_dst * K + nxt[m], -1))
        act.append(g.E_act)
        vv.append(viol[m])
    return g, K, init, np.concatenate(src), np.concatenate(dst), np.concatenate(act), np.concatenate(vv)


def distances_to_goal(n_nodes: int, src, dst, allowed) -> np.ndarray:
    """Unit-cost shortest distance from every node to termination using only `allowed` edges."""
    INF = np.iinfo(np.int64).max // 4
    dist = np.full(n_nodes, INF, dtype=np.int64)
    s, d = src[allowed], dst[allowed]
    while True:
        cand = np.where(d < 0, 1, np.where(d >= 0, dist[np.maximum(d, 0)] + 1, INF))
        new = dist.copy()
        np.minimum.at(new, s, cand)
        if np.array_equal(new, dist):
            return dist
        dist = new


def plan_optimal(domain: ShoppingDomain, task: Task, hyp=None, rng: Optional[np.random.Generator] = None,
                 name: str = "", allow_violations: bool = False) -> Trajectory:
    """Shortest trajectory that never violates `hyp` (hard constraint).

    Ties among optimal actions are broken by canonical action order (rng=None) or uniformly at random.
    """
    g, K, init, src, dst, act, vv = _product(domain, task, hyp)
    allowed = np.ones(len(src), dtype=bool) if allow_violations else (vv == 0)
    n_nodes = g.n_states * K
    dist = distances_to_goal(n_nodes, src, dst, allowed)
    order = np.argsort(src, kind="stable")
    starts = np.searchsorted(src[order], np.arange(n_nodes + 1))
    node = 0 * K + init
    if dist[node] > 10 ** 9:
        raise ValueError("no compliant trajectory exists")
    actions = []
    while True:
        es = order[starts[node]:starts[node + 1]]
        es = es[allowed[es]]
        good = [e for e in es if (dst[e] < 0 and dist[node] == 1) or (dst[e] >= 0 and dist[dst[e]] == dist[node] - 1)]
        e = good[0] if rng is None else good[int(rng.integers(len(good)))]
        actions.append(int(act[e]))
        if dst[e] < 0:
            break
        node = int(dst[e])
    t = Trajectory(task=task, actions=actions, name=name)
    assert is_valid(domain, t)[0]
    return t
