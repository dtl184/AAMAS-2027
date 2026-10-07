"""Reconstructed maximum-likelihood constraint inference (MLCI), after Scobee & Sastry (ICLR 2020).

Model.  Demonstrations are drawn from the maximum-entropy distribution over trajectories of the
known task model restricted to trajectories that satisfy a set of *hard* constraints C:
    P(tau | C) = exp(-beta C_task(tau)) 1[tau satisfies C] / Z_C(s0, g).
For trajectories that satisfy C, adding a constraint never lowers P, and the likelihood of the
demonstrations increases by  sum_d [log Z_C(ctx_d) - log Z_{C ∪ {c}}(ctx_d)], i.e. by the probability
mass the new constraint removes.  Constraints violated by any demonstration have likelihood zero and
are never added.

Candidate constraints (over the low-level representation only -- no invented predicates):
    forbid transitions (s, a) with  A(s) ∧ B(s, a)
    A ∈ {true} ∪ {at(x,y)} ∪ {cartAt(x,y)}                      (state features)
    B ∈ {true, hasCart, ¬hasCart} ∪ {N,S,E,W,PICKUP_CART,LEAVE_CART,PICKUP_ITEM,INTERACT,EXIT}
The families available are those in domain.mlci_families (default: coordinates, cart possession,
cart location, primitive and coarse actions; geometric relations are *excluded* by default and can
be enabled with include_geometry=True as an ablation, adding A ∈ {free, adjShelf, shelfN/S/E/W}).

Greedy inference (Scobee & Sastry, Alg. 1): repeatedly
  1. compute the expected visitation of every candidate under the current constrained
     max-ent distribution of each demonstration context (exact forward/backward on the state graph);
  2. take the K candidates with the largest summed expected visitation that no demonstration violates;
  3. evaluate their exact log-likelihood gain and add the best;
  4. stop when the best gain < epsilon (nats, summed over demonstrations) or max_constraints reached.
Held-out prediction: a trajectory is classified violating iff it contains a forbidden transition.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from environment.gridworld import GEOM_BOOL
from environment.shopping_domain import (ACT_EXIT, ACT_ITEM0, ACT_LEAVE_CART, ACT_PICKUP_CART, CART_HELD,
                                         CART_STATION, ShoppingDomain, Task)
from environment.trajectories import Trajectory
from inference.likelihood import solve_direct
from inference.predictive import evaluate_predictions

B_NAMES = ["true", "hasCart", "¬hasCart", "N", "S", "E", "W", "PICKUP_CART", "LEAVE_CART", "PICKUP_ITEM",
           "INTERACT", "EXIT"]


@dataclass
class MLCIConfig:
    beta: float = 2.0
    # minimum log-likelihood gain (nats, summed over demonstrations) required to add a constraint.
    # "log_candidates" (default) = log(number of admissible candidate constraints): a Bonferroni/BIC-style
    # correction for choosing one constraint among many, fixed a priori (mirrors the description-length
    # prior of the Bayesian methods).  A float gives a fixed threshold (e.g. 0.1, used as a sensitivity check).
    epsilon: object = "log_candidates"
    max_constraints: int = 30
    top_k: int = 5
    include_geometry: bool = False


class _EdgeFeatures:
    """For every transition: the A-feature ids and B-feature ids it satisfies."""

    def __init__(self, domain: ShoppingDomain, pos, cart, act, include_geometry: bool):
        L = domain.layout
        n_cells = len(L.free_cells)
        fam = set(domain.mlci_families)
        self.a_names = ["true"]
        a_cols = [np.zeros(len(act), dtype=np.int64)]                    # every edge matches 'true'
        if "coords" in fam:
            a_cols.append(1 + pos)
        self.a_names += [f"at{L.free_cells[i]}" for i in range(n_cells)]
        if "cart_location" in fam:
            cl = np.where(cart == CART_STATION, domain.station_idx, cart)
            a_cols.append(np.where(cart == CART_HELD, -1, 1 + n_cells + cl))
        self.a_names += [f"cartAt{L.free_cells[i]}" for i in range(n_cells)]
        self.geom = list(GEOM_BOOL) if include_geometry else []
        for gi, g in enumerate(self.geom):
            a_cols.append(np.where(L.geometry[g][pos], 1 + 2 * n_cells + gi, -1))
        self.a_names += self.geom
        b_cols = [np.zeros(len(act), dtype=np.int64)]
        if "cart_possession" in fam:
            b_cols.append(np.where(cart == CART_HELD, 1, 2))
        prim = "actions_primitive" in fam
        coarse = "actions_coarse" in fam
        code = np.full(len(act), -1, dtype=np.int64)
        if coarse:
            code = np.where(act <= 3, 3 + act, code)
            code = np.where(act == ACT_EXIT, 11, code)
        if prim:
            code = np.where(act == ACT_PICKUP_CART, 7, code)
            code = np.where(act == ACT_LEAVE_CART, 8, code)
            code = np.where(act >= ACT_ITEM0, 9, code)
        b_cols.append(code)
        if coarse:
            inter = (act == ACT_PICKUP_CART) | (act == ACT_LEAVE_CART) | (act >= ACT_ITEM0)
            b_cols.append(np.where(inter, 10, -1))
        self.A = np.stack(a_cols, 1)
        self.B = np.stack(b_cols, 1)
        self.nA, self.nB = len(self.a_names), len(B_NAMES)

    def table(self, weights: np.ndarray) -> np.ndarray:
        """T[a, b] = sum of weights of edges matching (a, b)."""
        T = np.zeros(self.nA * self.nB)
        for i in range(self.A.shape[1]):
            for j in range(self.B.shape[1]):
                a, b = self.A[:, i], self.B[:, j]
                ok = (a >= 0) & (b >= 0)
                np.add.at(T, a[ok] * self.nB + b[ok], weights[ok])
        T = T.reshape(self.nA, self.nB)
        T[0, 0] = 0.0                                   # 'true ∧ true' is not a constraint
        return T

    def matches(self, a: int, b: int) -> np.ndarray:
        return (self.A == a).any(1) & (self.B == b).any(1)


class _Context:
    def __init__(self, domain: ShoppingDomain, task: Task, beta: float, include_geometry: bool):
        g = domain.graph(task)
        self.g = g
        self.feat = _EdgeFeatures(domain, g.S_pos[g.E_src], g.S_cart[g.E_src], g.E_act, include_geometry)
        self.wgt = np.exp(-beta)

    def solve(self, forbidden: np.ndarray, need_forward: bool):
        g = self.g
        keep = ~forbidden
        nt = keep & (g.E_dst >= 0)
        tm = keep & (g.E_dst < 0)
        n = g.n_states
        M = sp.csr_matrix((np.full(nt.sum(), self.wgt), (g.E_src[nt], g.E_dst[nt])), shape=(n, n))
        b = np.zeros(n)
        np.add.at(b, g.E_src[tm], self.wgt)
        z = solve_direct(M, b)
        if not need_forward:
            return z, None
        e0 = np.zeros(n)
        e0[0] = 1.0
        f = solve_direct(M.T.tocsr(), e0)
        return z, f

    def occupancy(self, forbidden: np.ndarray) -> Tuple[float, np.ndarray]:
        g = self.g
        z, f = self.solve(forbidden, True)
        zd = np.where(g.E_dst >= 0, z[np.maximum(g.E_dst, 0)], 1.0)
        occ = np.where(forbidden, 0.0, f[g.E_src] * self.wgt * zd / z[0])
        return float(np.log(z[0])), occ


def _traj_edges(domain, traj: Trajectory):
    st = np.array(traj.states(domain)[:-1], dtype=np.int64)
    return st[:, 0], st[:, 1], np.array(traj.actions, dtype=np.int64)


class MLCI:
    def __init__(self, spec, cfg: Optional[MLCIConfig] = None):
        self.spec, self.cfg = spec, cfg or MLCIConfig()
        self.constraints: List[Tuple[int, int]] = []
        self.history: List[dict] = []

    def describe(self, c) -> str:
        a, b = c
        fa = self._names_a[a]
        return f"FORBID {fa} ∧ {B_NAMES[b]}" if a else f"FORBID {B_NAMES[b]}"

    def fit(self, demos: Sequence[Trajectory]) -> "MLCI":
        t0 = time.time()
        d, cfg = self.spec.domain, self.cfg
        ctx: Dict[str, _Context] = {}
        counts: Dict[str, int] = {}
        for t in demos:
            k = t.task.key
            if k not in ctx:
                ctx[k] = _Context(d, t.task, cfg.beta, cfg.include_geometry)
                counts[k] = 0
            counts[k] += 1
        any_ctx = next(iter(ctx.values()))
        self._names_a = any_ctx.feat.a_names
        # candidates violated by some demonstration are excluded
        violated = np.zeros((any_ctx.feat.nA, any_ctx.feat.nB), dtype=bool)
        for t in demos:
            pos, cart, act = _traj_edges(d, t)
            fe = _EdgeFeatures(d, pos, cart, act, cfg.include_geometry)
            violated |= fe.table(np.ones(len(act))) > 0
        violated[0, 0] = True
        n_cand = int((~violated).sum())
        eps = float(np.log(max(n_cand, 2))) if cfg.epsilon == "log_candidates" else float(cfg.epsilon)
        self.epsilon_used, self.n_candidates = eps, n_cand
        forb = {k: np.zeros(c.g.n_edges, dtype=bool) for k, c in ctx.items()}
        logZ = {}
        for it in range(cfg.max_constraints):
            score = np.zeros_like(violated, dtype=float)
            for k, c in ctx.items():
                lz, occ = c.occupancy(forb[k])
                logZ[k] = lz
                score += counts[k] * c.feat.table(occ)
            score[violated] = -np.inf
            for (a, b) in self.constraints:
                score[a, b] = -np.inf
            flat = np.argsort(-score, axis=None)[:cfg.top_k]
            best, best_gain = None, -np.inf
            for f in flat:
                a, b = np.unravel_index(f, score.shape)
                if not np.isfinite(score[a, b]) or score[a, b] <= 0:
                    continue
                gain = 0.0
                for k, c in ctx.items():
                    newf = forb[k] | c.feat.matches(a, b)
                    z, _ = c.solve(newf, False)
                    gain += counts[k] * (logZ[k] - np.log(z[0]))
                if gain > best_gain:
                    best, best_gain = (int(a), int(b)), gain
            if best is None or best_gain < eps:
                self.history.append({"iteration": it, "stop": True, "best_gain": float(best_gain)})
                break
            self.constraints.append(best)
            for k, c in ctx.items():
                forb[k] |= c.feat.matches(*best)
            self.history.append({"iteration": it, "constraint": self.describe(best), "gain": float(best_gain)})
        self.runtime_s = time.time() - t0
        return self

    def violates(self, traj: Trajectory) -> bool:
        d = self.spec.domain
        pos, cart, act = _traj_edges(d, traj)
        fe = _EdgeFeatures(d, pos, cart, act, self.cfg.include_geometry)
        return any(fe.matches(a, b).any() for a, b in self.constraints)

    def evaluate(self) -> Dict:
        pred = [self.violates(t) for t in self.spec.heldout]
        ev = evaluate_predictions(self.spec, pred)
        ev.update({"n_constraints": len(self.constraints), "epsilon": getattr(self, "epsilon_used", None),
                   "n_candidates": getattr(self, "n_candidates", None),
                   "constraints": [self.describe(c) for c in self.constraints],
                   "runtime_s": getattr(self, "runtime_s", None)})
        return ev
