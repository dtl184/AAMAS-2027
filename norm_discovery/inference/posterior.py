"""Posterior P(h, w | D, M) over normative hypotheses and the violation-cost grid W.

  P(h, w_j | D) ∝ P(D | h, w_j) P(w_j | h) P(h),   P(w_j | h) = 1/|W|,
  P(h) = P(N | alpha) P(alpha),  P(alpha) ∝ exp(-lambda L(alpha)),  P(N | alpha) ∝ exp(-gamma L(N)).

prior_mode="joint" (default): P(h) ∝ exp(-lambda L(alpha) - gamma L(N)), normalised over the current
hypothesis set.  prior_mode="conditional": P(alpha) normalised over the distinct abstractions in the
set and P(N | alpha) normalised among the norms sharing that abstraction (see README).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import logsumexp

from environment.trajectories import Trajectory
from inference.likelihood import ZComputer
from norms.hypotheses import Hypothesis


class Posterior:
    def __init__(self, zc: ZComputer, lam: float, gamma: float, prior_mode: str = "joint", dl_coding: str = "nodes"):
        self.zc, self.lam, self.gamma, self.prior_mode = zc, float(lam), float(gamma), prior_mode
        self.dl_coding = dl_coding
        self.W = zc.W
        self.hyps: List[Hypothesis] = []
        self._keys: Dict[str, int] = {}
        self.demos: List[Trajectory] = []
        self.loglik: List[List[np.ndarray]] = []       # [h][t] -> (J,)

    # ------------------------------------------------------------------ bookkeeping
    def has(self, h: Hypothesis) -> bool:
        return h.key in self._keys

    def index_of_key(self, key: str) -> Optional[int]:
        return self._keys.get(key)

    def add_hypotheses(self, hyps: Sequence[Hypothesis]) -> List[Hypothesis]:
        added = []
        for h in hyps:
            if h.key in self._keys:
                continue
            self._keys[h.key] = len(self.hyps)
            self.hyps.append(h)
            self.loglik.append([self.zc.loglik(h, t) for t in self.demos])
            added.append(h)
        return added

    def add_demo(self, traj: Trajectory) -> None:
        self.demos.append(traj)
        for i, h in enumerate(self.hyps):
            self.loglik[i].append(self.zc.loglik(h, traj))

    # ------------------------------------------------------------------ probabilities
    def log_prior_h(self) -> np.ndarray:
        La = np.array([h.L_alpha(self.dl_coding) for h in self.hyps], dtype=float)
        Ln = np.array([h.L_norm(self.dl_coding) for h in self.hyps], dtype=float)
        if self.prior_mode == "joint":
            lp = -self.lam * La - self.gamma * Ln
            return lp - logsumexp(lp)
        if self.prior_mode == "conditional":
            akeys = [str(h.abstraction.to_dict()) for h in self.hyps]
            uniq = sorted(set(akeys))
            la = {k: -self.lam * La[akeys.index(k)] for k in uniq}
            Z_a = logsumexp(list(la.values()))
            out = np.empty(len(self.hyps))
            for i, k in enumerate(akeys):
                members = [j for j, kk in enumerate(akeys) if kk == k]
                ln = -self.gamma * Ln[members]
                out[i] = (la[k] - Z_a) + (-self.gamma * Ln[i] - logsumexp(ln))
            return out
        raise ValueError(self.prior_mode)

    def log_joint(self, upto: Optional[int] = None) -> np.ndarray:
        """Normalised log P(h, w | D_{1:upto}) as an (H, J) array."""
        J = len(self.W)
        lp = self.log_prior_h()[:, None] - np.log(J)
        n = len(self.demos) if upto is None else upto
        ll = np.array([np.sum(rows[:n], axis=0) if n else np.zeros(J) for rows in self.loglik])
        x = lp + ll
        return x - logsumexp(x)

    def predictive_logprob(self, traj: Trajectory) -> float:
        """log P(tau | D, M) = log sum_{h,w} P(tau | h, w) P(h, w | D)   (Eq. 29, sum over W)."""
        lj = self.log_joint()
        lt = np.array([self.zc.loglik(h, traj) for h in self.hyps])
        return float(logsumexp(lj + lt))

    def marginal(self, upto: Optional[int] = None) -> np.ndarray:
        return np.exp(logsumexp(self.log_joint(upto), axis=1))

    def entropy(self, upto: Optional[int] = None) -> float:
        p = self.marginal(upto)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def map_index(self, upto: Optional[int] = None) -> int:
        return int(np.argmax(self.marginal(upto)))

    def w_posterior_mean(self, i: int, upto: Optional[int] = None) -> float:
        lj = self.log_joint(upto)[i]
        p = np.exp(lj - logsumexp(lj))
        return float((p * self.W).sum())

    def table(self, top: int = 10, upto: Optional[int] = None) -> List[dict]:
        m = self.marginal(upto)
        lp = self.log_prior_h()
        order = np.argsort(-m)[:top]
        return [{"id": self.hyps[i].hid, "level": self.hyps[i].level, "posterior": float(m[i]),
                 "log_prior": float(lp[i]), "L_alpha": self.hyps[i].L_alpha(self.dl_coding), "L_norm": self.hyps[i].L_norm(self.dl_coding),
                 "E_w": self.w_posterior_mean(int(i), upto), "description": self.hyps[i].describe()}
                for i in order]
