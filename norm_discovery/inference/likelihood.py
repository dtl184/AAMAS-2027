"""Maximum-entropy trajectory likelihood and exact partition functions.

  P(tau | h, w, M) = exp(-beta (C_task(tau) + w V_h(tau))) / Z(h, w, s0, g)

Z sums over *all* feasible trajectories from s0 that terminate with EXIT at the goal
(including arbitrarily long / cyclic ones).  Because V_h is produced by a finite
monitor (norms/hypotheses.py), Z is computed exactly on the product of the task's
low-level state graph with the norm monitor by solving the linear system

    z(x) = sum_{edges x -a-> x'} exp(-beta c(x,a)) exp(-beta w v(x,a)) z(x')     (z(terminal) = 1)

for all w in the grid W simultaneously (fixed-point iteration, which converges because
the per-node outgoing weight is < 1 for beta * c = 2; convergence is checked).
Partition functions depend only on (hypothesis content, task context, beta, W), so they
are cached on disk and shared by all experiments.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Dict, Optional, Sequence

import time as _time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from abstractions.base import Batch
from environment.shopping_domain import ShoppingDomain, Task
from environment.trajectories import Trajectory
from norms.hypotheses import Hypothesis
from norms.violations import count_violations


def fixed_point(M0, B0, M1=None, B1=None, ew=None, tol=1e-13, maxit=20000):
    """Solve z = B0 + M0 z  (+ ew*(B1 + M1 z)) column-wise for each scale in ew."""
    if ew is None:
        z = B0.copy()
        for it in range(maxit):
            zn = B0 + M0 @ z
            if np.max(np.abs(zn - z) / np.maximum(np.abs(zn), 1e-300)) <= tol:
                return zn
            z = zn
        raise RuntimeError("fixed point did not converge")
    B = B0[:, None] + B1[:, None] * ew[None, :]
    z = B.copy()
    for it in range(maxit):
        zn = B + M0 @ z
        if M1 is not None and M1.nnz:
            zn += (M1 @ z) * ew[None, :]
        if np.max(np.abs(zn - z) / np.maximum(np.abs(zn), 1e-300)) <= tol:
            return zn
        z = zn
    raise RuntimeError("fixed point did not converge")


def solve_direct(M, B):
    """Solve z = B + M z exactly by sparse LU of (I - M)."""
    A = (sp.identity(M.shape[0], format="csc") - M).tocsc()
    return spl.splu(A).solve(np.asarray(B, dtype=float))


class ProductSystem:
    """Weighted product graph (task graph x norm monitor) for one hypothesis and task."""

    def __init__(self, domain: ShoppingDomain, task: Task, hyp: Optional[Hypothesis], beta: float,
                 edge_mask: Optional[np.ndarray] = None):
        g = domain.graph(task)
        self.g = g
        if hyp is None:
            K, init = 1, 0
            nxt = [np.zeros(g.n_edges, dtype=np.int64)]
            viol = [np.zeros(g.n_edges, dtype=np.int64)]
        else:
            batch = Batch.from_graph(domain, g)
            K, init, nxt, viol = hyp.norm.monitor(hyp.abstraction, batch)
        self.K, self.init = K, init
        n = g.n_states * K
        self.n = n
        wgt = np.exp(-beta * 1.0)
        keep = np.ones(g.n_edges, dtype=bool) if edge_mask is None else edge_mask
        rows0, cols0, rows1, cols1, b0, b1 = [], [], [], [], np.zeros(n), np.zeros(n)
        for m in range(K):
            src = g.E_src * K + m
            term = (g.E_dst < 0) & keep
            nont = (g.E_dst >= 0) & keep
            dst = g.E_dst * K + nxt[m]
            v = viol[m].astype(bool)
            rows0.append(src[nont & ~v]); cols0.append(dst[nont & ~v])
            rows1.append(src[nont & v]); cols1.append(dst[nont & v])
            np.add.at(b0, src[term & ~v], wgt)
            np.add.at(b1, src[term & v], wgt)
        r0, c0 = np.concatenate(rows0), np.concatenate(cols0)
        r1, c1 = np.concatenate(rows1), np.concatenate(cols1)
        self.M0 = sp.csr_matrix((np.full(len(r0), wgt), (r0, c0)), shape=(n, n))
        self.M1 = sp.csr_matrix((np.full(len(r1), wgt), (r1, c1)), shape=(n, n))
        self.b0, self.b1 = b0, b1
        self.start = 0 * K + init

    def z(self, W: np.ndarray, beta: float, max_rank: int = 400) -> np.ndarray:
        """z over all product nodes for each w in W -> array (n, len(W)).

        Exact solution of (I - M0 - e^{-beta w} M1) z = b0 + e^{-beta w} b1 for every w: one sparse LU of
        (I - M0) plus a Woodbury low-rank correction over the (few) states that have violating
        transitions; falls back to vectorised fixed-point iteration if that set is large.
        """
        ew = np.exp(-beta * np.asarray(W, dtype=float))
        rows = np.unique(self.M1.nonzero()[0])
        if len(rows) <= max_rank:
            A0 = (sp.identity(self.n, format="csc") - self.M0).tocsc()
            lu = spl.splu(A0)
            x0 = lu.solve(self.b0)
            y0 = lu.solve(self.b1) if self.b1.any() else np.zeros(self.n)
            X = x0[:, None] + y0[:, None] * ew[None, :]           # A0^{-1}(b0 + e b1)
            k = len(rows)
            if k == 0:
                return X
            U = np.zeros((self.n, k))
            U[rows, np.arange(k)] = 1.0
            AU = lu.solve(U)                                       # n x k
            V = self.M1[rows, :]                                   # k x n
            VAU = np.asarray(V @ AU)
            VX = np.asarray(V @ X)                                 # k x J
            out = np.empty_like(X)
            for j, e in enumerate(ew):
                c = np.linalg.solve(np.eye(k) - e * VAU, VX[:, j])
                out[:, j] = X[:, j] + e * (AU @ c)
            return out
        if self.M1.nnz == 0:
            # violations only on terminal transitions: Z(w) = u + e^{-beta w} v
            M = self.M0
            u = fixed_point(M, self.b0)
            v = fixed_point(M, self.b1) if self.b1.any() else np.zeros_like(u)
            return u[:, None] + v[:, None] * ew[None, :]
        return fixed_point(self.M0, self.b0, self.M1, self.b1, ew)


class ZComputer:
    """log Z(h, w, s0, g) for all w in W, with in-memory and on-disk caching."""

    def __init__(self, domain: ShoppingDomain, beta: float, W: Sequence[float], cache_dir: Optional[str] = None):
        self.domain, self.beta = domain, float(beta)
        self.W = np.asarray(W, dtype=float)
        self.cache_dir = cache_dir
        self._mem: Dict[str, np.ndarray] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.n_computed = 0

    def _key(self, hyp: Optional[Hypothesis], task: Task) -> str:
        payload = json.dumps({"domain": self.domain.name, "layout": self.domain.layout.ascii(),
                              "exit_requires_cart": self.domain.exit_requires_cart,
                              "items": self.domain.item_names,
                              "hyp": None if hyp is None else hyp.key, "task": task.key, "beta": self.beta,
                              "W": [round(float(w), 6) for w in self.W]}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()

    def logZ(self, hyp: Optional[Hypothesis], task: Task) -> np.ndarray:
        k = self._key(hyp, task)
        hit = self._mem.get(k)
        if hit is not None:
            return hit
        path = os.path.join(self.cache_dir, k + ".npy") if self.cache_dir else None
        if path and os.path.exists(path):
            try:
                val = np.load(path)
                self._mem[k] = val
                return val
            except Exception:
                pass
        lock = None
        if path:   # de-duplicate concurrent computation of the same key across worker processes
            try:
                lock = os.open(path + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                t0 = _time.time()
                while _time.time() - t0 < 600 and os.path.exists(path + ".lock") and not os.path.exists(path):
                    _time.sleep(0.05)
                if os.path.exists(path):
                    try:
                        val = np.load(path)
                        self._mem[k] = val
                        return val
                    except Exception:
                        pass
        try:
            ps = ProductSystem(self.domain, task, hyp, self.beta)
            z = ps.z(self.W, self.beta)
            val = np.log(z[ps.start, :])
            self.n_computed += 1
            self._mem[k] = val
            if path:
                fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp.npy")
                os.close(fd)
                np.save(tmp, val)
                os.replace(tmp, path)
        finally:
            if lock is not None:
                os.close(lock)
                try:
                    os.remove(path + ".lock")
                except FileNotFoundError:
                    pass
        return val

    def loglik(self, hyp: Hypothesis, traj: Trajectory) -> np.ndarray:
        """log P(tau | h, w) for every w in W."""
        V = count_violations(self.domain, hyp, traj)
        return -self.beta * (traj.cost + self.W * V) - self.logZ(hyp, traj.task)


# ------------------------------------------------------------------------------------- sampling
def sample_maxent(domain: ShoppingDomain, task: Task, hyp: Optional[Hypothesis], w: float, beta: float,
                  rng: np.random.Generator, n: int = 1, max_len: int = 400) -> list:
    """Draw trajectories from P(tau | h, w) by forward sampling with exact z (soft-optimal policy)."""
    ps = ProductSystem(domain, task, hyp, beta)
    z = ps.z(np.array([w]), beta)[:, 0]
    g, K = ps.g, ps.K
    batch = Batch.from_graph(domain, g) if hyp is not None else None
    if hyp is not None:
        _, _, nxt, viol = hyp.norm.monitor(hyp.abstraction, batch)
    else:
        nxt = [np.zeros(g.n_edges, dtype=np.int64)]
        viol = [np.zeros(g.n_edges, dtype=np.int64)]
    order = np.argsort(g.E_src, kind="stable")
    starts = np.searchsorted(g.E_src[order], np.arange(g.n_states + 1))
    wgt = np.exp(-beta)
    out = []
    for _ in range(n):
        s, m = 0, ps.init
        acts = []
        while True:
            es = order[starts[s]:starts[s + 1]]
            p = np.empty(len(es))
            for i, e in enumerate(es):
                ev = wgt * np.exp(-beta * w * viol[m][e])
                p[i] = ev * (1.0 if g.E_dst[e] < 0 else z[g.E_dst[e] * K + nxt[m][e]])
            p /= p.sum()
            e = es[rng.choice(len(es), p=p)]
            acts.append(int(g.E_act[e]))
            if g.E_dst[e] < 0 or len(acts) >= max_len:
                break
            s, m = int(g.E_dst[e]), int(nxt[m][e])
        if g.E_dst[e] < 0:
            out.append(Trajectory(task=task, actions=acts, name="sample"))
    return out
