"""Refinement-trigger statistics and rules (used ONLY to decide when to refine; the posterior is unchanged).

Statistics for a new demonstration tau_t, computed before tau_t is added to the posterior:

  raw   log p_t = log P(tau_t | D_1:t-1, M) = logsumexp_{h,w} [log P(h,w|D) - beta C(tau) - beta w V_h(tau)
                                                               - log Z_h,w(s0,g)]
  len   log p_t / |tau_t|                      (average log predictive per action)
  task  log p_t + beta C_task(tau_t)
        Exact: C_task(tau) is identical in every (h, w) term of the mixture, so this equals
        logsumexp_{h,w}[log P(h,w|D) - beta w V_h(tau) - log Z_h,w(s0,g)].  It no longer depends on the path
        except through the violation counts V_h(tau); it still depends on the task context through Z.
  rel   log p_t - log P_task(tau_t),   P_task(tau) = exp(-beta C(tau)) / Z_0(s0,g)
        P_task is the max-ent model with no norm (w = 0) of the same planning domain.  It is a FIXED reference:
        it does not depend on D.  rel = task + log Z_0(s0,g)
            = logsumexp_{h,w}[log P(h,w|D) - beta w V_h(tau) + log(Z_0 / Z_h,w)],
        i.e. the log-likelihood advantage of the current normative model over task cost alone.  For a
        demonstration that violates no hypothesis it is the posterior-averaged size-principle gain, >= 0.

Rules (applied to the chosen statistic s_t; "accepted" = previous demonstrations that did not trigger):
  ratio     s_t < b_t + margin,  b_t = running minimum (or median) of accepted scores.  For family "raw" with
            margin = log(rho) this is exactly the paper's p_t < rho * b_t.
  quantile  s_t < Q_q(accepted scores)  (linear-interpolation empirical quantile; with one accepted score it is
            that score).  Scale-free: the same q is meaningful for every statistic and domain.
  never     no refinement.
As in Algorithm 1, nothing can trigger at t = 2 (no accepted score exists before t = 2).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

FAMILIES = ("raw", "len", "task", "rel")
FAMILY_NAMES = {"raw": "A. raw predictive", "len": "B. length-normalised", "task": "C. task-cost-adjusted",
                "rel": "D. norm-relative (vs task-only)"}


def trigger_scores(zc, traj, log_p: float, beta: float) -> Dict[str, float]:
    C = traj.cost
    task = log_p + beta * C
    logZ0 = float(zc.logZ(None, traj.task)[0])     # no-norm model; identical for every w
    return {"raw": log_p, "len": log_p / max(len(traj.actions), 1), "task": task, "rel": task + logZ0,
            "log_p_task": -beta * C - logZ0}


def should_trigger(rule: str, param: Optional[float], score: float, accepted: List[float], baseline: str = "min"):
    """Return (triggered, baseline_value, threshold_value)."""
    if rule == "never" or not accepted:
        return False, None, None
    if rule == "ratio":
        b = min(accepted) if baseline == "min" else float(np.median(accepted))
        thr = b + param
        return score < thr, b, thr
    if rule == "quantile":
        thr = float(np.quantile(accepted, param))
        return score < thr, thr, thr
    raise ValueError(rule)
