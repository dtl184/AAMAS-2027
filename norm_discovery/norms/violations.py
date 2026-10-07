"""Violation counting V_h(tau) and abstract-trajectory construction alpha(tau)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from abstractions.base import Batch
from environment.trajectories import Trajectory
from norms.hypotheses import Hypothesis


def trajectory_batch(domain, traj: Trajectory) -> Batch:
    cache = traj._cache
    b = cache.get(("batch", domain.name))
    if b is None:
        b = Batch.from_states(domain, traj.states(domain), traj.actions)
        cache[("batch", domain.name)] = b
    return b


def violation_steps(domain, hyp: Hypothesis, traj: Trajectory) -> List[int]:
    """Indices k of transitions (s_k, a_k, s_k+1) on which a violation is counted."""
    batch = trajectory_batch(domain, traj)
    K, m, nxt, viol = hyp.norm.monitor(hyp.abstraction, batch)
    out = []
    for k in range(batch.n):
        if viol[m][k]:
            out.append(k)
        m = int(nxt[m][k])
    return out


def count_violations(domain, hyp: Hypothesis, traj: Trajectory) -> int:
    key = ("V", domain.name, hyp.key)
    v = traj._cache.get(key)
    if v is None:
        v = len(violation_steps(domain, hyp, traj))
        traj._cache[key] = v
    return v


def abstract_trajectory(domain, hyp: Hypothesis, traj: Trajectory) -> List[Dict]:
    """tau_bar = alpha(tau): per transition, the abstract state properties holding and events completing."""
    batch = trajectory_batch(domain, traj)
    a = hyp.abstraction
    props = {p: a.eval(p, batch, "pre") for p in a.props}
    events = {e: a.eval(e, batch, "pre") for e in a.events}
    out = []
    for k in range(batch.n):
        out.append({"k": k, "action": domain.action_name(int(batch.act[k])),
                    "sigma": [p for p, v in props.items() if v[k]],
                    "events": [e for e, v in events.items() if v[k]]})
    return out
