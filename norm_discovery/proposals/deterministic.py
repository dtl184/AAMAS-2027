"""Deterministic, reproducible stand-in for the LLM proposer.

It instantiates fixed hypothesis templates from the demonstrations it is shown, *using only the
vocabulary level it is allowed to see* (it reads the demonstrations through that vocabulary).  It
returns the intended hypothesis together with plausible competitors -- including coordinate-tied
('memorising') alternatives and alternatives tied to a different event -- so that selection is
left entirely to Bayesian inference.

Level 0 (initial abstraction): templates instantiated from the first demonstration only.
Level 1 (after a refinement trigger): templates over the exposed vocabulary, instantiated from all
demonstrations seen so far.  Further refinement calls re-instantiate the level-1 templates on the
current data (data-dependent templates may then change; duplicates are dropped by the learner).
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence

from environment.shopping_domain import ACT_LEAVE_CART, ACT_PICKUP_CART, CART_HELD
from environment.trajectories import Trajectory
from norms.hypotheses import Hypothesis

from . import templates as T
from .base import ProposalProvider


def _interaction_cells(domain, traj: Trajectory) -> List[tuple]:
    """Cells of INTERACT steps (coarse view: object interactions are indistinguishable)."""
    states = traj.states(domain)
    return [domain.cell(states[k][0]) for k, a in enumerate(traj.actions) if a in (4, 5) or a >= 7]


def _cartless_excursions(domain, traj: Trajectory):
    """Cells visited without the cart after the agent put it down (coords + hasCart view)."""
    states = traj.states(domain)
    away, picks, with_cart = set(), set(), set()
    had_cart = False
    for k, a in enumerate(traj.actions):
        pos, cart = states[k][0], states[k][1]
        holding = cart == CART_HELD
        if holding:
            had_cart = True
            with_cart.add(domain.cell(pos))
        elif had_cart:
            away.add(domain.cell(pos))
            nxt_holding = states[k + 1][1] == CART_HELD
            if (a in (4, 5) or a >= 7) and not nxt_holding:   # INTERACT that did not change hasCart
                picks.add(domain.cell(pos))
    return away, picks, with_cart


class DeterministicProposalProvider(ProposalProvider):
    name = "deterministic"
    max_level = 1

    def propose_initial(self, spec, demos: Sequence[Trajectory]) -> List[dict]:
        d = spec.domain
        tau1 = demos[0]
        if spec.key == "cart":
            cells = _interaction_cells(d, tau1)
            first, last = (cells[0], cells[-1]) if cells else (None, None)
            return T.exp1_level0(first, last)
        away, picks, with_cart = _cartless_excursions(d, tau1)
        return T.exp2_level0(sorted(away - with_cart), sorted(picks - with_cart))

    def refine(self, spec, demos: Sequence[Trajectory], current: Sequence[Hypothesis], level: int) -> List[dict]:
        d = spec.domain
        if spec.key == "cart":
            # primitive actions are now visible: where do demonstrators put the cart down?
            leaves = Counter()
            for t in demos:
                st = t.states(d)
                for k, a in enumerate(t.actions):
                    if a == ACT_LEAVE_CART:
                        leaves[d.cell(st[k][0])] += 1
            ret = None
            if leaves:
                top = max(leaves.values())
                ret = sorted(c for c, n in leaves.items() if n == top)[0]
            return T.exp1_level1(ret)
        away, with_cart = set(), set()
        for t in demos:
            a, _, w = _cartless_excursions(d, t)
            away |= a
            with_cart |= w
        return T.exp2_level1(sorted(away - with_cart))
