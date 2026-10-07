"""Common interface for hypothesis proposal (the role played by the LLM in the paper)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from abstractions.base import FAMILIES, Batch, atom_family
from environment.shopping_domain import CART_HELD, CART_STATION
from environment.trajectories import Trajectory
from norms.hypotheses import Hypothesis


class ProposalProvider(ABC):
    """Proposes executable abstraction-norm hypotheses.

    propose_initial: norms over the initial abstraction alpha_0 only (may not change the representation).
    refine:          triggered by the learner; exposes the next vocabulary level and proposes refined
                     abstractions together with norms over them.
    Proposals are only *candidates*: the learner validates them and Bayesian inference decides.
    """

    name = "base"

    @abstractmethod
    def propose_initial(self, spec, demos: Sequence[Trajectory]) -> List[dict]:
        ...

    @abstractmethod
    def refine(self, spec, demos: Sequence[Trajectory], current: Sequence[Hypothesis], level: int) -> List[dict]:
        ...


def render_label(domain, traj: Trajectory, families: Sequence[str]) -> List[str]:
    """Symbolic view of a trajectory restricted to a vocabulary (what a proposer is shown)."""
    states = traj.states(domain)
    L = domain.layout
    lines = []
    for k, a in enumerate(traj.actions + [None]):
        pos, cart, mask, hand = states[k]
        props = []
        if "coords" in families:
            props.append("at(%d,%d)" % L.free_cells[pos])
        if "cart_possession" in families and cart == CART_HELD:
            props.append("hasCart")
        if "cart_location" in families and cart not in (CART_HELD,):
            c = L.cart_station if cart == CART_STATION else L.free_cells[cart]
            props.append("cartAt(%d,%d)" % c)
        if "hand" in families and hand:
            props.append("handFull")
        if "geometry" in families:
            for g in ("adjShelf", "shelfN", "shelfS", "shelfE", "shelfW"):
                if L.geometry[g][pos]:
                    props.append(g)
            props.append(f"freeWidthH={L.geometry['freeWidthH'][pos]}")
            props.append(f"freeWidthV={L.geometry['freeWidthV'][pos]}")
        if a is None:
            act = "<terminal>"
        else:
            name = domain.action_name(a)
            if name.startswith("PICKUP_ITEM"):
                name = "PICKUP_ITEM"
            if name in ("PICKUP_CART", "LEAVE_CART", "PICKUP_ITEM"):
                act = name if "actions_primitive" in families else "INTERACT"
            else:
                act = name
        lines.append(f"{k:3d}: {{{', '.join(props)}}} {act}")
    return lines


def vocabulary_description(families: Sequence[str]) -> str:
    return "\n".join(f"- {f}: {FAMILIES[f]}" for f in families)
