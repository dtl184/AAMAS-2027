"""Hypothesis templates (JSON) shared by the deterministic proposal provider and the domain specs.

Every hypothesis is a JSON object {id, level, template, text, abstraction, norm}; it is parsed by
norms.hypotheses.Hypothesis.from_dict and validated against the vocabulary level it was proposed at.
An LLM provider would return objects of exactly the same shape (proposals/llm_stub.py).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def at(c) -> str:
    return f"at({c[0]},{c[1]})"


def any_cell(cells: Sequence) -> object:
    cells = sorted({tuple(c) for c in cells})
    return at(cells[0]) if len(cells) == 1 else {"or": [at(c) for c in cells]}


def null_hypothesis(level: int = 0) -> dict:
    return {"id": "H_null", "level": level, "template": "null", "text": "no norm",
            "abstraction": {}, "norm": {"type": "prohibition", "condition": {"or": []}}}


# ------------------------------------------------------------------------------ Experiment 1: cart use
# Event definitions follow Eq. (12): a pre-label formula followed by a post-state formula.
def cart_events(return_cell) -> dict:
    ev = {
        "pickUpCart": {"pre": {"and": ["INTERACT", {"not": "hasCart"}]}, "post": "hasCart"},
        "leaveCart": {"pre": {"and": ["INTERACT", "hasCart"]}, "post": {"not": "hasCart"}},
        "pickUpItem": {"pre": "PICKUP_ITEM", "post": None},
    }
    if return_cell is not None:
        ev["returnCart"] = {"pre": {"and": ["INTERACT", "hasCart", "atReturn"]}, "post": {"not": "hasCart"}}
    return ev


def _abs(events_used: Sequence[str], props_used: Sequence[str], return_cell) -> dict:
    ev = cart_events(return_cell)
    props = {"atReturn": at(return_cell)} if return_cell is not None else {}
    return {"state_properties": {p: props[p] for p in props_used},
            "events": {e: ev[e] for e in events_used}}


def exp1_level1(return_cell) -> List[dict]:
    """Refined (level-1) hypotheses for the cart domain.  H_int is the intended norm."""
    out = []
    if return_cell is not None:
        out += [
            {"id": "H_int", "template": "cart_triggered_return",
             "text": "after picking up a cart, the agent must return it to the return area before exiting",
             "abstraction": _abs(["pickUpCart", "returnCart"], ["atReturn"], return_cell),
             "norm": {"type": "obligation", "trigger": "pickUpCart", "discharge": "returnCart"}},
            {"id": "H_uncond", "template": "unconditional_return",
             "text": "a cart must be returned to the return area before exiting (unconditional)",
             "abstraction": _abs(["returnCart"], ["atReturn"], return_cell),
             "norm": {"type": "obligation", "trigger": "START", "discharge": "returnCart"}},
            {"id": "H_item", "template": "item_triggered_return",
             "text": "after picking up an item, the agent must return the cart before exiting",
             "abstraction": _abs(["pickUpItem", "returnCart"], ["atReturn"], return_cell),
             "norm": {"type": "obligation", "trigger": "pickUpItem", "discharge": "returnCart"}},
            {"id": "H_visit", "template": "cart_triggered_visit",
             "text": "after picking up a cart, the agent must visit the return area before exiting",
             "abstraction": _abs(["pickUpCart"], ["atReturn"], return_cell),
             "norm": {"type": "obligation", "trigger": "pickUpCart", "discharge": "atReturn"}},
        ]
    out += [
        {"id": "H_leave", "template": "cart_triggered_leave_anywhere",
         "text": "after picking up a cart, the agent must leave it somewhere before exiting",
         "abstraction": _abs(["pickUpCart", "leaveCart"], [], return_cell),
         "norm": {"type": "obligation", "trigger": "pickUpCart", "discharge": "leaveCart"}},
        {"id": "H_noexit", "template": "no_exit_with_cart",
         "text": "the agent may not exit while holding a cart",
         "abstraction": {}, "norm": {"type": "prohibition", "condition": {"and": ["EXIT", "hasCart"]}}},
    ]
    for h in out:
        h["level"] = 1
    return out


def exp1_level0(first_cell, last_cell) -> List[dict]:
    """Initial (alpha_0: coordinates + coarse actions) hypotheses instantiated from the first demo."""
    out = [null_hypothesis(0)]
    if last_cell is None:
        return out
    ia = {"and": ["INTERACT", at(last_cell)]}
    out += [
        {"id": "A0_interact_last", "template": "interact_at_last_cell",
         "text": f"the agent must interact at {tuple(last_cell)} before exiting",
         "abstraction": {}, "norm": {"type": "obligation", "trigger": "START", "discharge": ia}},
        {"id": "A0_visit_last", "template": "visit_last_cell",
         "text": f"the agent must visit {tuple(last_cell)} before exiting",
         "abstraction": {}, "norm": {"type": "obligation", "trigger": "START", "discharge": at(last_cell)}},
        {"id": "A0_any_to_last", "template": "interaction_then_last",
         "text": f"after any interaction, the agent must later interact at {tuple(last_cell)}",
         "abstraction": {}, "norm": {"type": "obligation", "trigger": "INTERACT", "discharge": ia}},
    ]
    if first_cell is not None and tuple(first_cell) != tuple(last_cell):
        out.append({"id": "A0_first_to_last", "template": "first_interaction_then_last",
                    "text": f"after interacting at {tuple(first_cell)}, the agent must later interact at "
                            f"{tuple(last_cell)}",
                    "abstraction": {},
                    "norm": {"type": "obligation", "trigger": {"and": ["INTERACT", at(first_cell)]},
                             "discharge": ia}})
    for h in out:
        h["level"] = 0
    return out


# ------------------------------------------------------------------------------ Experiment 2: aisle use
IN_AISLE_DEF = {"or": [{"and": ["shelfW", "shelfE"]}, {"and": ["shelfN", "shelfS"]}]}
CORRIDOR_DEF = {"or": [{"le": ["freeWidthH", 1]}, {"le": ["freeWidthV", 1]}]}


def exp2_level1(memorised_cells: Optional[Sequence] = None) -> List[dict]:
    out = [
        {"id": "S_inAisle", "template": "cart_in_bounded_region",
         "text": "the agent may not hold its cart in a cell bounded by shelves on opposing sides (an aisle)",
         "abstraction": {"state_properties": {"inAisle": IN_AISLE_DEF}},
         "norm": {"type": "prohibition", "condition": {"and": ["hasCart", "inAisle"]}}},
        {"id": "S_corridor", "template": "cart_in_narrow_corridor",
         "text": "the agent may not hold its cart in a corridor of free width <= 1",
         "abstraction": {"state_properties": {"corridor": CORRIDOR_DEF}},
         "norm": {"type": "prohibition", "condition": {"and": ["hasCart", "corridor"]}}},
        {"id": "S_nearShelf", "template": "cart_near_shelf",
         "text": "the agent may not hold its cart next to any shelf",
         "abstraction": {}, "norm": {"type": "prohibition", "condition": {"and": ["hasCart", "adjShelf"]}}},
        {"id": "S_noShelfInteractWithCart", "template": "no_shelf_interaction_with_cart",
         "text": "the agent may not interact with a shelf while holding its cart",
         "abstraction": {},
         "norm": {"type": "prohibition", "condition": {"and": ["INTERACT", "hasCart", "adjShelf"]}}},
    ]
    if memorised_cells:
        out.append({"id": "S_memorised", "template": "cart_not_in_visited_cells",
                    "text": "the agent may not hold its cart in the cells demonstrators visited without it",
                    "abstraction": {},
                    "norm": {"type": "prohibition", "condition": {"and": ["hasCart", any_cell(memorised_cells)]}}})
    for h in out:
        h["level"] = 1
    return out


def exp2_level0(cartless_cells: Sequence, cartless_pick_cells: Sequence) -> List[dict]:
    out = [null_hypothesis(0)]
    if cartless_cells:
        out.append({"id": "B0_cells", "template": "cart_not_in_cartless_cells",
                    "text": "the agent may not hold its cart in the cells the demonstrator visited without it",
                    "abstraction": {},
                    "norm": {"type": "prohibition", "condition": {"and": ["hasCart", any_cell(cartless_cells)]}}})
    if cartless_pick_cells and sorted(map(tuple, cartless_pick_cells)) != sorted(map(tuple, cartless_cells)):
        out.append({"id": "B0_pickcells", "template": "cart_not_at_pick_cells",
                    "text": "the agent may not hold its cart where the demonstrator picked items without it",
                    "abstraction": {},
                    "norm": {"type": "prohibition",
                             "condition": {"and": ["hasCart", any_cell(cartless_pick_cells)]}}})
    if cartless_pick_cells:
        out.append({"id": "B0_interact_cells", "template": "no_cart_interaction_at_pick_cells",
                    "text": "the agent may not interact while holding its cart at the demonstrated pick cells",
                    "abstraction": {},
                    "norm": {"type": "prohibition",
                             "condition": {"and": ["INTERACT", "hasCart", any_cell(cartless_pick_cells)]}}})
    for h in out:
        h["level"] = 0
    return out
