import numpy as np
import pytest

from environment.domains import get_spec
from environment.shopping_domain import ACT_EXIT, ACT_LEAVE_CART, ACT_PICKUP_CART, CART_HELD, CART_STATION
from environment.trajectories import ScriptBuilder, Trajectory, is_valid, plan_optimal


@pytest.mark.parametrize("key", ["cart", "aisle"])
def test_all_reconstructed_trajectories_valid(key):
    spec = get_spec(key)
    for t in spec.training + spec.heldout:
        ok, msg = is_valid(spec.domain, t)
        assert ok, (t.name, msg)


def test_invalid_trajectories_rejected():
    spec = get_spec("cart")
    d = spec.domain
    task = d.make_task(["apple"])
    assert not is_valid(d, Trajectory(task, [ACT_EXIT]))[0]                     # not at exit, no items
    assert not is_valid(d, Trajectory(task, [ACT_LEAVE_CART]))[0]               # no cart held
    assert not is_valid(d, Trajectory(task, [0, 0, 0]))[0]                      # no EXIT at the end
    assert not is_valid(d, Trajectory(task, [3]))[0]                            # walk into the wall (W from x=0)
    t = spec.training[0]
    assert not is_valid(d, Trajectory(t.task, t.actions + [ACT_EXIT]))[0]       # actions after EXIT


def test_items_need_a_usable_cart():
    spec = get_spec("cart")
    d = spec.domain
    task = d.make_task(["apple"])
    s = task.start
    # walk next to the apple without a cart: no PICKUP_ITEM available
    b = ScriptBuilder(d, task).goto((1, 3))
    assert d.item_code["apple"] not in [a for a, _ in d.successors(b.s, task)]
    b = ScriptBuilder(d, task).goto(d.layout.cart_station).pickup_cart().goto((1, 3))
    assert d.item_code["apple"] in [a for a, _ in d.successors(b.s, task)]


def test_cart_mechanics():
    spec = get_spec("cart")
    d = spec.domain
    task = d.make_task(["apple"])
    b = ScriptBuilder(d, task).goto(d.layout.cart_station).pickup_cart()
    assert b.s[1] == CART_HELD
    b.goto((5, 1)).leave_cart()
    assert b.s[1] == d.layout.cell_index[(5, 1)]
    b.pickup_cart()
    assert b.s[1] == CART_HELD
    b.goto(d.layout.cart_station).leave_cart()
    assert b.s[1] == CART_STATION


def test_aisle_geometry_and_no_inAisle_primitive():
    from abstractions.base import atom_family
    spec = get_spec("aisle")
    L = spec.domain.layout
    g = L.geometry
    for cells in L.aisles.values():
        for c in cells:
            i = L.cell_index[c]
            assert g["shelfW"][i] and g["shelfE"][i]
    assert atom_family("inAisle") is None and atom_family("IN_AISLE") is None
    for lvl, fams in spec.domain.vocab_levels.items():
        assert "aisle" not in " ".join(fams).lower()


def test_optimal_demo_follows_hidden_norm():
    for key in ("cart", "aisle"):
        spec = get_spec(key)
        for t in spec.training:
            assert not spec.violates(t)
        # unconstrained optimal behaviour violates the norm in tasks that need an aisle / a cart
        t = plan_optimal(spec.domain, spec.training[0].task, None)
        assert spec.violates(t)
