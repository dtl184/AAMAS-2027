import numpy as np

from environment.domains import get_spec
from norms.hypotheses import Hypothesis
from norms.violations import abstract_trajectory, count_violations, violation_steps
from proposals import templates as T


def H(d):
    return Hypothesis.from_dict({"id": "x", **d})


def test_heldout_labels_match_hidden_norm():
    for key in ("cart", "aisle"):
        spec = get_spec(key)
        for t in spec.heldout:
            assert (count_violations(spec.domain, spec.true_hypothesis, t) > 0) == t.label, t.name


def test_cart_obligation_semantics():
    spec = get_spec("cart")
    d = spec.domain
    by = {t.name: t for t in spec.heldout}
    h = spec.true_hypothesis
    assert count_violations(d, h, by["C1_milk_apple"]) == 0
    assert count_violations(d, h, by["V1_no_return"]) == 1
    assert count_violations(d, h, by["V3_wrong_location_near_exit"]) == 1
    # re-acquisition re-opens the obligation; the violation is counted on the terminal EXIT step
    v = violation_steps(d, h, by["V5_return_then_reacquire"])
    assert v == [len(by["V5_return_then_reacquire"].actions) - 1]


def test_item_triggered_competitor_misses_reacquisition():
    spec = get_spec("cart")
    h_item = Hypothesis.from_dict(next(x for x in T.exp1_level1((0, 1)) if x["id"] == "H_item"))
    by = {t.name: t for t in spec.heldout}
    assert count_violations(spec.domain, h_item, by["V5_return_then_reacquire"]) == 0
    assert count_violations(spec.domain, h_item, by["V1_no_return"]) == 1


def test_prohibition_counts_every_step():
    spec = get_spec("aisle")
    by = {t.name: t for t in spec.heldout}
    h = spec.true_hypothesis
    # cart pushed 2 cells into A1 and back out: (3,2),(3,1) entered with cart, PICKUP at (3,1), then out
    n = count_violations(spec.domain, h, by["H2_A1_cart_in_aisle"])
    assert n >= 3
    assert count_violations(spec.domain, h, by["H1_A1_compliant"]) == 0


def test_event_detection_two_step_patterns():
    spec = get_spec("cart")
    d = spec.domain
    h = Hypothesis.from_dict(next(x for x in T.exp1_level1((0, 1)) if x["id"] == "H_int"))
    t = {t.name: t for t in spec.heldout}["V5_return_then_reacquire"]
    ab = abstract_trajectory(d, h, t)
    picks = [s["k"] for s in ab if "pickUpCart" in s["events"]]
    rets = [s["k"] for s in ab if "returnCart" in s["events"]]
    acts = [d.action_name(a) for a in t.actions]
    assert [acts[k] for k in picks] == ["PICKUP_CART", "PICKUP_CART"]
    assert [acts[k] for k in rets] == ["LEAVE_CART"]
    assert d.cell(t.states(d)[rets[0]][0]) == (0, 1)


def test_coarse_vocabulary_cannot_reference_primitives():
    spec = get_spec("cart")
    h = Hypothesis.from_dict(next(x for x in T.exp1_level1((0, 1)) if x["id"] == "H_int"))
    import pytest
    with pytest.raises(ValueError):
        h.validate(spec.domain.vocab_levels[0])
    h.validate(spec.domain.vocab_levels[1])


def test_description_length():
    h = Hypothesis.from_dict(next(x for x in T.exp1_level1((0, 1)) if x["id"] == "H_int"))
    # pickUpCart: (INTERACT ∧ ¬hasCart)·hasCart = 4+1 ; returnCart: (INTERACT∧hasCart∧atReturn)·¬hasCart = 4+2 ; atReturn = 1
    assert h.L_alpha() == 12 and h.L_norm() == 3
    assert h.L_alpha("tokens") == 14
