"""Domain specifications for the two reconstructed experiments.

A DomainSpec bundles: the planning model, the vocabulary levels (alpha_0 and the refinement
vocabulary), the ground-truth (hidden) norm, the original three training demonstrations,
the held-out evaluation set, a task pool for larger generated training sets, and the
generators for clean / behaviourally-noisy / norm-violating demonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from norms.hypotheses import Hypothesis
from norms.violations import count_violations
from proposals import templates as T

from .layouts import exp1_layout, exp2_layout
from .shopping_domain import ACT_EXIT, ACT_LEAVE_CART, ACT_PICKUP_CART, CART_HELD, ShoppingDomain, Task
from .trajectories import ScriptBuilder, Trajectory, is_valid, plan_optimal

MLCI_FAMILIES = ("coords", "cart_possession", "cart_location", "hand", "actions_coarse", "actions_primitive")


@dataclass
class DomainSpec:
    key: str                                  # "cart" (Exp. 1) or "aisle" (Exp. 2)
    domain: ShoppingDomain
    true_hypothesis: Hypothesis               # hidden norm used to generate / label data
    intended_id: str                          # id of the intended hypothesis in the proposal set
    training: List[Trajectory]
    heldout: List[Trajectory]
    unseen_names: List[str] = field(default_factory=list)
    pool_tasks: List[Task] = field(default_factory=list)
    violation_tasks: List[Task] = field(default_factory=list)
    notes: Dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.domain.name

    # -------------------------------------------------------------- data generation
    def clean_demo(self, task: Task, rng: np.random.Generator, name: str = "") -> Trajectory:
        t = plan_optimal(self.domain, task, self.true_hypothesis, rng=rng, name=name)
        t.meta.update({"kind": "clean"})
        t.label = False
        return t

    def violates(self, t: Trajectory) -> bool:
        return count_violations(self.domain, self.true_hypothesis, t) > 0

    def behavioural_noise_demo(self, task: Task, rng: np.random.Generator, name: str = "",
                               max_tries: int = 200) -> Trajectory:
        """Norm-compliant but inefficient: random out-and-back detours and redundant leave/pickup pairs."""
        base = self.clean_demo(task, rng)
        d = self.domain
        for _ in range(max_tries):
            acts = list(base.actions)
            n_detours = int(rng.integers(1, 4))
            for _k in range(n_detours):
                i = int(rng.integers(0, len(acts)))      # insert before action i (never after EXIT)
                states = Trajectory(task=task, actions=acts[:i]).states(d)
                s = states[-1]
                walk = []
                L = int(rng.integers(2, 7))
                for _ in range(L):
                    moves = [a for a, nxt in d.successors(s, task) if a <= 3]
                    a = int(moves[rng.integers(len(moves))])
                    walk.append(a)
                    s = d.step(s, a, task)
                back = [{0: 1, 1: 0, 2: 3, 3: 2}[a] for a in reversed(walk)]
                acts = acts[:i] + walk + back + acts[i:]
            if rng.random() < 0.5:   # repeated harmless interaction: leave the cart and immediately take it back
                states = Trajectory(task=task, actions=acts).states(d)
                held = [k for k in range(len(acts)) if states[k][1] == CART_HELD and acts[k] != ACT_EXIT]
                if held:
                    k = int(held[rng.integers(len(held))])
                    acts = acts[:k] + [ACT_LEAVE_CART, ACT_PICKUP_CART] + acts[k:]
            t = Trajectory(task=task, actions=acts, name=name, label=False, meta={"kind": "behavioural_noise"})
            if is_valid(d, t)[0] and not self.violates(t):
                return t
        raise RuntimeError("could not generate a compliant noisy demonstration")

    def violation_demo(self, rng: np.random.Generator, name: str = "", max_tries: int = 200) -> Trajectory:
        d = self.domain
        for _ in range(max_tries):
            task = self.violation_tasks[int(rng.integers(len(self.violation_tasks)))]
            if self.key == "aisle":
                t = plan_optimal(d, task, None, rng=rng)      # takes the cart straight into the aisle
                vtype = "cart_into_aisle"
            else:
                vtype = ["no_return", "wrong_location", "reacquire"][int(rng.integers(3))]
                if vtype == "no_return":
                    t = plan_optimal(d, task, None, rng=rng)
                elif vtype == "wrong_location":
                    base = plan_optimal(d, task, None, rng=rng)
                    states = base.states(d)
                    last_item = max(k for k, a in enumerate(base.actions) if a >= 7)
                    ret = d.layout.cell_index[d.layout.return_area]
                    cands = [k for k in range(last_item + 1, len(base.actions))
                             if states[k][1] == CART_HELD and states[k][0] != ret]
                    k = int(cands[rng.integers(len(cands))])
                    t = Trajectory(task=task, actions=base.actions[:k] + [ACT_LEAVE_CART] + base.actions[k:])
                else:
                    base = plan_optimal(d, task, self.true_hypothesis, rng=rng)
                    k = max(i for i, a in enumerate(base.actions) if a == ACT_LEAVE_CART)
                    t = Trajectory(task=task, actions=base.actions[:k + 1] + [ACT_PICKUP_CART] + base.actions[k + 1:])
            t.name, t.label, t.meta = name, True, {"kind": "violation", "violation_type": vtype}
            if is_valid(d, t)[0] and self.violates(t):
                return t
        raise RuntimeError("could not generate a violating demonstration")

    def sample_pool_task(self, rng: np.random.Generator) -> Task:
        return self.pool_tasks[int(rng.integers(len(self.pool_tasks)))]


# ====================================================================================== Exp 1
def make_cart_spec() -> DomainSpec:
    layout = exp1_layout()
    domain = ShoppingDomain(
        "cart", layout, exit_requires_cart=False,
        vocab_levels={0: ("coords", "actions_coarse"),
                      1: ("coords", "actions_coarse", "cart_possession", "cart_location", "hand",
                          "actions_primitive")},
        mlci_families=MLCI_FAMILIES)
    R = layout.return_area
    true_h = Hypothesis.from_dict(next(h for h in T.exp1_level1(R) if h["id"] == "H_int"))
    true_h.hid = "TRUE"
    mk = domain.make_task

    train = [plan_optimal(domain, mk(items), true_h, name=f"D{i + 1}")
             for i, items in enumerate([("apple", "milk"), ("bread", "eggs"), ("apple", "bread", "milk")])]
    for t in train:
        t.label = False
        t.meta["kind"] = "clean"

    C = layout.cart_station
    H = []

    def sb(items):
        return ScriptBuilder(domain, mk(items))
    # compliant (different items and item orders)
    H.append(sb(["milk", "apple"]).goto(C).pickup_cart().pick("milk").pick("apple").goto(R).leave_cart().exit()
             .build("C1_milk_apple", False, kind="compliant"))
    H.append(sb(["bread", "milk"]).goto(C).pickup_cart().pick("bread").pick("milk").goto(R).leave_cart().exit()
             .build("C2_bread_milk", False, kind="compliant"))
    H.append(sb(["eggs", "apple"]).goto(C).pickup_cart().pick("eggs").pick("apple").goto(R).leave_cart().exit()
             .build("C3_eggs_apple", False, kind="compliant"))
    H.append(sb(["apple", "bread", "eggs"]).goto(C).pickup_cart().pick("eggs").pick("apple").pick("bread")
             .goto(R).leave_cart().exit().build("C4_eggs_apple_bread", False, kind="compliant"))
    # violations: no return
    H.append(sb(["apple", "milk"]).goto(C).pickup_cart().pick("apple").pick("milk").exit()
             .build("V1_no_return", True, kind="no_return"))
    H.append(sb(["bread", "eggs"]).goto(C).pickup_cart().pick("eggs").pick("bread").exit()
             .build("V2_no_return", True, kind="no_return"))
    # violations: cart left at the wrong location
    H.append(sb(["milk", "bread"]).goto(C).pickup_cart().pick("milk").pick("bread").goto((8, 5)).leave_cart().exit()
             .build("V3_wrong_location_near_exit", True, kind="wrong_location"))
    H.append(sb(["apple", "eggs"]).goto(C).pickup_cart().pick("apple").pick("eggs").goto((5, 1)).leave_cart().exit()
             .build("V4_wrong_location_mid_store", True, kind="wrong_location"))
    # violations: return, then pick the cart up again before exiting
    H.append(sb(["apple", "milk"]).goto(C).pickup_cart().pick("apple").pick("milk").goto(R).leave_cart()
             .pickup_cart().exit().build("V5_return_then_reacquire", True, kind="reacquire"))
    H.append(sb(["bread", "eggs"]).goto(C).pickup_cart().pick("bread").pick("eggs").goto(R).leave_cart()
             .pickup_cart().exit().build("V6_return_then_reacquire", True, kind="reacquire"))

    items = domain.item_names
    import itertools
    pool = [mk(c) for k in (1, 2, 3) for c in itertools.combinations(items, k)]
    vtasks = [mk(c) for k in (2, 3) for c in itertools.combinations(items, k)]
    spec = DomainSpec("cart", domain, true_h, "H_int", train, H, [], pool, vtasks,
                      notes={"return_area": R, "cart_station": C})
    _check_labels(spec)
    return spec


# ====================================================================================== Exp 2
def make_aisle_spec() -> DomainSpec:
    layout = exp2_layout()
    domain = ShoppingDomain(
        "aisle", layout, exit_requires_cart=True,
        vocab_levels={0: ("coords", "cart_possession", "actions_coarse"),
                      1: ("coords", "cart_possession", "actions_coarse", "geometry")},
        mlci_families=MLCI_FAMILIES)
    true_h = Hypothesis.from_dict(next(h for h in T.exp2_level1() if h["id"] == "S_inAisle"))
    true_h.hid = "TRUE"
    mk = domain.make_task
    train = [plan_optimal(domain, mk(items), true_h, name=f"D{i + 1}")
             for i, items in enumerate([("a1",), ("a2",), ("s1",)])]
    for t in train:
        t.label = False
        t.meta["kind"] = "clean"
    C = layout.cart_station
    mouth = {"A1": (3, 3), "A2": (6, 3), "A3": (9, 3)}
    pickc = {"a1": (3, 1), "a2": (6, 1), "a3": (9, 1)}
    aisle_cells = [c for cs in layout.aisles.values() for c in cs]

    def sb(items):
        return ScriptBuilder(domain, mk(items))

    def fetch(b, item, aisle):          # compliant: park at the aisle mouth, walk in, come back
        return b.goto(mouth[aisle], avoid=aisle_cells).leave_cart().pick(item, pickc[item]) \
            .goto(mouth[aisle]).pickup_cart()

    H = []
    b = sb(["s1", "a1"]).goto(C).pickup_cart().pick("s1", avoid=aisle_cells)
    H.append(fetch(b, "a1", "A1").exit(avoid=aisle_cells).build("H1_A1_compliant", False, aisles=["A1"]))
    H.append(sb(["a1"]).goto(C).pickup_cart().pick("a1", pickc["a1"]).exit()
             .build("H2_A1_cart_in_aisle", True, aisles=["A1"]))
    b = fetch(sb(["a2", "s2"]).goto(C).pickup_cart(), "a2", "A2")
    H.append(b.pick("s2", avoid=aisle_cells).exit(avoid=aisle_cells).build("H3_A2_compliant", False, aisles=["A2"]))
    H.append(sb(["a2"]).goto(C).pickup_cart().pick("a2", pickc["a2"]).exit()
             .build("H4_A2_cart_in_aisle", True, aisles=["A2"]))
    H.append(fetch(sb(["a3"]).goto(C).pickup_cart(), "a3", "A3").exit(avoid=aisle_cells)
             .build("H5_A3_compliant", False, aisles=["A3"]))
    H.append(sb(["a3"]).goto(C).pickup_cart().pick("a3", pickc["a3"]).exit()
             .build("H6_A3_cart_deep_in_aisle", True, aisles=["A3"]))
    H.append(sb(["a3"]).goto(C).pickup_cart().goto((9, 2)).leave_cart().pick("a3", pickc["a3"]).goto((9, 2))
             .pickup_cart().exit(avoid=aisle_cells).build("H7_A3_cart_parked_inside_aisle", True, aisles=["A3"]))
    H.append(sb(["s1", "s2"]).goto(C).pickup_cart().pick("s1", avoid=aisle_cells).pick("s2", avoid=aisle_cells)
             .exit(avoid=aisle_cells).build("H8_standalone_with_cart", False, aisles=[]))
    b = fetch(sb(["a1", "a3"]).goto(C).pickup_cart(), "a1", "A1")
    H.append(fetch(b, "a3", "A3").exit(avoid=aisle_cells).build("H9_A1_A3_compliant", False, aisles=["A1", "A3"]))
    b = fetch(sb(["a2", "a3"]).goto(C).pickup_cart(), "a2", "A2")
    H.append(b.goto((9, 3)).pick("a3", pickc["a3"]).goto((9, 3)).exit(avoid=aisle_cells)
             .build("H10_A2_ok_A3_cart_in_aisle", True, aisles=["A2", "A3"]))
    unseen = [h.name for h in H if "A3" in h.meta["aisles"]]

    import itertools
    train_items = ["a1", "a2", "s1", "s2"]          # a3 / aisle A3 is never used for training
    pool = [mk(c) for k in (1, 2) for c in itertools.combinations(train_items, k)]
    vtasks = [t for t in pool if any(i in ("a1", "a2") for i in t.items)]
    spec = DomainSpec("aisle", domain, true_h, "S_inAisle", train, H, unseen, pool, vtasks,
                      notes={"aisles": layout.aisles, "training_items": train_items})
    _check_labels(spec)
    return spec


def _check_labels(spec: DomainSpec) -> None:
    """Held-out labels are fixed by construction; verify them against the hidden norm."""
    for t in spec.training + spec.heldout:
        ok, msg = is_valid(spec.domain, t)
        assert ok, (t.name, msg)
        v = count_violations(spec.domain, spec.true_hypothesis, t) > 0
        assert v == bool(t.label), f"label mismatch for {t.name}: scripted {t.label}, hidden norm says {v}"


_SPECS: Dict[str, DomainSpec] = {}


def get_spec(key: str) -> DomainSpec:
    if key not in _SPECS:
        _SPECS[key] = {"cart": make_cart_spec, "aisle": make_aisle_spec}[key]()
    return _SPECS[key]
