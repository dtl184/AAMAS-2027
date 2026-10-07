import numpy as np

from environment.domains import get_spec
from environment.trajectories import is_valid
from experiments.common import DEFAULTS, make_pool_dataset, run_method


def test_noisy_data_generation():
    for key in ("cart", "aisle"):
        spec = get_spec(key)
        clean = make_pool_dataset(spec, 20, 11, None, 0.0)
        beh = make_pool_dataset(spec, 20, 11, "behavioural", 0.2)
        vio = make_pool_dataset(spec, 20, 11, "violation", 0.3)
        assert sum(t.meta["corrupted"] for t in beh) == 4
        assert sum(t.meta["corrupted"] for t in vio) == 6
        for t in clean + beh + vio:
            assert is_valid(spec.domain, t)[0]
        assert not any(spec.violates(t) for t in clean + beh)          # behavioural noise keeps the label
        assert all(spec.violates(t) == t.meta["corrupted"] for t in vio)
        for a, b in zip(clean, beh):
            if b.meta["corrupted"]:
                assert len(b.actions) > len(a.actions)
            else:
                assert a.actions == b.actions                            # paired design
        # nested corruption sets across rates
        lo = {i for i, t in enumerate(make_pool_dataset(spec, 20, 11, "violation", 0.1)) if t.meta["corrupted"]}
        hi = {i for i, t in enumerate(vio) if t.meta["corrupted"]}
        assert lo <= hi


def test_deterministic_reproducibility():
    spec = get_spec("aisle")
    a = make_pool_dataset(spec, 12, 5, "behavioural", 0.25)
    b = make_pool_dataset(spec, 12, 5, "behavioural", 0.25)
    assert [t.actions for t in a] == [t.actions for t in b]
    cfg = dict(DEFAULTS)
    r1 = run_method(spec, "full", a, cfg)
    r2 = run_method(spec, "full", b, cfg)
    for k in ("accuracy", "map_id", "P_intended", "refinement_points"):
        assert r1[k] == r2[k]
    m1 = run_method(spec, "mlci", a[:4], cfg)
    m2 = run_method(spec, "mlci", b[:4], cfg)
    assert m1["constraints"] == m2["constraints"]
