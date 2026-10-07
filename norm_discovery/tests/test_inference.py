import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import logsumexp

from environment.domains import get_spec
from experiments.common import DEFAULTS, make_zc
from inference.likelihood import ProductSystem, ZComputer, sample_maxent
from inference.posterior import Posterior
from inference.refinement import LearnerConfig, NormLearner, method_config
from norms.hypotheses import Hypothesis
from proposals import templates as T
from proposals.base import ProposalProvider
from proposals.deterministic import DeterministicProposalProvider

W = LearnerConfig().W


def test_partition_function_matches_power_series():
    """Z from the LU/Woodbury solver equals the explicit sum over trajectory lengths."""
    spec = get_spec("aisle")
    h = spec.true_hypothesis
    task = spec.domain.make_task(["s1"])
    ps = ProductSystem(spec.domain, task, h, 2.0)
    z = ps.z(np.array([0.0, 3.0]), 2.0)
    for j, w in enumerate((0.0, 3.0)):
        M = (ps.M0 + np.exp(-2 * w) * ps.M1).tocsr()
        b = ps.b0 + np.exp(-2 * w) * ps.b1
        x = np.zeros(ps.n); x[ps.start] = 1.0
        total = 0.0
        for L in range(400):
            total += x @ b
            x = M.T @ x
        assert np.isclose(np.log(total), np.log(z[ps.start, j]), atol=1e-8)


def test_trajectory_probabilities_normalised_by_sampling():
    """Empirical frequency of the most likely trajectory under P(tau|h,w) matches its exact probability."""
    spec = get_spec("aisle")
    d = spec.domain
    task = d.make_task(["s1"])
    zc = ZComputer(d, 2.0, np.array([0.0, 5.0]))
    rng = np.random.default_rng(0)
    samples = sample_maxent(d, task, spec.true_hypothesis, 5.0, 2.0, rng, n=3000)
    counts = {}
    for t in samples:
        counts[tuple(t.actions)] = counts.get(tuple(t.actions), 0) + 1
    top = max(counts, key=counts.get)
    from environment.trajectories import Trajectory
    p = np.exp(zc.loglik(spec.true_hypothesis, Trajectory(task, list(top)))[1])
    freq = counts[top] / len(samples)
    assert abs(freq - p) < 4 * np.sqrt(p * (1 - p) / len(samples)) + 1e-3


def test_posterior_normalisation_and_predictive():
    spec = get_spec("cart")
    zc = make_zc(spec, DEFAULTS)
    post = Posterior(zc, 0.2, 0.2)
    post.add_demo(spec.training[0])
    post.add_hypotheses([Hypothesis.from_dict(x) for x in T.exp1_level0((1, 5), (0, 1)) + T.exp1_level1((0, 1))])
    lj = post.log_joint()
    assert np.isclose(logsumexp(lj), 0.0)
    assert np.isclose(post.marginal().sum(), 1.0)
    # predictive = sum_{h,w} P(tau|h,w) P(h,w|D), computed by hand
    tau = spec.training[1]
    manual = logsumexp([lj[i] + zc.loglik(h, tau) for i, h in enumerate(post.hyps)])
    assert np.isclose(post.predictive_logprob(tau), manual)
    # sequential update == batch recomputation
    post.add_demo(tau)
    post2 = Posterior(zc, 0.2, 0.2)
    for t in spec.training[:2]:
        post2.add_demo(t)
    post2.add_hypotheses(post.hyps)
    assert np.allclose(post.log_joint(), post2.log_joint())


def test_prior_follows_description_length():
    spec = get_spec("cart")
    post = Posterior(make_zc(spec, DEFAULTS), 0.2, 0.2)
    hs = [Hypothesis.from_dict(x) for x in T.exp1_level1((0, 1))]
    post.add_hypotheses(hs)
    lp = post.log_prior_h()
    L = np.array([0.2 * h.L_alpha() + 0.2 * h.L_norm() for h in hs])
    assert np.allclose(lp - lp[0], -(L - L[0]))


class _CountingProvider(ProposalProvider):
    def __init__(self):
        self.inner = DeterministicProposalProvider()
        self.calls = 0

    def propose_initial(self, spec, demos):
        return self.inner.propose_initial(spec, demos)

    def refine(self, spec, demos, current, level):
        self.calls += 1
        return self.inner.refine(spec, demos, current, level)


def _run(rho, baseline="min", demos=None):
    spec = get_spec("cart")
    cfg = LearnerConfig(rho=rho, baseline=baseline)
    prov = _CountingProvider()
    L = NormLearner(spec, prov, cfg, make_zc(spec, DEFAULTS))
    return L.run(demos or spec.training), prov


def test_refinement_trigger_rule():
    res, prov = _run(rho=0.1)
    tr = res["trace"]
    # b is undefined at t=2, so no trigger is possible before t=3 (Algorithm 1)
    assert not tr[1]["triggered"] and tr[1]["log_b"] is None
    ratio = tr[2]["log_p"] - tr[2]["log_b"]
    assert tr[2]["triggered"] == (ratio < np.log(0.1))
    assert prov.calls == int(tr[2]["triggered"])
    # a threshold above the observed ratio must trigger, one below must not
    hi = float(np.exp(ratio)) * 1.01
    lo = float(np.exp(ratio)) * 0.99
    assert _run(rho=hi)[0]["trace"][2]["triggered"]
    assert not _run(rho=lo)[0]["trace"][2]["triggered"]


def test_running_min_and_median_baselines():
    spec = get_spec("cart")
    demos = spec.training + spec.training[:2]
    res_min, _ = _run(rho=1e-9, baseline="min", demos=demos)
    res_med, _ = _run(rho=1e-9, baseline="median", demos=demos)
    lp = [s["log_p"] for s in res_min["trace"][1:]]
    assert np.isclose(res_min["trace"][4]["log_b"], min(lp[:3]))
    assert np.isclose(res_med["trace"][4]["log_b"], np.median(lp[:3]))


def test_fixed_never_refines_and_oracle_has_level1():
    spec = get_spec("cart")
    for m in ("fixed", "oracle"):
        cfg = method_config(m, rho=10.0)
        L = NormLearner(spec, DeterministicProposalProvider(), cfg, make_zc(spec, DEFAULTS))
        r = L.run(spec.training)
        assert r["final"]["n_refinements"] == 0
        assert (r["final"]["final_level"] == 1) == (m == "oracle")
