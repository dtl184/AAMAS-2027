"""Algorithm 1: LLM-guided (here: provider-guided) normative abstraction discovery.

  H <- ProposeNorms(alpha_0, tau_1);  posterior with tau_1;  b <- None
  for t = 2..T:
      p_t <- P(tau_t | D_{1:t-1})                                   (Eq. 29)
      if b is not None and p_t < rho * b:                            (Eq. 31)
          H <- H ∪ RefineAbstraction(D_{1:t}, H);  recompute posterior on D_{1:t}
      else:
          update posterior with tau_t;  b <- baseline(non-refining p's)

baseline="min"    : running minimum of the predictive probabilities of previous non-refining
                    demonstrations (the paper's rule; primary).
baseline="median" : median of the same set (secondary diagnostic only).
All probabilities are handled in log space; the trigger is  log p_t < log rho + log b.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from environment.trajectories import Trajectory
from inference.likelihood import ZComputer
from inference.posterior import Posterior
from inference.predictive import classify, evaluate_predictions
from norms.hypotheses import Hypothesis


@dataclass
class LearnerConfig:
    beta: float = 2.0
    lam: float = 0.2
    gamma: float = 0.2
    w_max: float = 30.0
    w_step: float = 0.5
    rho: float = 0.1
    baseline: str = "min"            # "min" (paper) | "median" (diagnostic)
    refinement: bool = True          # False -> fixed abstraction
    oracle: bool = False             # True  -> refined vocabulary + its hypotheses available from t = 1
    prior_mode: str = "joint"
    dl_coding: str = "nodes"         # "nodes" (primary) | "tokens" (sensitivity analysis)
    record_posteriors: bool = False  # store P(h | D_1:t) for every hypothesis at every step

    @property
    def W(self) -> np.ndarray:
        return np.round(np.arange(0.0, self.w_max + 1e-9, self.w_step), 6)


def method_config(method: str, **kw) -> LearnerConfig:
    if method == "full":
        return LearnerConfig(**kw)
    if method == "fixed":
        return LearnerConfig(refinement=False, **kw)
    if method == "oracle":
        return LearnerConfig(refinement=False, oracle=True, **kw)
    raise ValueError(method)


class NormLearner:
    def __init__(self, spec, provider, cfg: LearnerConfig, zc: ZComputer):
        self.spec, self.provider, self.cfg, self.zc = spec, provider, cfg, zc
        self.post = Posterior(zc, cfg.lam, cfg.gamma, cfg.prior_mode, cfg.dl_coding)
        self.level = 0
        self.invalid: List[dict] = []
        self.n_generated = 0
        self.n_rejected = 0

    # ------------------------------------------------------------------ proposals
    def _accept(self, dicts: Sequence[dict], level: int) -> List[Hypothesis]:
        """Validate proposals; record accepted / rejected ones (also into the provider transcript)."""
        fams = self.spec.domain.vocab_levels[level]
        ok, accepted, rejected = [], [], []
        for d in dicts:
            try:
                h = Hypothesis.from_dict(d)
                if level == 0 and (h.abstraction.props or h.abstraction.events):
                    raise ValueError("initial proposals may not define new state properties or events")
                h.validate(fams)
                ok.append(h)
            except Exception as e:  # non-executable proposals are rejected, as in the paper
                rej = {"id": d.get("id"), "error": f"{type(e).__name__}: {e}", "proposal": d}
                self.invalid.append(rej)
                rejected.append(rej)
        self.n_generated += len(ok)
        added = self.post.add_hypotheses(ok)
        added_keys = {h.key for h in added}
        for h in ok:
            accepted.append({"id": h.hid, "description": h.describe(), "L_alpha": h.L_alpha(self.cfg.dl_coding),
                             "L_norm": h.L_norm(self.cfg.dl_coding), "intended_equivalent": self.spec.is_intended(h),
                             "duplicate_of_existing": h.key not in added_keys})
        if hasattr(self.provider, "record_validation"):
            self.provider.record_validation(accepted, rejected)
        self.n_rejected += len(rejected)
        return added

    def _refine(self, demos: Sequence[Trajectory]) -> List[Hypothesis]:
        max_level = max(self.spec.domain.vocab_levels)
        new_level = min(self.level + 1, max_level)
        added = self._accept(self.provider.refine(self.spec, demos, self.post.hyps, new_level), new_level)
        self.level = new_level
        return added

    # ------------------------------------------------------------------ main loop
    def run(self, demos: Sequence[Trajectory], record_prefix_eval: bool = False) -> Dict:
        t0 = time.time()
        cfg, spec = self.cfg, self.spec
        intended_key = spec.true_hypothesis.key
        trace = []
        # t = 1
        self.post.add_demo(demos[0])
        self._accept(self.provider.propose_initial(spec, demos[:1]), 0)
        if cfg.oracle:
            self._refine(demos[:1])
        log_b, nonref = None, []
        trace.append(self._snapshot(1, demos[0], None, None, False, intended_key, record_prefix_eval))
        refinement_points = []
        for t in range(2, len(demos) + 1):
            tau = demos[t - 1]
            log_p = self.post.predictive_logprob(tau)
            log_b_before = log_b
            triggered = (cfg.refinement and log_b is not None and log_p < np.log(cfg.rho) + log_b)
            n_before = len(self.post.hyps)
            self.post.add_demo(tau)
            if triggered:
                self._refine(demos[:t])
                refinement_points.append(t)
            else:
                nonref.append(log_p)
                if cfg.baseline == "min":
                    log_b = min(nonref)
                elif cfg.baseline == "median":
                    log_b = float(np.median(nonref))
                else:
                    raise ValueError(cfg.baseline)
            snap = self._snapshot(t, tau, log_p, log_b_before, triggered, intended_key, record_prefix_eval)
            snap["n_new_hypotheses"] = len(self.post.hyps) - n_before
            trace.append(snap)
        final = self.final_summary(intended_key)
        final.update({"refinement_points": refinement_points, "n_refinements": len(refinement_points),
                      "n_refinement_triggers": len(refinement_points), "runtime_s": time.time() - t0,
                      "n_hypotheses_generated": self.n_generated, "n_invalid_proposals": len(self.invalid),
                      "n_rejected_proposals": self.n_rejected,
                      "final_level": self.level})
        return {"trace": trace, "final": final}

    def _intended_mass(self, m) -> tuple:
        eq = [i for i, h in enumerate(self.post.hyps) if self.spec.is_intended(h)]
        return (float(sum(m[i] for i in eq)) if eq else None), eq

    def _snapshot(self, t, tau, log_p, log_b, triggered, intended_key, with_eval) -> Dict:
        m = self.post.marginal()
        i_int = self.post.index_of_key(intended_key)
        p_eq, eq = self._intended_mass(m)
        imap = int(np.argmax(m))
        snap = {"t": t, "demo": tau.name, "demo_len": len(tau.actions), "log_p": log_p, "log_b": log_b,
                "log_rho_b": (None if log_b is None else float(np.log(self.cfg.rho) + log_b)),
                "log_ratio_p_over_b": (None if log_p is None or log_b is None else log_p - log_b),
                "triggered": bool(triggered), "n_hypotheses": len(self.post.hyps), "level": self.level,
                "P_intended": p_eq, "P_intended_exact": (float(m[i_int]) if i_int is not None else None),
                "intended_proposed": bool(eq),
                "map_id": self.post.hyps[imap].hid, "map_posterior": float(m[imap]),
                "entropy": self.post.entropy()}
        if self.cfg.record_posteriors:
            snap["posterior"] = [{"id": h.hid, "level": h.level, "P": float(m[i]),
                                  "intended_equivalent": i in eq} for i, h in enumerate(self.post.hyps)]
        if with_eval:
            ev = evaluate_predictions(self.spec, classify(self.spec.domain, self.post.hyps[imap], self.spec.heldout))
            snap.update({"accuracy": ev["accuracy"], "violation_f1": ev["violation_f1"],
                         "unseen_accuracy": ev["unseen_accuracy"]})
        return snap

    def final_summary(self, intended_key: str) -> Dict:
        m = self.post.marginal()
        imap = int(np.argmax(m))
        hmap = self.post.hyps[imap]
        ev = evaluate_predictions(self.spec, classify(self.spec.domain, hmap, self.spec.heldout))
        i_int = self.post.index_of_key(intended_key)
        p_eq, eq = self._intended_mass(m)
        true_pred = classify(self.spec.domain, self.spec.true_hypothesis, self.spec.heldout)
        map_pred = classify(self.spec.domain, hmap, self.spec.heldout)
        return {"map_id": hmap.hid, "map_equivalent_on_heldout": map_pred == true_pred, "map_description": hmap.describe(), "map_level": hmap.level,
                "map_posterior": float(m[imap]), "map_is_intended": imap in eq,
                "map_is_intended_exact": hmap.key == intended_key,
                "intended_present": bool(eq), "intended_present_exact": i_int is not None,
                "P_intended": p_eq, "P_intended_exact": float(m[i_int]) if i_int is not None else None,
                "intended_proposed_at_level": sorted({self.post.hyps[i].level for i in eq}),
                "n_hypotheses": len(self.post.hyps), "posterior_entropy": self.post.entropy(),
                "posterior_table": self.post.table(top=8),
                "all_hypotheses": [{"id": h.hid, "level": h.level, "P": float(m[i]), "intended_equivalent": i in eq,
                                    "description": h.describe(), "json": h.to_dict()}
                                   for i, h in enumerate(self.post.hyps)],
                **{k: v for k, v in ev.items()}}
