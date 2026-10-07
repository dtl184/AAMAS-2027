"""Held-out prediction with the MAP normative hypothesis and evaluation metrics."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from environment.trajectories import Trajectory
from norms.hypotheses import Hypothesis
from norms.violations import count_violations


def classify(domain, hyp: Hypothesis, trajs: Sequence[Trajectory]) -> List[bool]:
    """A trajectory is predicted violating iff the hypothesis assigns it >= 1 violation."""
    return [count_violations(domain, hyp, t) >= 1 for t in trajs]


def metrics(pred: Sequence[bool], truth: Sequence[bool]) -> Dict[str, float]:
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    acc = float((pred == truth).mean()) if len(truth) else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "violation_f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn,
            "tn": int((~pred & ~truth).sum())}


def evaluate_predictions(spec, pred: Sequence[bool]) -> Dict[str, object]:
    truth = [bool(t.label) for t in spec.heldout]
    out = metrics(pred, truth)
    if spec.unseen_names:
        idx = [i for i, t in enumerate(spec.heldout) if t.name in spec.unseen_names]
        out["unseen_accuracy"] = metrics([pred[i] for i in idx], [truth[i] for i in idx])["accuracy"]
    else:
        out["unseen_accuracy"] = None
    out["predictions"] = {t.name: bool(p) for t, p in zip(spec.heldout, pred)}
    return out


def evaluate_hypothesis(spec, hyp: Hypothesis) -> Dict[str, object]:
    return evaluate_predictions(spec, classify(spec.domain, hyp, spec.heldout))
