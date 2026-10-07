"""Shared experiment infrastructure: configuration, run metadata, method runners, I/O, parallelism."""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures", "deterministic_proposals")   # deterministic-proposal ablation figures
ZCACHE = os.path.join(RESULTS, "zcache")
LLM_CACHE = os.path.join(RESULTS, "llm_cache")

DEFAULTS = {
    # paper hyper-parameters (Section V.A)
    "beta": 2.0, "lam": 0.2, "gamma": 0.2, "w_max": 30.0, "w_step": 0.5, "rho": 0.1,
    "prior_mode": "joint", "dl_coding": "nodes",
    # hypothesis proposal: "deterministic" (template ablation) | "openai" (actual LLM through the OpenAI API)
    "provider": "deterministic", "openai_model": "gpt-5.5", "llm_n_hypotheses": 8,
    # the paper used temperature 0.2; the API rejects "temperature" for gpt-5.5 (smoke test), so it is left unset
    "openai_temperature": None, "openai_reasoning_effort": None, "openai_max_output_tokens": None,
    "llm_max_parse_retries": 1, "openai_max_retries": 3, "gpt_runs": 10,
    # MLCI
    "mlci_epsilon": "log_candidates", "mlci_max_constraints": 30, "mlci_top_k": 5, "mlci_geometry": False,
    # Experiment 1 (threshold sensitivity)
    "rho_grid": [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    "threshold_pool_seeds": 20, "threshold_pool_demos": 20,
    # Experiment 2 (noise)
    "noise_rates": [0.0, 0.05, 0.10, 0.20, 0.30], "noise_seeds": 20, "noise_demos": 20,
    "noise_methods": ["full", "fixed", "oracle", "mlci"],
    # Experiment 3 (identifiability)
    "ident_random_seeds": 50, "ident_max_items": 2,
    # misc
    "domains": ["cart", "aisle"], "workers": max(1, (os.cpu_count() or 2) - 2), "bootstrap": 2000,
    "base_seed": 20260601,
}


def parse_args(description: str, extra: Optional[Callable] = None) -> Dict:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", help="JSON file overriding defaults")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override individual settings, VALUE parsed as JSON (e.g. --set rho=0.2 noise_seeds=10)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    ap.add_argument("--out", default=None, help="output directory (default results/<experiment>)")
    if extra:
        extra(ap)
    a = ap.parse_args()
    cfg = copy.deepcopy(DEFAULTS)
    if a.config:
        with open(a.config) as f:
            cfg.update(json.load(f))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    if a.quick:
        cfg.update({"threshold_pool_seeds": 2, "threshold_pool_demos": 8, "noise_seeds": 2, "noise_demos": 10,
                    "noise_rates": [0.0, 0.2], "ident_random_seeds": 4, "rho_grid": [0.05, 0.1, 0.4],
                    "bootstrap": 200})
    cfg["_args"] = {k: v for k, v in vars(a).items()}
    return cfg


def learner_kwargs(cfg: Dict) -> Dict:
    return {k: cfg[k] for k in ("beta", "lam", "gamma", "w_max", "w_step", "rho", "prior_mode", "dl_coding")}


# --------------------------------------------------------------------------------------- metadata
def code_hash() -> str:
    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(ROOT)):
        if any(p in d for p in ("results", "figures", "__pycache__", ".git")):
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                with open(os.path.join(d, f), "rb") as fh:
                    h.update(f.encode() + fh.read())
    return h.hexdigest()[:16]


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def run_metadata(experiment: str, cfg: Dict) -> Dict:
    import matplotlib
    import scipy
    from environment.domains import get_spec
    envs = {}
    for k in cfg.get("domains", ["cart", "aisle"]):
        s = get_spec(k)
        envs[k] = {"layout": s.domain.layout.ascii().split("\n"), "exit_requires_cart": s.domain.exit_requires_cart,
                   "cart_reach": s.domain.cart_reach, "hand_capacity": s.domain.hand_capacity,
                   "items": {n: list(c) for n, c in s.domain.layout.items.items()},
                   "vocab_levels": {str(l): list(v) for l, v in s.domain.vocab_levels.items()},
                   "mlci_families": list(s.domain.mlci_families)}
    return {"experiment": experiment, "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit() or "unavailable (not a git repository)", "code_sha256_16": code_hash(),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "cli": cfg.get("_args"),
            "environments": envs}


def provider_dir(cfg: Dict) -> str:
    """Results of different proposal providers are kept strictly separate."""
    if cfg.get("provider", "deterministic") == "deterministic":
        return "deterministic_proposals"
    if cfg["provider"] == "openai":
        return "gpt" + "".join(ch for ch in cfg["openai_model"].split("gpt", 1)[-1] if ch.isalnum())
    raise ValueError(cfg["provider"])


def fig_dir(cfg: Dict) -> str:
    d = os.path.join(ROOT, "figures", provider_dir(cfg))
    os.makedirs(d, exist_ok=True)
    return d


def out_dir(cfg: Dict, name: str) -> str:
    d = cfg.get("_args", {}).get("out") or os.path.join(RESULTS, provider_dir(cfg), name)
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_jsonable) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_csv(path: str, rows: List[Dict], fields: Optional[List[str]] = None) -> None:
    if not rows:
        open(path, "w").close()
        return
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, default=_jsonable) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


# --------------------------------------------------------------------------------------- methods
def make_zc(spec, cfg: Dict):
    from inference.likelihood import ZComputer
    from inference.refinement import LearnerConfig
    lc = LearnerConfig(**learner_kwargs(cfg))
    return ZComputer(spec.domain, lc.beta, lc.W, cache_dir=ZCACHE)


def make_provider(cfg: Dict, replicate: object = 0, run_info: Optional[Dict] = None):
    """Proposal provider chosen by cfg["provider"].  No fallback between providers."""
    if cfg["provider"] == "deterministic":
        from proposals.deterministic import DeterministicProposalProvider
        return DeterministicProposalProvider()
    if cfg["provider"] == "openai":
        from proposals.llm_stub import LLMProposalProvider
        from proposals.openai_provider import OpenAICompletion
        comp = OpenAICompletion(model=cfg["openai_model"], temperature=cfg["openai_temperature"],
                                reasoning_effort=cfg["openai_reasoning_effort"],
                                max_output_tokens=cfg["openai_max_output_tokens"], cache_dir=LLM_CACHE,
                                replicate=replicate, max_retries=cfg["openai_max_retries"],
                                cache_only=bool(cfg.get("openai_cache_only", False)))
        return LLMProposalProvider(comp, n_hypotheses=cfg["llm_n_hypotheses"],
                                   max_parse_retries=cfg["llm_max_parse_retries"],
                                   run_info={"model": cfg["openai_model"], "replicate": replicate, **(run_info or {})})
    raise ValueError(f"unknown provider {cfg['provider']!r}")


def run_method(spec, method: str, demos, cfg: Dict, rho: Optional[float] = None, baseline: str = "min",
               record_prefix_eval: bool = False, replicate: object = 0, run_info: Optional[Dict] = None,
               record_posteriors: bool = False) -> Dict:
    """Run one method on a demonstration sequence; returns a flat result dict (+ trace, + LLM transcript)."""
    t0 = time.time()
    if method == "mlci":
        from baselines.mlci import MLCI, MLCIConfig
        m = MLCI(spec, MLCIConfig(beta=cfg["beta"], epsilon=cfg["mlci_epsilon"],
                                  max_constraints=cfg["mlci_max_constraints"], top_k=cfg["mlci_top_k"],
                                  include_geometry=cfg["mlci_geometry"])).fit(demos)
        ev = m.evaluate()
        return {"method": "mlci", **{k: v for k, v in ev.items()}, "map_is_intended": None, "P_intended": None,
                "n_refinements": None, "n_hypotheses": ev["n_constraints"], "runtime_s": time.time() - t0,
                "trace": m.history}
    from inference.refinement import NormLearner, method_config
    kw = learner_kwargs(cfg)
    if rho is not None:
        kw["rho"] = rho
    lc = method_config(method, **kw)
    lc.baseline = baseline
    lc.record_posteriors = record_posteriors
    provider = make_provider(cfg, replicate, run_info)
    L = NormLearner(spec, provider, lc, make_zc(spec, cfg))
    out = {"method": method, "provider": cfg["provider"],
           "model": cfg["openai_model"] if cfg["provider"] == "openai" else None}
    try:
        r = L.run(demos, record_prefix_eval=record_prefix_eval)
    except Exception as e:      # recorded, never replaced by another provider
        out.update({"error": f"{type(e).__name__}: {e}", "runtime_s": time.time() - t0,
                    "transcript": getattr(provider, "transcript", None), "llm_calls": _llm_calls(provider)})
        return out
    f = r["final"]
    f["runtime_s"] = time.time() - t0
    out.update({**f, "trace": r["trace"], "transcript": getattr(provider, "transcript", None),
                "llm_calls": _llm_calls(provider), "error": None})
    return out


def _llm_calls(provider):
    comp = getattr(provider, "complete", None)
    return [{k: v for k, v in c.items() if k != "output_text"} for c in getattr(comp, "calls", [])]


SCALAR_KEYS = ["method", "accuracy", "violation_f1", "unseen_accuracy", "precision", "recall", "map_id", "map_is_intended", "map_equivalent_on_heldout",
               "P_intended", "map_posterior", "n_refinements", "refinement_points", "n_hypotheses",
               "n_hypotheses_generated", "posterior_entropy", "runtime_s", "n_constraints", "constraints"]


def flat(result: Dict) -> Dict:
    return {k: result.get(k) for k in SCALAR_KEYS if k in result}


# --------------------------------------------------------------------------------------- parallel
def pmap(fn: Callable, items: Sequence, workers: int) -> List:
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(items))) as pool:
        return pool.map(fn, items, chunksize=1)


def bootstrap_ci(x: Sequence[float], n: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))], float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def seed_for(cfg: Dict, *parts) -> int:
    h = hashlib.sha256(json.dumps([cfg["base_seed"], *parts]).encode()).hexdigest()
    return int(h[:8], 16)


def make_pool_dataset(spec, n: int, seed: int, noise_type: Optional[str] = None, rate: float = 0.0):
    """n demonstrations from the task pool; round(rate*n) of them corrupted (positions chosen at random).

    The clean demonstrations are identical across noise types/rates for the same seed (only the
    corrupted positions change), so noise conditions are paired."""
    rng = np.random.default_rng(seed)
    tasks = [spec.sample_pool_task(rng) for _ in range(n)]
    tie_seeds = rng.integers(0, 2 ** 31, size=n)
    order = rng.permutation(n)
    k = int(round(rate * n)) if noise_type else 0
    corrupt = set(order[:k].tolist())
    nrng = np.random.default_rng(seed + 7919)
    demos = []
    for i in range(n):
        r_i = np.random.default_rng(int(tie_seeds[i]))
        if i in corrupt and noise_type == "behavioural":
            t = spec.behavioural_noise_demo(tasks[i], np.random.default_rng(int(tie_seeds[i]) + 1), name=f"D{i + 1}")
        elif i in corrupt and noise_type == "violation":
            t = spec.violation_demo(np.random.default_rng(int(tie_seeds[i]) + 2), name=f"D{i + 1}")
        else:
            t = spec.clean_demo(tasks[i], r_i, name=f"D{i + 1}")
        t.meta["corrupted"] = i in corrupt
        demos.append(t)
    return demos
