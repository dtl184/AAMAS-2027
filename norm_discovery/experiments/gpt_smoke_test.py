"""One real OpenAI call to verify the GPT-5.5 proposal pipeline before any sweep.

bash -lc 'source ~/.bashrc && cd /home/train/norm_discovery && /home/train/anaconda3/bin/python -m experiments.gpt_smoke_test'

Makes exactly one initial-proposal call (cart domain, demonstration D1), then saves and prints: requested model,
model reported by the API, request id, token usage, prompt, raw response, parsed hypotheses, accepted / rejected
hypotheses and validation errors.  Exit code 0 only if the API reported the requested model and the response
parsed; otherwise the problem is reported and the script exits with code 2 (no fallback).
"""
from __future__ import annotations

import os
import sys

from environment.domains import get_spec
from experiments.common import LLM_CACHE, make_zc, out_dir, parse_args, run_metadata, write_json
from inference.refinement import LearnerConfig, NormLearner
from proposals.llm_stub import LLMProposalProvider
from proposals.openai_provider import OpenAICompletion, model_matches


def main():
    cfg = parse_args("GPT smoke test")
    cfg["provider"] = "openai"
    od = out_dir(cfg, "smoke_test")
    rec = {"metadata": run_metadata("gpt_smoke_test", cfg), "requested_model": cfg["openai_model"]}
    ok = False
    try:
        comp = OpenAICompletion(model=cfg["openai_model"], temperature=cfg["openai_temperature"],
                                reasoning_effort=cfg["openai_reasoning_effort"],
                                max_output_tokens=cfg["openai_max_output_tokens"], cache_dir=None,
                                replicate="smoke", max_retries=cfg["openai_max_retries"])
        spec = get_spec("cart")
        prov = LLMProposalProvider(comp, n_hypotheses=cfg["llm_n_hypotheses"],
                                   max_parse_retries=cfg["llm_max_parse_retries"],
                                   run_info={"experiment": "smoke_test", "domain": "cart", "replicate": "smoke"})
        learner = NormLearner(spec, prov, LearnerConfig(), make_zc(spec, cfg))
        learner.post.add_demo(spec.training[0])
        try:
            dicts = prov.propose_initial(spec, spec.training[:1])
            learner._accept(dicts, 0)
        finally:
            rec["transcript"] = prov.transcript
            rec["calls"] = [{k: v for k, v in c.items() if k != "output_text"} for c in comp.calls]
        call = comp.calls[-1]
        rec["returned_model"] = call.get("returned_model")
        rec["model_confirmed"] = model_matches(cfg["openai_model"], call.get("returned_model"))
        ok = rec["model_confirmed"]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    write_json(os.path.join(od, "smoke_test.json"), rec)
    t = (rec.get("transcript") or [{}])[-1]
    print("requested model :", rec["requested_model"])
    print("returned model  :", rec.get("returned_model"))
    for c in rec.get("calls", []):
        print("request id      :", c.get("response_id"), "| usage:", c.get("usage"), "| retries:", c.get("retries"))
    print("error           :", rec.get("error"))
    print("prompt chars    :", len(t.get("prompt", "")))
    print("parsed          :", None if t.get("parsed") is None else len(t["parsed"]), "hypotheses")
    for a in t.get("accepted") or []:
        print("  ACCEPTED", a["id"], "|", a["description"][:140])
    for r in t.get("rejected") or []:
        print("  REJECTED", r["id"], "|", r["error"][:160])
    print("saved to", os.path.join(od, "smoke_test.json"))
    print("MODEL CONFIRMED" if ok else "SMOKE TEST FAILED -- do not run sweeps")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
