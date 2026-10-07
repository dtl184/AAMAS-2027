from proposals.llm_stub import LLMProposalProvider, parse_hypotheses
from proposals.openai_provider import model_matches


def test_model_matching_is_strict():
    assert model_matches("gpt-5.5", "gpt-5.5")
    assert model_matches("gpt-5.5", "gpt-5.5-2026-04-23")
    assert not model_matches("gpt-5.5", "gpt-5.5-mini")
    assert not model_matches("gpt-5.5", "gpt-5")
    assert not model_matches("gpt-5.5", None)


def test_parse_hypotheses_variants():
    assert parse_hypotheses('[{"id": "a", "norm": {}}]') == [{"id": "a", "norm": {}}]
    assert parse_hypotheses('```json\n[{"id": "a"}]\n```') == [{"id": "a"}]
    assert parse_hypotheses('{"hypotheses": [{"id": "a"}]}') == [{"id": "a"}]
    assert parse_hypotheses("no json here") is None


def test_llm_provider_records_transcript_and_retries_parse():
    from environment.domains import get_spec
    answers = iter(["not json", '[{"id": "x", "abstraction": {}, "norm": {"type": "prohibition", "condition": "EXIT"}}]'])
    prov = LLMProposalProvider(lambda p: next(answers), n_hypotheses=2)
    spec = get_spec("cart")
    out = prov.propose_initial(spec, spec.training)
    assert len(out) == 1 and out[0]["id"].startswith("g0_")
    t = prov.transcript[0]
    assert t["kind"] == "initial" and len(t["attempts"]) == 2 and t["parsed"] is not None
    assert "Demonstration 1" in t["prompt"] and "Demonstration 2" not in t["prompt"]
