import json
import sys
from pathlib import Path

import httpx
import pytest

from rag_agent.llm import ModelUnavailable, OllamaModel

# eval is an independently runnable harness, not part of the application package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from judge import judge_answer  # noqa: E402
from run_eval import retrieval_scores, summarize  # noqa: E402


def test_ollama_sends_native_tool_schema_and_fixed_options(monkeypatch):
    seen = {}

    def post(url, json, timeout):
        seen.update(json)
        return httpx.Response(
            200, json={"message": {"content": "Answer"}}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", post)
    model = OllamaModel("http://localhost:11434", "fixture")
    assert model.chat([], tools=[{"type": "function"}])["content"] == "Answer"
    assert seen["tools"] == [{"type": "function"}]
    assert seen["options"]["temperature"] == 0
    assert not seen["stream"]


def test_network_failure_has_sanitized_error(monkeypatch):
    def post(*args, **kwargs):
        raise httpx.ConnectError("private endpoint secret")

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ModelUnavailable) as error:
        OllamaModel("http://localhost", "fixture").chat([])
    assert "private endpoint" not in str(error.value)


def test_source_recall_and_mrr_are_not_answer_scores():
    assert retrieval_scores(["wrong", "right", "right"], ["right", "missing"]) == {
        "recall_at_3": 0.5,
        "reciprocal_rank": 0.5,
    }
    assert retrieval_scores(["wrong"], ["right"])["recall_at_3"] == 0
    assert retrieval_scores([], [])["recall_at_3"] is None


def test_unavailable_metrics_remain_null_with_coverage():
    summary = summarize([{"status": "unavailable"}])
    assert summary["completed"] == 0
    assert summary["correctness"] is None
    assert summary["correctness_n"] == 0


def test_judge_validates_structured_scores():
    class Model:
        def chat(self, messages, format_schema):
            assert "correctness" in format_schema["properties"]
            return {
                "content": json.dumps(
                    {
                        "correctness": 0.8,
                        "relevance": 1,
                        "groundedness": 0.9,
                        "notes": "Partial support",
                    }
                )
            }

    case = {"question": "q", "expected": "a", "criteria": ["correctness"]}
    result = {"answer": "a", "sources": []}
    assert judge_answer(Model(), case, result)["correctness"] == 0.8

    class Invalid:
        def chat(self, messages, format_schema):
            return {"content": '{"correctness": 9}'}

    with pytest.raises(ValueError):
        judge_answer(Invalid(), case, result)


def test_dataset_and_experiments_are_consistent():
    cases = json.loads(Path("eval/cases.json").read_text())
    assert 15 <= len(cases) <= 20
    assert len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        assert case["expected"] and case["criteria"]
        for source in case["relevant_sources"]:
            assert (Path("data/sample") / source).is_file()
