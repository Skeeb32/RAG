"""Run paired retrieval or end-to-end experiments and save auditable measurements."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
from judge import judge_answer

from rag_agent.agent import Agent
from rag_agent.cli import load_index
from rag_agent.config import Settings
from rag_agent.llm import ModelUnavailable, OllamaModel
from rag_agent.memory import Memory
from rag_agent.retrieval import CrossEncoderReranker
from rag_agent.tools import ServiceCatalog

ROOT = Path(__file__).resolve().parent


def retrieval_scores(sources: list[str], relevant: list[str]) -> dict:
    if not relevant:
        return {"recall_at_3": None, "reciprocal_rank": None}
    found = set(sources[:3]) & set(relevant)
    rank = next((i for i, source in enumerate(sources[:3], 1) if source in relevant), None)
    return {
        "recall_at_3": len(found) / len(set(relevant)),
        "reciprocal_rank": 1 / rank if rank else 0.0,
    }


def model_digest(settings: Settings, model: str) -> str | None:
    try:
        response = httpx.get(settings.ollama_url.rstrip("/") + "/api/tags", timeout=3)
        response.raise_for_status()
        for item in response.json()["models"]:
            if item["name"] in {model, model + ":latest"}:
                return item["digest"]
    except (httpx.HTTPError, KeyError, ValueError):
        pass
    return None


def summarize(rows: list[dict]) -> dict:
    summary = {"cases": len(rows), "completed": sum(r["status"] == "completed" for r in rows)}
    for key in [
        "recall_at_3",
        "reciprocal_rank",
        "latency_ms",
        "correctness",
        "groundedness",
        "relevance",
        "route_correct",
    ]:
        values = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = statistics.mean(values) if values else None
        summary[key + "_n"] = len(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrieval", "agent"], default="retrieval")
    parser.add_argument("--judge", choices=["none", "llm"], default="none")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "retrieval" and args.judge != "none":
        parser.error("An LLM judge requires --mode agent")
    settings = Settings.from_env()
    cases_bytes = (ROOT / "cases.json").read_bytes()
    cases = json.loads(cases_bytes)
    configs = [
        json.loads((ROOT / "experiments" / f"{name}.json").read_text())
        for name in ("baseline", "improved")
    ]
    index = load_index(settings)
    reranker = CrossEncoderReranker(settings.reranker_model)
    model = OllamaModel(settings.ollama_url, settings.model)
    judge = OllamaModel(settings.ollama_url, settings.judge_model)
    digest = model_digest(settings, settings.model) if args.mode == "agent" else None
    judge_digest = model_digest(settings, settings.judge_model) if args.judge == "llm" else None
    results = []
    for config in configs:
        if config["candidate_k"] != 8 or config["top_k"] != 3:
            raise ValueError("This application uses candidate_k=8 and top_k=3")
        for case in cases:
            row = {
                "configuration": config["name"],
                "case_id": case["id"],
                "status": "completed",
                "judgment_status": "not_requested",
            }
            if args.mode == "retrieval" and (
                not case["relevant_sources"] or case.get("prior_questions")
            ):
                row.update(status="not_applicable", reason="Requires agent routing or history")
                results.append(row)
                continue
            if args.mode == "agent" and digest is None:
                row.update(
                    status="unavailable",
                    reason="Configured Ollama model is unavailable",
                    judgment_status="unavailable" if args.judge == "llm" else "not_requested",
                )
                results.append(row)
                continue
            start = time.perf_counter()
            if args.mode == "retrieval":
                hits = index.search(case["question"], config["candidate_k"])
                if config["rerank"]:
                    hits = reranker.rank(case["question"], hits, config["top_k"])
                sources = [hit.chunk.source for hit in hits[: config["top_k"]]]
                row["sources"] = sources
            else:
                # Both configurations get clean and identical memory for each case.
                with tempfile.TemporaryDirectory() as directory:
                    memory = Memory(Path(directory) / "preferences.sqlite3")
                    for key, value in case.get("preferences", {}).items():
                        memory.set_preference("evaluation", key, value)
                    agent = Agent(
                        index,
                        reranker if config["rerank"] else None,
                        model,
                        ServiceCatalog(settings.index / "catalog.sqlite3"),
                        memory,
                        args.out.parent / (args.out.stem + "_traces"),
                    )
                    prior_results = [
                        agent.ask(q, "evaluation", case["id"])
                        for q in case.get("prior_questions", [])
                    ]
                    result = agent.ask(case["question"], "evaluation", case["id"])
                sources = [item["source"] for item in result["sources"]]
                row.update(
                    answer=result["answer"],
                    sources=sources,
                    agent_status=result["status"],
                    route_correct=float(result["route"] == case["expected_route"]),
                    trace_id=result["run_id"],
                )
                if result["status"] in {"error", "tool_error"} or any(
                    prior["status"] == "error" for prior in prior_results
                ):
                    row.update(status="failed", judgment_status="not_scored")
                elif args.judge == "llm":
                    try:
                        if judge_digest is None:
                            raise ModelUnavailable("Judge model unavailable")
                        row.update(judge_answer(judge, case, result), judgment_status="completed")
                    except (ModelUnavailable, ValueError, KeyError):
                        row["judgment_status"] = "unavailable_or_invalid"
            row["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            row.update(retrieval_scores(sources, case["relevant_sources"]))
            results.append(row)
    summaries = {
        config["name"]: summarize([r for r in results if r["configuration"] == config["name"]])
        for config in configs
    }
    deltas = {
        key: summaries["improved"][key] - summaries["baseline"][key]
        for key in (
            "recall_at_3",
            "reciprocal_rank",
            "correctness",
            "groundedness",
            "relevance",
            "latency_ms",
        )
        if summaries["improved"].get(key) is not None and summaries["baseline"].get(key) is not None
    }
    metadata = {
        "utc": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "judge": args.judge,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
        "dataset_sha256": hashlib.sha256(cases_bytes).hexdigest(),
        "dataset_cases": len(cases),
        "index_sha256": hashlib.sha256((settings.index / "index.npz").read_bytes()).hexdigest(),
        "configuration": {
            k: str(v) for k, v in asdict(replace(settings, ollama_url="redacted")).items()
        },
        "generator_digest": digest,
        "judge_digest": judge_digest,
        "embedding_revision": getattr(
            index.encoder.model[0].auto_model.config, "_commit_hash", None
        ),
        "reranker_revision": getattr(reranker.model.model.config, "_commit_hash", None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["langgraph", "rank-bm25", "sentence-transformers", "numpy", "torch"]
        },
        "experiments": configs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary": summaries,
                "improved_minus_baseline": deltas,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"summary": summaries, "improved_minus_baseline": deltas}, indent=2))


if __name__ == "__main__":
    main()
