# Verification record

Verified on 2026-09-24 with Python 3.12.14 on macOS ARM64.

- `pytest -q`: 45 passed. Tests use controlled external-model doubles and real
  LangGraph execution, real BM25, NumPy search, filesystem artifacts and SQLite.
- `ruff check src tests eval`: passed.
- `ruff format --check src tests eval`: 17 files formatted correctly.
- `git diff --check`: passed.
- Built `rag_agent_portfolio-0.1.0-py3-none-any.whl` with pip, without dependencies.
- Installed the package in editable mode and invoked its `rag-agent` entry point.
- Real `rag-agent ingest`: ten chunks; repeated ingestion produced fingerprint
  `1c1a3baa8d01ede18396408be2b509c5c5833e0af2dcd63f7f9fa11fdb94400b`.
- Real `rag-agent search "Why repeat words across passage boundaries?"`: the first
  cross-encoder-ranked source was `chunking.md`.
- Real baseline `ask`: no Ollama server was running; returned status `error`, no
  sources, and an input/decision/output-guard/final trace. This verifies failure
  behavior, not live LLM answer quality.
- Paired retrieval experiments use actual downloaded SentenceTransformer and
  CrossEncoder models, not the test fixtures. Both scored recall@3 1.0 and mean
  reciprocal rank 0.944 on the 12 applicable cases; six agent-dependent cases were
  not applicable to standalone retrieval.
- Agent/judge preflight found no configured Ollama model available. Both sets of
  18 cases are recorded as unavailable, with no fabricated answer or judge scores.

Raw per-case measurements and model/package metadata are in `eval/results/`.
The sample knowledge base, questions and labels are committed with the code.
No hosted deployment, live generation, judge inference, large-corpus performance,
security assurance, or cross-platform runtime guarantee is claimed.
