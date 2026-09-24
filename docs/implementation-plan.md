# Implementation plan

Repository inspected before implementation: one standalone Python retrieval script,
 two sample Markdown files, and a README; no package, dependencies, or tests.
The original starter is retained under examples/.

1. Package configuration and deterministic, versioned document ingestion.
2. Established BM25 + normalized sentence embeddings, reciprocal rank fusion,
   and a replaceable local cross-encoder reranker.
3. Explicit LangGraph routing with Ollama-native tool calls, a read-only SQLite
   service-catalog tool, bounded session history, persistent opt-in preferences,
   input/output guards, and metadata-only traces.
4. Eighteen evaluation cases, paired retrieval experiments and an optional real
   LLM judge; meaningful unit/integration tests and exact setup documentation.

Local models avoid paid API requirements. No synthetic provider is used in the
application; test doubles exist only in tests. If inference is unavailable,
record it as unavailable rather than calling a simulation a model evaluation.
