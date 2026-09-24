# Chunking documents

Chunking divides a document into smaller passages for retrieval. A passage should
contain enough context to make sense on its own without including too many unrelated
topics. Start with a consistent window size and evaluate alternatives.

Chunk overlap repeats some words from the end of one passage at the beginning of
the next. This can preserve context across boundaries, but also increases index
size and can produce redundant search results. Deduplication or reranking can help.
