# Evaluating a RAG system

Evaluate retrieval quality separately from answer quality. Create representative
questions and label the source passages needed to answer each one. Recall@k measures
the fraction of relevant passages present in the top k retrieved results. Track it
while changing chunk sizes, retrieval methods, and ranking settings.

For generated answers, check factual correctness, whether claims are supported by
the retrieved evidence, and whether citations identify the supporting passages.
Include questions that cannot be answered from the documents and check that the
system acknowledges missing information. Also measure latency and cost.
