# RAG

A small, dependency-free retrieval-augmented generation (RAG) starter for Python 3.9+.
It searches local documents and builds a source-cited prompt you can give to a
language model. It does not call a model or generate an answer itself.

## Try it

From the repository directory:

```sh
python3 rag.py "How can I evaluate retrieval quality?"
python3 rag.py "What is chunk overlap?" --top-k 2
```

Add UTF-8 `.txt` or `.md` files to `documents/`, or choose another directory:

```sh
python3 rag.py "Your question" --docs /path/to/documents
```

## How it works

1. Split documents into overlapping word windows (180 words, 30-word overlap).
2. Rank passages using BM25 lexical search.
3. Include the best matching passages with file and chunk citations in a prompt.
4. Copy the printed prompt into your preferred language model to generate an answer.

Only passages with positive scores are included. When nothing matches, the tool
reports that instead of constructing a prompt without evidence. Files stay local
until you choose to send the prompt to a model.

This is an inspectable baseline: it has no embeddings, vector database, API keys,
or external packages. Matching is based on words, so synonyms and paraphrases can
be missed. Chunk numbers identify passages within the current version of a file;
they can change when a document changes. Retrieved text is untrusted input, and
prompt instructions alone do not guarantee protection against prompt injection.

## Next steps

- Create a set of questions with expected source passages and measure recall@k.
- Compare lexical retrieval with embeddings or a hybrid approach.
- Add a model call and verify that each answer's citations support its claims.
- Tune chunk size and overlap using evaluation results.
