import json

import numpy as np
import pytest

from rag_agent.retrieval import CrossEncoderReranker, HybridIndex, chunk_documents, normalize


def test_chunk_boundaries_stable_ids_and_ignored_files(tmp_path):
    (tmp_path / "b.txt").write_text("one two three four five six seven")
    (tmp_path / "a.md").write_text("alpha beta")
    (tmp_path / "empty.md").write_text("")
    (tmp_path / "skip.csv").write_text("not indexed")
    chunks = chunk_documents(tmp_path, 4, 1)
    assert [c.source for c in chunks] == ["a.md", "b.txt", "b.txt"]
    assert chunks[1].text == "one two three four"
    assert chunks[2].text == "four five six seven"
    assert chunks == chunk_documents(tmp_path, 4, 1)
    assert len({c.id for c in chunks}) == 3
    old = chunks[0].id
    (tmp_path / "a.md").write_text("changed text")
    assert chunk_documents(tmp_path, 4, 1)[0].id != old


def test_empty_and_invalid_chunking(tmp_path):
    with pytest.raises(ValueError, match="No nonempty"):
        chunk_documents(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        chunk_documents(tmp_path, 3, 3)
    with pytest.raises(ValueError, match="does not exist"):
        chunk_documents(tmp_path / "absent")


def test_ingestion_excludes_symlinks(tmp_path):
    (tmp_path / "document.md").write_text("real document")
    (tmp_path / "link.md").symlink_to(tmp_path / "document.md")
    assert len(chunk_documents(tmp_path)) == 1


def test_bm25_prefers_exact_term(index):
    assert index.lexical("bark")[0].chunk.source == "dogs.md"
    assert index.lexical("xyzunmatched") == []


def test_vector_retrieval_matches_fixture_synonym(index):
    assert index.semantic("feline")[0].chunk.source == "cats.md"


def test_fusion_deduplicates_and_respects_cutoff(index):
    hits = index.search("cats", 2)
    assert hits[0].chunk.source == "cats.md"
    assert len(hits) == len({h.chunk.id for h in hits}) == 2
    assert index.search(" ") == []


def test_persisted_index_integrity_and_model_validation(index, tmp_path):
    index.save(tmp_path, "fixture", 4, 1)
    loaded = HybridIndex.load(tmp_path, index.encoder, "fixture")
    assert loaded.search("cat")[0].chunk.id == "a"
    with pytest.raises(ValueError, match="mismatch"):
        HybridIndex.load(tmp_path, index.encoder, "other-model")
    with np.load(tmp_path / "index.npz", allow_pickle=False) as archive:
        vectors = archive["vectors"].copy()
        manifest = json.loads(str(archive["manifest"]))
    vectors[0, 0] = 99
    np.savez_compressed(tmp_path / "index.npz", vectors=vectors, manifest=json.dumps(manifest))
    with pytest.raises(ValueError, match="integrity"):
        HybridIndex.load(tmp_path, index.encoder, "fixture")


def test_invalid_embeddings_rejected():
    for vectors in [np.array([[0, 0]]), np.array([[np.nan, 1]]), np.array([1, 2])]:
        with pytest.raises(ValueError):
            normalize(vectors)


def test_cross_encoder_adapter_uses_pair_scores_and_top_k(index):
    seen = []

    class Model:
        def predict(self, pairs):
            seen.extend(pairs)
            return [0.1, 0.8, 0.3]

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = Model()
    hits = index.semantic("cat")
    result = reranker.rank("question", hits, 1)
    assert result[0].chunk == hits[1].chunk
    assert seen[0] == ("question", hits[0].chunk.text)
    assert reranker.rank("question", [], 1) == []
