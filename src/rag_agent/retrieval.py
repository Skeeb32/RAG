"""Deterministic ingestion, BM25/vector fusion, and replaceable reranking."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    ordinal: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def rank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class SentenceEncoder:
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device="cpu", trust_remote_code=False)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device="cpu", trust_remote_code=False)

    def rank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, hit.chunk.text) for hit in hits])
        ranked = [Hit(hit.chunk, float(score)) for hit, score in zip(hits, scores, strict=True)]
        return sorted(ranked, key=lambda hit: (-hit.score, hit.chunk.id))[:k]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def chunk_documents(root: Path, size: int = 160, overlap: int = 30) -> list[Chunk]:
    if not root.is_dir():
        raise ValueError(f"Document directory does not exist: {root}")
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size")
    chunks = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        words = path.read_text(encoding="utf-8").split()
        source = path.relative_to(root).as_posix()
        for ordinal, start in enumerate(range(0, len(words), size - overlap), 1):
            text = " ".join(words[start:start + size])
            if tokenize(text):
                identity = f"{source}\0{ordinal}\0{text}".encode()
                chunks.append(Chunk(hashlib.sha256(identity).hexdigest()[:20], source, ordinal, text))
            if start + size >= len(words):
                break
    if not chunks:
        raise ValueError("No nonempty UTF-8 Markdown/text documents found")
    return chunks


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (lengths == 0).any():
        raise ValueError("Zero-length embeddings are invalid")
    return vectors / lengths


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, encoder: Encoder) -> None:
        if len(chunks) == 0 or len(chunks) != len(vectors):
            raise ValueError("Chunk and embedding counts must match and be nonzero")
        self.chunks = chunks
        self.vectors = normalize(vectors)
        self.encoder = encoder
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def lexical(self, query: str, k: int = 8) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].id))
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0][:k]

    def semantic(self, query: str, k: int = 8) -> list[Hit]:
        vector = normalize(self.encoder.encode([query]))[0]
        scores = self.vectors @ vector
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].id))
        return [Hit(self.chunks[i], float(scores[i])) for i in order][:k]

    def search(self, query: str, k: int = 8) -> list[Hit]:
        if not query.strip() or k < 1:
            return []
        scores: dict[str, float] = defaultdict(float)
        candidates = {}
        for ranking in (self.lexical(query, k), self.semantic(query, k)):
            for rank, hit in enumerate(ranking, 1):
                scores[hit.chunk.id] += 1.0 / (60 + rank)
                candidates[hit.chunk.id] = hit.chunk
        ordered = sorted(scores, key=lambda key: (-scores[key], key))[:k]
        return [Hit(candidates[key], scores[key]) for key in ordered]

    def save(self, directory: Path, model: str, size: int, overlap: int) -> str:
        payload = json.dumps([asdict(c) for c in self.chunks], sort_keys=True)
        fingerprint = hashlib.sha256(payload.encode() + self.vectors.tobytes()).hexdigest()
        manifest = {"version": 1, "model": model, "size": size, "overlap": overlap,
                    "fingerprint": fingerprint, "chunks": json.loads(payload)}
        directory.mkdir(parents=True, exist_ok=True)
        # One atomically replaced artifact avoids mixed metadata/vector generations.
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            np.savez_compressed(temp_path, vectors=self.vectors, manifest=json.dumps(manifest))
            os.replace(temp_path, directory / "index.npz")
        finally:
            temp_path.unlink(missing_ok=True)
        return fingerprint

    @classmethod
    def load(cls, directory: Path, encoder: Encoder, model: str) -> HybridIndex:
        with np.load(directory / "index.npz", allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest"]))
            vectors = archive["vectors"].copy()
        if manifest["version"] != 1 or manifest["model"] != model:
            raise ValueError("Index version/model mismatch; rebuild with rag-agent ingest")
        payload = json.dumps(manifest["chunks"], sort_keys=True)
        fingerprint = hashlib.sha256(payload.encode() + vectors.tobytes()).hexdigest()
        if fingerprint != manifest["fingerprint"]:
            raise ValueError("Index integrity check failed; rebuild the index")
        return cls([Chunk(**c) for c in manifest["chunks"]], vectors, encoder)
