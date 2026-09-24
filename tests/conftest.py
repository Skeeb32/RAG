import numpy as np
import pytest

from rag_agent.retrieval import Chunk, HybridIndex


class TinyEncoder:
    """Vector fixture; never used by the app or measured experiments."""

    def encode(self, texts):
        return np.array(
            [
                [1.0, 0.0]
                if any(term in text.lower() for term in ["cat", "feline", "kitten"])
                else [0.0, 1.0]
                for text in texts
            ]
        )


@pytest.fixture
def index():
    chunks = [
        Chunk("a", "cats.md", 1, "cats purr softly"),
        Chunk("b", "dogs.md", 1, "dogs bark loudly"),
        Chunk("c", "birds.md", 1, "birds fly high"),
    ]
    encoder = TinyEncoder()
    return HybridIndex(chunks, encoder.encode([c.text for c in chunks]), encoder)
