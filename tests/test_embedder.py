"""tests/test_embedder.py — Unit tests for embedder and chunker."""

import numpy as np
import pytest
from src.embeddings.chunker import RecursiveChunker, ChunkStrategy
from src.embeddings.embedder import Embedder


class TestRecursiveChunker:
    def test_basic_split(self):
        chunker = RecursiveChunker(chunk_size=100, overlap=10)
        text = "This is sentence one. " * 20
        chunks = chunker.split(text, source_id="test_001", title="Test Doc")
        assert len(chunks) >= 2
        for c in chunks:
            assert c.source_id == "test_001"
            assert c.strategy == ChunkStrategy.RECURSIVE

    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=500, overlap=50)
        text = "Short text."
        chunks = chunker.split(text, source_id="x", title="X")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_chunk_ids_unique(self):
        chunker = RecursiveChunker(chunk_size=50, overlap=5)
        text = "word " * 200
        chunks = chunker.split(text, source_id="doc1", title="D")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_metadata_passthrough(self):
        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        extra = {"arxiv_id": "2301.00001", "authors": "X; Y"}
        chunks = chunker.split("text " * 50, source_id="src", title="T",
                               extra_metadata=extra)
        for c in chunks:
            assert c.metadata["arxiv_id"] == "2301.00001"

    def test_word_count_property(self):
        chunker = RecursiveChunker(chunk_size=200, overlap=0)
        chunks = chunker.split("hello world foo bar " * 10, source_id="s", title="t")
        for c in chunks:
            assert c.word_count == len(c.text.split())


class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        return Embedder(model_name="all-MiniLM-L6-v2")

    def test_embed_returns_correct_shape(self, embedder):
        texts = ["Hello world", "Scientific paper abstract"]
        embs = embedder.embed(texts)
        assert embs.shape == (2, embedder.dim)
        assert embs.dtype == np.float32

    def test_embed_empty_returns_empty(self, embedder):
        embs = embedder.embed([])
        assert embs.shape == (0, embedder.dim)

    def test_embed_single_shape(self, embedder):
        emb = embedder.embed_single("test sentence")
        assert emb.shape == (embedder.dim,)

    def test_similarity_identical_texts(self, embedder):
        t = "the same sentence"
        e1 = embedder.embed_single(t)
        e2 = embedder.embed_single(t)
        sim = embedder.similarity(e1, e2)
        assert sim > 0.99

    def test_similarity_unrelated_texts(self, embedder):
        e1 = embedder.embed_single("quantum physics equations")
        e2 = embedder.embed_single("chocolate cake recipe")
        sim = embedder.similarity(e1, e2)
        assert sim < 0.8

    def test_batch_similarity_shape(self, embedder):
        query = embedder.embed_single("query text")
        corpus = embedder.embed(["doc one", "doc two", "doc three"])
        sims = embedder.batch_similarity(query, corpus)
        assert sims.shape == (3,)
        assert all(0.0 <= s <= 1.0 for s in sims)
