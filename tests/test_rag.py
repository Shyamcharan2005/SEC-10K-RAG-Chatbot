import numpy as np

import rag


class FakeEmbedder:
    def encode(self, texts):
        return np.zeros((len(texts), 3))


class FakeCollection:
    def __init__(self, ids):
        self.ids = ids

    def query(self, query_embeddings, n_results):
        return {"ids": [self.ids[:n_results]]}


class FakeBM25:
    def __init__(self, scores):
        self.scores = np.array(scores, dtype=float)

    def get_scores(self, tokens):
        return self.scores


class FakeReranker:
    def predict(self, pairs):
        return [float(len(text)) for _, text in pairs]


def make_pipeline(chunks, companies, vector_ids, bm25_scores):
    return rag.RAGPipeline(
        chunks, companies, FakeCollection(vector_ids), FakeEmbedder(),
        FakeBM25(bm25_scores), FakeReranker(),
    )


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rag.reciprocal_rank_fusion([1, 2, 3], [3, 1, 4])
    assert fused[0] == 1
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_handles_empty_input():
    assert rag.reciprocal_rank_fusion([], []) == []


def test_retrieve_returns_top_k_reranked_with_company():
    chunks = ["a", "bbb", "cc", "dddd"]
    companies = ["AAPL", "MSFT", "NVDA", "TSLA"]
    p = make_pipeline(chunks, companies, ["chunk0", "chunk1", "chunk2"], [0.1, 0.9, 0.2, 0.8])
    result = p.retrieve("anything", top_k=2)
    assert [r.text for r in result] == ["dddd", "bbb"]
    assert [r.company for r in result] == ["TSLA", "MSFT"]


def test_duplicate_chunk_text_keeps_correct_company():
    chunks = ["same boilerplate text", "same boilerplate text"]
    companies = ["AAPL", "TSLA"]
    p = make_pipeline(chunks, companies, ["chunk0", "chunk1"], [0.5, 0.6])
    result = p.retrieve("q", top_k=2)
    assert sorted(r.company for r in result) == ["AAPL", "TSLA"]


def test_answer_reports_timings_and_passages(monkeypatch):
    p = make_pipeline(["x"], ["AAPL"], ["chunk0"], [1.0])
    monkeypatch.setattr(p, "generate", lambda q, passages: "stub answer")
    result = p.answer("q")
    assert result.answer == "stub answer"
    assert result.passages[0].company == "AAPL"