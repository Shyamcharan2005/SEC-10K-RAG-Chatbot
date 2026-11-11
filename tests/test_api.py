import pytest
from fastapi.testclient import TestClient

import api
import rag


class FakePipeline:
    chunks = ["alpha", "beta"]
    companies = ["AAPL", "NVDA"]

    def __init__(self, fail=False):
        self.fail = fail

    def answer(self, question, top_k=5):
        if self.fail:
            raise rag.GenerationError("boom")
        return rag.Answer(f"echo: {question}", [rag.Passage("alpha", "AAPL")][:top_k], 0.0123456, 0.5)


@pytest.fixture
def client():
    api.app.dependency_overrides[api.get_pipeline] = lambda: FakePipeline()
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "chunks": 2, "companies": ["AAPL", "NVDA"]}


def test_query_success(client):
    r = client.post("/query", json={"question": "  What is Apple's buyback?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo: What is Apple's buyback?"
    assert body["sources"] == [{"company": "AAPL", "text": "alpha"}]
    assert body["retrieval_seconds"] == 0.012


@pytest.mark.parametrize("payload", [
    {},
    {"question": "hi"},
    {"question": "x" * 1001},
    {"question": "valid question", "top_k": 0},
    {"question": "valid question", "top_k": 11},
])
def test_query_validation_errors(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_llm_failure_returns_503():
    api.app.dependency_overrides[api.get_pipeline] = lambda: FakePipeline(fail=True)
    try:
        r = TestClient(api.app).post("/query", json={"question": "valid question"})
    finally:
        api.app.dependency_overrides.clear()
    assert r.status_code == 503


def test_pipeline_not_loaded_returns_503():
    assert TestClient(api.app).get("/health").status_code == 503