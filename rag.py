from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FILINGS_DIR = Path(os.environ.get("FILINGS_DIR", "./sec_filings_clean"))
CACHE_PATH = Path(os.environ.get("CHUNKS_CACHE", "./chunks_cache.pkl"))
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CANDIDATES_PER_RETRIEVER = 10
RERANK_POOL = 20

SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about SEC 10-K filings. "
    "Answer using only the provided context. Each passage is labelled with its company "
    "in square brackets; always name the company you are referring to. "
    "If the context does not contain the answer, say so clearly instead of guessing."
)


class GenerationError(RuntimeError):
    """The LLM call failed after all retries."""


@dataclass(frozen=True)
class Passage:
    text: str
    company: str


@dataclass
class Answer:
    answer: str
    passages: list[Passage] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0


def reciprocal_rank_fusion(*rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (rank + k)
    return sorted(scores, key=scores.__getitem__, reverse=True)


class RAGPipeline:
    # Chunks are addressed by integer index, never by text, so duplicate
    # chunk texts across filings can't get the wrong company.
    def __init__(self, chunks, companies, collection, embedder, bm25, reranker, genai_client=None):
        self.chunks = chunks
        self.companies = companies
        self.collection = collection
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker
        self.genai_client = genai_client

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedder.encode([query]).tolist()
        hits = self.collection.query(
            query_embeddings=query_embedding, n_results=CANDIDATES_PER_RETRIEVER
        )
        vector_ids = [int(i.removeprefix("chunk")) for i in hits["ids"][0]]

        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ids = [int(i) for i in bm25_scores.argsort()[-CANDIDATES_PER_RETRIEVER:][::-1]]

        fused = reciprocal_rank_fusion(vector_ids, bm25_ids)[:RERANK_POOL]
        rerank_scores = self.reranker.predict([(query, self.chunks[i]) for i in fused])
        ranked = sorted(zip(rerank_scores, fused), reverse=True)[:top_k]
        return [Passage(self.chunks[i], self.companies[i]) for _, i in ranked]

    def generate(self, query, passages, max_retries=3):
        from google.genai import types

        context = "\n\n".join(f"[{p.company}]: {p.text}" for p in passages)
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=f"Context:\n{context}\n\nQuestion: {query}",
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                )
                if response.text:
                    return response.text
                last_error = RuntimeError("empty response")
            except Exception as exc:
                last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
        raise GenerationError(f"Gemini failed after {max_retries} attempts: {last_error}") from last_error

    def answer(self, query, top_k=5):
        t0 = time.perf_counter()
        passages = self.retrieve(query, top_k)
        t1 = time.perf_counter()
        text = self.generate(query, passages)
        t2 = time.perf_counter()
        return Answer(text, passages, retrieval_seconds=t1 - t0, generation_seconds=t2 - t1)


def _build_chunks():
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = DirectoryLoader(
        str(FILINGS_DIR), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()
    for doc in docs:
        doc.metadata["company"] = os.path.basename(doc.metadata["source"]).split("_")[0]

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunk_docs = splitter.split_documents(docs)
    return [d.page_content for d in chunk_docs], [d.metadata["company"] for d in chunk_docs]


def _load_chunks():
    # Compatible with your existing chunks_cache.pkl
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        return cached["chunks"], cached["companies"]
    chunks, companies = _build_chunks()
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "companies": companies}, f)
    return chunks, companies


def _gemini_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("Gemini_api_key")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (or Gemini_api_key) in the environment or .env file.")
    return key


def load_pipeline():
    import chromadb
    from google import genai
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer

    chunks, companies = _load_chunks()
    bm25 = BM25Okapi([chunk.split() for chunk in chunks])

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() != len(chunks):  # empty or stale index -> rebuild
        if collection.count():
            client.delete_collection(COLLECTION_NAME)
            collection = client.create_collection(COLLECTION_NAME)
        embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()
        batch = 5000
        for start in range(0, len(chunks), batch):
            end = min(start + batch, len(chunks))
            collection.add(
                ids=[f"chunk{j}" for j in range(start, end)],
                documents=chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=[{"company": c} for c in companies[start:end]],
            )

    return RAGPipeline(
        chunks, companies, collection, embedder, bm25,
        CrossEncoder(RERANKER_MODEL), genai.Client(api_key=_gemini_key()),
    )