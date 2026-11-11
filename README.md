# SEC 10-K RAG Chatbot

A Retrieval-Augmented Generation (RAG) system for questions about SEC 10-K filings. Hybrid retrieval (dense vectors + BM25), Reciprocal Rank Fusion and cross-encoder reranking feed Google's Gemini API. It is served as a **FastAPI service** and a **Streamlit app**, and ships with **Docker**, **pytest tests** and **GitHub Actions CI**.

**Live demo:** [sec-10k-rag-chatbot.streamlit.app](https://sec-10k-rag-chatbot-m75ccdbrprkmpsxu9xayw8.streamlit.app/)

Covered filings: Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN), Meta (META), NVIDIA (NVDA), Tesla (TSLA), split into 7,202 chunks.

## Architecture

```text
User Query
    │
    ├─► Sentence-Transformer embedding ─► ChromaDB vector search ─┐
    │                                                             ├─► Reciprocal Rank Fusion
    └─► BM25 keyword search ──────────────────────────────────────┘          │
                                                                             ▼
                                                              Cross-encoder reranking
                                                                             │
                                                                             ▼
                                                    Top-k chunks ─► Gemini ─► grounded answer

        rag.py (core)  ◄── api.py (FastAPI)   ◄── Docker
                       ◄── app.py (Streamlit)
                       ◄── RAGAS_EVAL.py
```

| Stage | Detail |
| --- | --- |
| Chunking | `RecursiveCharacterTextSplitter`, 500 chars, 50 overlap; company taken from the filename |
| Dense retrieval | `all-MiniLM-L6-v2` embeddings in a persistent ChromaDB collection (top 10) |
| Lexical retrieval | BM25 over whitespace tokens (top 10) |
| Fusion | Reciprocal Rank Fusion (k = 60), best 20 candidates kept |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 5 returned |
| Generation | Gemini (`gemini-3.5-flash-lite` by default, override with `GEMINI_MODEL`); answers only from the context and names the company |

Chunks are addressed by integer index end to end, so identical text appearing in two filings is never attributed to the wrong company.

## API

```bash
uvicorn api:app --port 8000      # interactive docs at http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What risks did Tesla disclose about Full Self-Driving?", "top_k": 5}'
```

| Endpoint | Description |
| --- | --- |
| `POST /query` | Pydantic-validated body: `question` (3-1000 chars), `top_k` (1-10, default 5). Returns `answer`, `sources` (company + text) and retrieval/generation timings. Invalid input returns `422`; an LLM failure returns `503`. |
| `GET /health` | Pipeline status, chunk count and companies loaded. |

## Docker

```bash
docker build -t sec-rag .
docker run -p 8000:8000 --env-file .env -v chroma_data:/app/chroma_db sec-rag
```

API keys are passed at runtime and never baked into the image. The named volume keeps the vector index between runs, so only the first start has to embed the chunks (a few minutes).

## Tests and CI

```bash
pip install -r requirements-dev.txt
pytest -q
```

The 14 tests cover Reciprocal Rank Fusion, retrieval ordering, a duplicate-chunk regression, request validation and error handling. They use small fakes instead of real models, so they run in seconds without API keys. GitHub Actions (`.github/workflows/ci.yml`) runs them on every push and pull request.

## Evaluation (RAGAS)

Evaluated on 10 questions across the 7 companies (`RAGAS_EVAL.py`, results in `ragas_results.csv`). Free-tier judge-model calls failed for some questions, so **each average covers only the questions that scored successfully**:

| Metric | Average | Questions scored |
| --- | ---: | ---: |
| Faithfulness | 1.0000 | 1 / 10 |
| Answer relevancy | 0.9609 | 4 / 10 |
| Context precision | 0.7315 | 9 / 10 |
| Context recall | 0.7778 | 9 / 10 |

The retrieval metrics (precision, recall) are the most reliable. Faithfulness and answer relevancy rest on too few questions to draw conclusions; a complete run needs more API headroom. `RAGAS_EVAL.py` prints how many questions each average is based on.

Revenue-style questions score worst: headline figures sit in dense financial tables that are flattened to text and split mid-table by the chunker.

## Running locally

```bash
pip install -r requirements.txt
```

Create a `.env` file (never commit it):

```text
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key     # only needed for RAGAS_EVAL.py
```

(`Gemini_api_key` and `Groq_api_key` are also accepted.)

```bash
streamlit run app.py          # UI
uvicorn api:app --port 8000   # or the API
```

To rebuild the corpus from scratch (`pip install -r requirements-tools.txt`): put your contact email in the `User-Agent` header in `docs.py` (SEC requires it), then run `python docs.py` to download the filings into `sec_filings/` and `python clean.py` to strip the HTML into `sec_filings_clean/`. Run `python RAGAS_EVAL.py` to evaluate.

## Project structure

```text
├── rag.py                  # Core pipeline: retrieval, fusion, rerank, generation
├── api.py                  # FastAPI service
├── app.py                  # Streamlit UI
├── RAGAS_EVAL.py           # RAGAS evaluation
├── docs.py                 # Download 10-Ks from SEC EDGAR
├── clean.py                # HTML to plain text
├── tests/                  # pytest suite (test_rag.py, test_api.py)
├── Dockerfile
├── .github/workflows/ci.yml
├── requirements.txt        # runtime
├── requirements-dev.txt    # CI and tests
├── requirements-tools.txt  # data prep and evaluation
├── chunks_cache.pkl        # cached chunks and company labels
└── ragas_results.csv
```

## Known limitations

* **Query phrasing sensitivity:** BM25 tokenizes on whitespace without lowercasing or stripping punctuation, so near-identical questions can retrieve different chunks.
* **Terminology differences:** Amazon says "Net Sales" where others say "Revenue"; a synonym layer would help lexical retrieval.
* **Tables:** table-aware parsing would preserve the structure of financial statements.
* **Cross-company questions:** retrieval searches the whole corpus without balancing companies, so comparisons are not guaranteed to find evidence for each one.
* **Evaluation coverage:** see the table above.

## Future improvements

Table-aware parsing, company-aware retrieval routing, query normalization and synonym expansion, retrieval and response caching, and a complete RAGAS run.