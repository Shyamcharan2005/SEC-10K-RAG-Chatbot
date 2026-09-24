# SEC 10-K RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for querying SEC 10-K filings from major technology companies. The system combines dense vector retrieval, BM25 lexical search, reciprocal rank fusion, and cross-encoder reranking before generating answers with Google's Gemini API.

## Overview

This project explores the application of RAG to financial filings, where relevant information can appear across narrative text, financial statements, tables, and company-specific terminology.

The chatbot currently works with SEC 10-K filings from:

* Apple (AAPL)
* Microsoft (MSFT)
* Alphabet (GOOGL)
* Amazon (AMZN)
* Meta (META)
* NVIDIA (NVDA)
* Tesla (TSLA)

Users can ask questions about individual companies and receive answers grounded in retrieved filing content. The application also allows users to inspect the retrieved source chunks used to generate each answer.

## Architecture

```text
User Query
    │
    ▼
Sentence Transformer Embedding
    │
    ├──────────────► ChromaDB Vector Search
    │
    └──────────────► BM25 Keyword Search
                         │
                         ▼
              Reciprocal Rank Fusion
                         │
                         ▼
              Cross-Encoder Reranking
                         │
                         ▼
                Top Retrieved Chunks
                         │
                         ▼
                  Gemini API
                         │
                         ▼
                  Generated Answer
```

## Retrieval Pipeline

### 1. Document Processing

SEC 10-K filings are stored as cleaned text files and loaded using LangChain document loaders.

Documents are split using `RecursiveCharacterTextSplitter` with:

* Chunk size: 500 characters
* Chunk overlap: 50 characters

Company metadata is extracted from source filenames and associated with the resulting chunks.

### 2. Dense Retrieval

The system uses:

`sentence-transformers/all-MiniLM-L6-v2`

to generate embeddings for document chunks and queries.

Embeddings are stored persistently using ChromaDB.

### 3. BM25 Retrieval

BM25 provides lexical retrieval based on token overlap. This complements dense retrieval by allowing exact financial terminology and keywords to influence retrieval.

### 4. Hybrid Retrieval

The top results from vector search and BM25 are combined using **Reciprocal Rank Fusion (RRF)**.

This produces a unified candidate set from both retrieval approaches.

### 5. Cross-Encoder Reranking

The fused candidates are reranked using:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The highest-ranked chunks are passed to the generation model as context.

## Generation

The retrieved context is passed to Google's Gemini API.

Current generation model:

`gemini-3.5-flash-lite`

The model is instructed to answer questions using the provided SEC filing context.

The application sends the current question and its retrieved context to the generation model rather than the entire chat history.

## Evaluation

The RAG pipeline was evaluated using **RAGAS** across 10 questions covering multiple companies and SEC 10-K filings.

### Overall Results

| Metric            |      Score |
| ----------------- | ---------: |
| Faithfulness      | **1.0000** |
| Answer Relevancy  | **0.9609** |
| Context Precision | **0.7315** |
| Context Recall    | **0.7778** |

The results indicate strong grounding and answer relevance, while retrieval precision and recall remain areas for improvement.

Some individual metric evaluations failed because of external LLM API limitations, including rate limits, timeouts, and incomplete generations. Therefore, the reported averages represent the successfully completed evaluations rather than a fully completed evaluation run for every metric and question.

Detailed evaluation results are available in:

`ragas_results.csv`

The evaluation script is available in:

`RAGAS_EVAL.py`

## Features

* Hybrid dense + lexical retrieval
* Reciprocal Rank Fusion
* Cross-encoder reranking
* Persistent ChromaDB vector store
* Cached document processing and retrieval resources
* Gemini-powered answer generation
* Retrieved-source inspection through the Streamlit interface
* RAGAS-based evaluation
* Streamlit deployment

## Known Limitations

### 1. Retrieval sensitivity to query phrasing

Hybrid retrieval can produce different results for semantically equivalent queries with minor differences in wording. BM25 is sensitive to lexical matches while dense retrieval depends on embedding similarity.

Potential improvements include query normalization, canonical query caching, and deterministic query routing.

### 2. Terminology differences across filings

Different companies may use different terminology for similar financial concepts. For example, Amazon uses **"Net Sales"** where other companies may use **"Revenue."**

This can affect lexical retrieval when a query uses terminology that does not appear literally in a particular filing.

A future improvement would be a domain-specific synonym or alias layer.

### 3. Structured and tabular financial data

Some headline financial figures appear inside dense financial tables. Converting these structures into plain text and splitting them into fixed-size chunks can reduce the structural context available to retrieval and generation.

A future implementation could use table-aware document parsing and structured representations.

### 4. Free-tier API constraints

RAGAS evaluation was performed using free-tier LLM APIs. Rate limits and token limits caused some individual evaluation calls to fail during development.

The detailed results therefore contain some missing metric values.

### 5. Cross-company comparisons

The current system is primarily designed for questions concerning a single company.

Because retrieval searches the full corpus without explicitly enforcing balanced representation across companies, comparative questions involving multiple companies are not guaranteed to retrieve sufficient evidence for every company.

The system prioritizes answering from available evidence rather than generating unsupported comparisons.

## Tech Stack

| Component         | Technology                |
| ----------------- | ------------------------- |
| Language          | Python                    |
| UI                | Streamlit                 |
| LLM               | Google Gemini API         |
| Embeddings        | Sentence Transformers     |
| Vector Database   | ChromaDB                  |
| Lexical Retrieval | BM25                      |
| Reranking         | Cross-Encoder             |
| RAG Framework     | LangChain                 |
| Evaluation        | RAGAS                     |
| Version Control   | Git / GitHub              |
| Deployment        | Streamlit Community Cloud |

## Project Structure

```text
SEC-10K-RAG-Chatbot/
│
├── app.py                  # Streamlit application and RAG pipeline
├── clean.py                # SEC filing cleaning/preprocessing
├── docs.py                 # SEC filing/document handling
├── RAGAS_EVAL.py           # RAGAS evaluation script
├── requirements.txt        # Python dependencies
│
├── sec_filings/            # Original SEC filing text
├── sec_filings_clean/      # Cleaned filing text
├── chroma_db/              # Persistent ChromaDB data
├── chunks_cache.pkl        # Cached chunks and BM25 index
└── ragas_results.csv       # Evaluation results
```

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shyamcharan2005/SEC-10K-RAG-Chatbot.git
cd SEC-10K-RAG-Chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Gemini API key

Create a `.env` file:

```text
Gemini_api_key=YOUR_API_KEY
```

The `.env` file should not be committed to Git.

### 4. Run the application

```bash
streamlit run app.py
```

## Deployment

The application is deployed using **Streamlit Community Cloud**.

The Gemini API key is provided through the deployment platform's secrets configuration rather than being stored in the repository.

## Scope

The primary goal of this project was to build and deploy a practical RAG system for querying financial filings while experimenting with hybrid retrieval and reranking techniques.

The current implementation intentionally focuses on **single-company SEC filing questions** rather than attempting to solve every possible financial research query.

## Future Improvements

* Query normalization and synonym expansion
* Table-aware financial document parsing
* Company-aware retrieval routing
* Improved handling of cross-company questions
* More comprehensive evaluation with higher API limits
* Retrieval and response caching
* Additional financial filings and document types
* Specialized financial-domain reranking
