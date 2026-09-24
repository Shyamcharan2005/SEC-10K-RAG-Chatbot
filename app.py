import os
from dotenv import load_dotenv
load_dotenv()


import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types
import chromadb
import time
import pickle


# ---- Everything expensive (loading, chunking, embedding, indexing) runs ONCE and is cached ----
@st.cache_resource
def load_pipeline():
    cache_path = "./chunks_cache.pkl"

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        chunks = cached["chunks"]
        companies = cached["companies"]
        chunk_to_company = cached["chunk_to_company"]
        bm25 = cached["bm25"]
    else:
        loader = DirectoryLoader(
            "./sec_filings_clean",
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs = loader.load()

        for doc in docs:
            filename = os.path.basename(doc.metadata["source"])
            company = filename.split("_")[0]
            doc.metadata["company"] = company

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunk_docs = splitter.split_documents(docs)
        chunks = [doc.page_content for doc in chunk_docs]
        companies = [doc.metadata["company"] for doc in chunk_docs]
        chunk_to_company = dict(zip(chunks, companies))

        tokenized = [chunk.split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized)

        with open(cache_path, "wb") as f:
            pickle.dump({
                "chunks": chunks,
                "companies": companies,
                "chunk_to_company": chunk_to_company,
                "bm25": bm25,
            }, f)

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("sec_filings")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if collection.count() == 0:
        embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            chunk_ids = [f"chunk{j}" for j in range(i, i+len(batch_chunks))]
            collection.add(
                documents=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=[{"company": c} for c in companies[i:i+batch_size]],
                ids=chunk_ids
            )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    genai_client = genai.Client(api_key=os.environ.get("Gemini_api_key"))

    return {
        "chunks": chunks,
        "chunk_to_company": chunk_to_company,
        "collection": collection,
        "embedder": embedder,
        "bm25": bm25,
        "reranker": reranker,
        "genai_client": genai_client,
    }

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    for rank, doc in enumerate(vector_results):
        scores[doc] = scores.get(doc, 0) + 1 / (rank + k)
    for rank, doc in enumerate(bm25_results):
        scores[doc] = scores.get(doc, 0) + 1 / (rank + k)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


# Single retrieval path for every query — hybrid search (vector + BM25) -> RRF fusion -> rerank.
# No company detection/filtering: that was scope creep beyond the original requirements.
def retrieve(query, pipeline, n=5):
    embedder = pipeline["embedder"]
    collection = pipeline["collection"]
    bm25 = pipeline["bm25"]
    chunks = pipeline["chunks"]
    reranker = pipeline["reranker"]

    query_embedding = embedder.encode([query]).tolist()
    vector_results = collection.query(query_embeddings=query_embedding, n_results=10)["documents"][0]

    bm25_scores = bm25.get_scores(query.split())
    top_bm25_indices = bm25_scores.argsort()[-10:][::-1]
    bm25_results = [chunks[i] for i in top_bm25_indices]

    fused = reciprocal_rank_fusion(vector_results, bm25_results)[:20]
    scores = reranker.predict([(query, chunk) for chunk in fused])
    reranked = sorted(zip(scores, fused), reverse=True)
    return [chunk for _, chunk in reranked[:n]]

def generate(query, chunks, pipeline, max_retries=3):
    context = "\n\n".join(
        [f"[{pipeline['chunk_to_company'].get(chunk, 'UNKNOWN')}]: {chunk}" for chunk in chunks]
    )
    for attempt in range(max_retries):
        try:
            response = pipeline["genai_client"].models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=f"Context:\n{context}\n\nQuestion:{query}",
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful AI assistant who is going to be used for SEC filings"
                )
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
             time.sleep(2 ** attempt)
             continue
            else:
                st.error(f"Gemini error: {e}")
                return f"⚠️ The AI service is temporarily unavailable. Please try again in a moment. (Error: {type(e).__name__})"


# ---- Streamlit UI ----
st.set_page_config(page_title="SEC 10-K RAG Chatbot", page_icon="📊")
st.title("📊 SEC 10-K RAG Chatbot")
st.caption("Ask questions about AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA 10-K filings")

with st.spinner("Loading pipeline (first run may take a few minutes)..."):
    pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []


def escape_dollar_signs(text):
    return text.replace("$", "\\$")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(escape_dollar_signs(msg["content"]))
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("View retrieved sources"):
                for chunk in msg["sources"]:
                    company = pipeline["chunk_to_company"].get(chunk, "UNKNOWN")
                    st.markdown(f"**[{company}]** {escape_dollar_signs(chunk)}")

query = st.chat_input("Ask a question about these companies' 10-K filings...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):

            # Measure retrieval
            retrieval_start = time.time()
            retrieved = retrieve(query, pipeline)
            retrieval_time = time.time() - retrieval_start

            # Measure Gemini generation
            generation_start = time.time()
            answer = generate(query, retrieved, pipeline)
            generation_time = time.time() - generation_start

        st.caption(
            f"Retrieval: {retrieval_time:.2f}s | "
            f"Generation: {generation_time:.2f}s"
        )
            
            
        st.markdown(escape_dollar_signs(answer))
        with st.expander("View retrieved sources"):
            for chunk in retrieved:
                company = pipeline["chunk_to_company"].get(chunk, "UNKNOWN")
                st.markdown(f"**[{company}]** {escape_dollar_signs(chunk)}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved})