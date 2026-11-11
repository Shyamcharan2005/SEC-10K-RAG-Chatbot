import streamlit as st

import rag


@st.cache_resource
def get_pipeline():
    return rag.load_pipeline()


def escape_dollar_signs(text):
    return text.replace("$", "\\$")


def show_sources(passages):
    with st.expander("View retrieved sources"):
        for p in passages:
            st.markdown(f"**[{p.company}]** {escape_dollar_signs(p.text)}")


st.set_page_config(page_title="SEC 10-K RAG Chatbot", page_icon="📊")
st.title("📊 SEC 10-K RAG Chatbot")
st.caption("Ask questions about AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA 10-K filings")

with st.spinner("Loading pipeline (first run may take a few minutes)..."):
    pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(escape_dollar_signs(msg["content"]))
        if msg["role"] == "assistant":
            show_sources(msg["sources"])

if question := st.chat_input("Ask a question about these companies' 10-K filings..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and generating answer..."):
                result = pipeline.answer(question)
        except rag.GenerationError:
            st.error("The AI service is temporarily unavailable. Please try again in a moment.")
            st.stop()

        st.caption(f"Retrieval: {result.retrieval_seconds:.2f}s | Generation: {result.generation_seconds:.2f}s")
        st.markdown(escape_dollar_signs(result.answer))
        show_sources(result.passages)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.passages}
    )