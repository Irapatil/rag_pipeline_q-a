import os
import streamlit as st

# Make sure PYTHONPATH includes project root or run with `streamlit run streamlit_app.py`
from src.qa_pipeline import answer_question
from src.ingestion import run_ingestion

st.set_page_config(page_title="RAG Q&A", layout="wide")

st.title("📄 Document-Aware Q&A (RAG)")
st.caption("Upload docs → Build index → Ask questions")

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM Provider", ["local", "openai", "vertex"])
    os.environ["LLM_PROVIDER"] = provider

    st.markdown("---")
    st.subheader("🔐 Keys (only if needed)")
    if provider == "openai":
        key = st.text_input("OPENAI_API_KEY", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    if provider == "vertex":
        key = st.text_input("GOOGLE_API_KEY", type="password")
        if key:
            os.environ["GOOGLE_API_KEY"] = key

    st.markdown("---")
    st.subheader("📥 Ingestion")
    uploaded = st.file_uploader(
        "Upload 2–3 docs (.txt)",
        type=["txt"],
        accept_multiple_files=True
    )

    if st.button("Build / Rebuild Index"):
        if not uploaded:
            st.warning("Upload at least one document.")
        else:
            import os, shutil
            os.makedirs("data/docs", exist_ok=True)
            # clear previous docs
            for f in os.listdir("data/docs"):
                os.remove(os.path.join("data/docs", f))

            for f in uploaded:
                with open(os.path.join("data/docs", f.name), "wb") as out:
                    out.write(f.getbuffer())

            run_ingestion()
            st.success("Index built successfully!")

st.markdown("## ❓ Ask a question")
query = st.text_input("Type your question")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            res = answer_question(query)
        st.markdown("### 💡 Answer")
        st.write(res["answer"])

        with st.expander("🔍 Retrieved Context"):
            for i, c in enumerate(res["context"], 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(c)