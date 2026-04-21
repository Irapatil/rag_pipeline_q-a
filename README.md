Document-Aware Q&A System (RAG)
📌 Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers user queries based on local documents.

The system:

Ingests text documents
Converts them into vector embeddings
Stores them in a FAISS index
Retrieves relevant content for a given query
Uses an LLM to generate context-aware answers
🧠 Architecture
Documents → Chunking → Embeddings → FAISS Index  
User Query → Embedding → Similarity Search → Context → LLM → Answer
⚙️ Tech Stack
Python 3.10+
Sentence Transformers (Embeddings)
FAISS (Vector Store)
HuggingFace Transformers (LLM)
Torch
📁 Project Structure
rag_system/
│
├── data/
│   └── docs/
│       ├── doc1.txt
│       ├── doc2.txt
│
├── src/
│   ├── ingestion.py
│   ├── retriever.py
│   ├── qa_pipeline.py
│   ├── main.py
│
├── vector_store/
├── requirements.txt
└── README.md
🚀 Setup Instructions
🔹 1. Clone the Repository
git clone <your-repo-link>
cd rag_system
🔹 2. Create Virtual Environment
python -m venv .venv

Activate:

Windows

.venv\Scripts\activate

Mac/Linux

source .venv/bin/activate
🔹 3. Install Dependencies
pip install -r requirements.txt
▶️ Run the System
🔹 Step 1: Ingest Documents
python src/ingestion.py

Expected Output:

📥 Loading documents...
✂️ Created X chunks
✅ Vector store created successfully!
🔹 Step 2: Start Q&A Interface
python src/main.py
🧪 Usage

Enter your query in the terminal:

Ask a question: What is machine learning?
🎯 Sample Output
💡 Answer:
Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data.
🧠 Design Decisions
Sentence Transformers used for embeddings to avoid API dependency
FAISS chosen for fast and lightweight vector search
Local LLM (DistilGPT2) used for offline inference
Modular pipeline: ingestion → retrieval → generation
⚖️ Trade-offs
Design Choice	Trade-off
Local LLM	Faster setup but lower accuracy vs GPT
FAISS	No metadata filtering
Simple chunking	No semantic splitting
📌 Assumptions
Documents are small (2–3 files)
Input format is .txt
System runs locally
