# 📄 PDF Chatbot — Personal RAG Q&A Tool

A beginner-friendly **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask natural-language questions about its content.

**Stack:** Python · LangChain · FAISS · Streamlit · Gemini API

---

## How It Works

1. **Upload** any PDF via the sidebar
2. LangChain **splits** the document into overlapping text chunks
3. Google Gemini **embeds** the chunks and stores them in a **FAISS** vector index
4. You ask a question → the top-4 most relevant chunks are **retrieved**
5. Gemini **generates** a context-aware answer grounded in your document

---

## Project Structure

```
pdf-chatbot/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
└── README.md
```

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/Hamsika248/PDF-chatbot.git
cd PDF-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key
```bash
cp .env.example .env
# Open .env and replace the placeholder with your actual key
```

Get your free Gemini API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Features

- Upload any PDF — textbooks, research papers, reports, contracts
- Multi-turn conversation with memory (follow-up questions work)
- Fast similarity search powered by FAISS
- Clean, minimal Streamlit UI
- API key secured via `.env` file

---

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| PDF parsing | PyPDF2 |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Google Generative AI (`embedding-001`) |
| Vector store | FAISS |
| LLM | Gemini 1.5 Flash |
| RAG chain | LangChain ConversationalRetrievalChain |
