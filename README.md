# Conversational RAG Chatbot

Chat with your own PDF, DOCX, or TXT documents. A full retrieval-augmented generation
(RAG) pipeline built from scratch:

- **Embeddings**: `sentence-transformers` (local, no external embedding API)
- **Vector store**: ChromaDB
- **LLM**: Groq API (Llama 3.1 8B)
- **Backend**: FastAPI
- **Frontend**: plain HTML/JS (no framework)

## Architecture

```
Upload → chunk (document_processor.py) → embed + store (vector_store.py)
Question → condense w/ chat history → retrieve top-k chunks → generate grounded answer (rag_chain.py)
```

## Features

- Custom document chunking and retrieval pipeline — no LangChain dependency
- Conversational query condensing, so follow-up questions retain context across turns
- Source citations returned alongside every answer
- Simple, responsive chat UI with drag-and-drop style document upload

## Setup

```bash
git clone https://github.com/Subham-2609/rag-document-chatbot.git
cd rag-document-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Groq API key (get one at https://console.groq.com/keys) to `.env`:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

## Run locally

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 — upload a document, then ask questions.

## Project structure

```
rag-document-chatbot/
├── app/
│   ├── main.py               # FastAPI routes
│   ├── document_processor.py # loading + chunking (PDF/DOCX/TXT)
│   ├── vector_store.py       # Chroma + sentence-transformers wrapper
│   └── rag_chain.py          # question condensing + retrieval + generation
├── static/
│   └── index.html            # chat UI
├── data/
│   ├── uploads/               # raw uploaded files
│   └── vectorstore/           # persisted Chroma DB
├── requirements.txt
└── .env.example
```

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload and index a document |
| `POST` | `/api/chat` | Ask a question, get a grounded answer + sources |
| `POST` | `/api/reset` | Clear conversation history |
| `GET` | `/api/documents` | List indexed documents |

## Next steps / ideas to extend it

- Adding streaming responses
- Adding a DELETE endpoint to remove a document
- Swap in reranking for better retrieval quality
- Adding authentication for multi-user or public hosting
- Write a short evaluation set (10-15 Q&A pairs) and report retrieval accuracy
