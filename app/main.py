"""
main.py
-------
FastAPI app exposing:
  POST /api/upload   - upload a document, chunk it, embed it, store it
  POST /api/chat      - ask a question, get a grounded answer + sources
  POST /api/reset      - clear chat history
  GET  /api/documents  - list ingested documents
  GET  /                - serves the frontend
"""

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.document_processor import process_file
from app.vector_store import VectorStore
from app.rag_chain import RAGChatbot

load_dotenv()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="RAG Chatbot API")

vector_store = VectorStore()
chatbot: RAGChatbot | None = None


def get_chatbot() -> RAGChatbot:
    global chatbot
    if chatbot is None:
        chatbot = RAGChatbot(vector_store=vector_store)
    return chatbot


class ChatRequest(BaseModel):
    question: str
    k: int = 4


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {allowed}")

    dest = UPLOAD_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)

    try:
        chunks = process_file(dest)
    except Exception as e:
        raise HTTPException(500, f"Failed to process file: {e}")

    vector_store.add_chunks(chunks)

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_chunks_in_store": vector_store.document_count(),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if vector_store.document_count() == 0:
        raise HTTPException(400, "No documents uploaded yet.")
    try:
        result = get_chatbot().ask(req.question, k=req.k)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return result


@app.post("/api/reset")
async def reset_chat():
    get_chatbot().clear_history()
    return {"status": "chat history cleared"}


@app.get("/api/documents")
async def list_documents():
    return {
        "sources": vector_store.list_sources(),
        "total_chunks": vector_store.document_count(),
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
