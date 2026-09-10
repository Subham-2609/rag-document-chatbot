import logging
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="RAG Chatbot API")

vector_store: VectorStore | None = None
chatbot: RAGChatbot | None = None

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def get_vector_store() -> VectorStore:
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store


def get_chatbot() -> RAGChatbot:
    global chatbot
    if chatbot is None:
        chatbot = RAGChatbot(vector_store=get_vector_store())
    return chatbot


class ChatRequest(BaseModel):
    question: str
    k: int = 4


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    dest = UPLOAD_DIR / Path(file.filename).name  # .name strips any path components
    contents = await file.read()
    dest.write_bytes(contents)

    try:
        chunks = process_file(dest)
    except Exception as e:
        logger.exception("Failed to process uploaded file")
        raise HTTPException(500, f"Failed to process file: {e}")

    if not chunks:
        raise HTTPException(422, "No extractable text found in this file.")

    store = get_vector_store()
    store.add_chunks(chunks)  # re-uploads of the same filename now replace old chunks

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_chunks_in_store": store.document_count(),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    store = get_vector_store()
    if store.document_count() == 0:
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
    store = get_vector_store()
    return {"sources": store.list_sources(), "total_chunks": store.document_count()}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    store = get_vector_store()
    deleted = store.delete_by_source(filename)

    if deleted == 0:
        raise HTTPException(404, f"No chunks found for '{filename}'.")

    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()

    return {
        "filename": filename,
        "chunks_deleted": deleted,
        "total_chunks_in_store": store.document_count(),
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))