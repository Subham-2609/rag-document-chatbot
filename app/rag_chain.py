import logging
import os
from typing import List, Dict

from groq import Groq

from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

GROQ_MODEL = "openai/gpt-oss-20b"

# Words that suggest a question depends on prior conversational context.
_FOLLOWUP_MARKERS = (
    "it", "that", "this", "those", "these", "he", "she", "they",
    "the same", "also", "again", "what about", "and the",
)


class RAGChatbot:
    def __init__(self, vector_store: VectorStore, groq_api_key: str | None = None):
        self.vector_store = vector_store

        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set.")

        self.client = Groq(api_key=api_key)
        self.chat_history: List[Dict[str, str]] = []

    def _looks_like_followup(self, question: str) -> bool:
        q = f" {question.lower().strip()} "
        if len(question.split()) <= 4:
            return True
        return any(f" {marker} " in q for marker in _FOLLOWUP_MARKERS)

    def _condense_question(self, question: str) -> str:
        """Rewrite a follow-up question into a standalone question. Only
        called when the question actually looks like it depends on prior
        context — otherwise a fresh, unrelated question gets needlessly
        rewritten and can drift enough to break retrieval."""

        if not self.chat_history or not self._looks_like_followup(question):
            return question

        history_text = "\n".join(
            f"User: {h['question']}\nAssistant: {h['answer']}"
            for h in self.chat_history[-5:]
        )

        prompt = f"""Given the conversation history and a follow-up question,
rewrite the follow-up question as a standalone question that contains all
necessary context.

If the follow-up question is already standalone, return it unchanged.

Chat History:
{history_text}

Follow-up question: {question}

Standalone question:"""

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            logger.exception("Condensation failed, falling back to raw question")
            return question

    def _generate_answer(self, question: str, context_chunks: List[dict]) -> str:
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['metadata']['source']}]\n{c['text']}"
            for c in context_chunks
        )

        prompt = f"""Answer the question using ONLY the context below.

If the context doesn't contain the answer, say you don't have enough
information. Do not make anything up.

<context>
{context_text}
</context>

Question: {question}

Answer:"""

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()

    def ask(self, question: str, k: int = 4) -> dict:
        standalone_question = self._condense_question(question)

        retrieved = self.vector_store.similarity_search(standalone_question, k=k)

        if not retrieved:
            answer = "I don't have any documents to search yet — please upload some first."
        else:
            answer = self._generate_answer(standalone_question, retrieved)

        # Don't let a non-answer pollute future condensation context.
        if "don't have enough information" not in answer.lower():
            self.chat_history.append({"question": question, "answer": answer})

        return {
            "answer": answer,
            "standalone_question": standalone_question,
            "sources": [
                {
                    "source": c["metadata"]["source"],
                    "page": c["metadata"]["page"],
                    "score": round(c["score"], 3),
                }
                for c in retrieved
            ],
        }

    def clear_history(self) -> None:
        self.chat_history = []