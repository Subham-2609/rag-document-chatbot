# """
# rag_chain.py
# ------------
# The conversational RAG logic:
# 1. Condense the follow-up question + chat history into a standalone question.
# 2. Retrieve relevant chunks from the vector store.
# 3. Generate an answer grounded only in those chunks.

# Uses Groq's free-tier API (fast Llama models) for both steps.
# """

# import os
# from typing import List, Dict
# from groq import Groq
# from app.vector_store import VectorStore

# GROQ_MODEL = "openai/gpt-oss-20b"


# class RAGChatbot:
#     def __init__(self, vector_store: VectorStore, groq_api_key: str | None = None):
#         self.vector_store = vector_store
#         api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
#         if not api_key:
#             raise RuntimeError(
#                 "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys"
#             )
#         self.client = Groq(api_key=api_key)
#         self.chat_history: List[Dict[str, str]] = []  # [{"question":..., "answer":...}]

#     def _condense_question(self, question: str) -> str:
#         """Rewrite a follow-up question into a standalone one using chat history."""
#         if not self.chat_history:
#             return question

#         history_text = "\n".join(
#             f"User: {h['question']}\nAssistant: {h['answer']}" for h in self.chat_history[-5:]
#         )
#         prompt = f"""Given the conversation history and a follow-up question, rewrite the
# follow-up question as a standalone question that contains all necessary context.
# If the follow-up question is already standalone, return it unchanged.

# Chat History:
# {history_text}

# Follow-up question: {question}

# Standalone question:"""

#         response = self.client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_tokens=200,
#         )
#         return response.choices[0].message.content.strip()

#     def _generate_answer(self, question: str, context_chunks: List[dict]) -> str:
#         context_text = "\n\n---\n\n".join(
#             f"[Source: {c['metadata']['source']}]\n{c['text']}" for c in context_chunks
#         )
#         prompt = f"""Answer the question using ONLY the context below. If the context doesn't
# contain the answer, say you don't have enough information — do not make anything up.

# <context>
# {context_text}
# </context>

# Question: {question}

# Answer:"""

#         response = self.client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             max_tokens=800,
#         )
#         return response.choices[0].message.content.strip()

# # changed
#     # def ask(self, question: str, k: int = 4) -> dict:
#     #     standalone_question = self._condense_question(question)
#     #     retrieved = self.vector_store.similarity_search(standalone_question, k=k)

#     #     if not retrieved:
#     #         answer = "I don't have any documents to search yet — please upload some first."
#     #     else:
#     #         answer = self._generate_answer(standalone_question, retrieved)

#     #     self.chat_history.append({"question": question, "answer": answer})

#     #     return {
#     #         "answer": answer,
#     #         "standalone_question": standalone_question,
#     #         "sources": [
#     #             {"source": c["metadata"]["source"], "page": c["metadata"]["page"], "score": round(c["score"], 3)}
#     #             for c in retrieved
#     #         ],
#     #     }


#         def ask(self, question: str, k: int = 4) -> dict:
#         standalone_question = self._condense_question(question)

#         retrieved = self.vector_store.similarity_search(
#             standalone_question,
#             k=k
#         )

#         print("\n========== RAG DEBUG ==========")
#         print("Original question:", question)
#         print("Standalone question:", standalone_question)
#         print("Retrieved chunks:", len(retrieved))

#         for i, chunk in enumerate(retrieved, 1):
#             print(f"\n--- Chunk {i} ---")
#             print("Source:", chunk["metadata"]["source"])
#             print("Score:", chunk["score"])
#             print("Text:", repr(chunk["text"]))

#         print("========== END DEBUG ==========\n")

#         if not retrieved:
#             answer = "I don't have any documents to search yet — please upload some first."
#         else:
#             answer = self._generate_answer(standalone_question, retrieved)

#         self.chat_history.append({
#             "question": question,
#             "answer": answer
#         })

#         return {
#             "answer": answer,
#             "standalone_question": standalone_question,
#             "sources": [
#                 {
#                     "source": c["metadata"]["source"],
#                     "page": c["metadata"]["page"],
#                     "score": round(c["score"], 3)
#                 }
#                 for c in retrieved
#             ],
#         }

#     def clear_history(self):
#         self.chat_history = []


# ## changed
#     # def clear_history(self):
#     #     self.chat_history = []

"""
rag_chain.py
------------
The conversational RAG logic:
1. Condense the follow-up question + chat history into a standalone question.
2. Retrieve relevant chunks from the vector store.
3. Generate an answer grounded only in those chunks.

Uses Groq's API for both steps.
"""

import os
from typing import List, Dict

from groq import Groq

from app.vector_store import VectorStore


GROQ_MODEL = "openai/gpt-oss-20b"


class RAGChatbot:
    def __init__(
        self,
        vector_store: VectorStore,
        groq_api_key: str | None = None
    ):
        self.vector_store = vector_store

        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set."
            )

        self.client = Groq(api_key=api_key)

        self.chat_history: List[Dict[str, str]] = []

    def _condense_question(self, question: str) -> str:
        """Rewrite a follow-up question into a standalone question."""

        if not self.chat_history:
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

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=200,
        )

        return response.choices[0].message.content.strip()

    def _generate_answer(
        self,
        question: str,
        context_chunks: List[dict]
    ) -> str:

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
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=800,
        )

        return response.choices[0].message.content.strip()

    def ask(self, question: str, k: int = 4) -> dict:
        standalone_question = self._condense_question(question)

        retrieved = self.vector_store.similarity_search(
            standalone_question,
            k=k
        )

        if not retrieved:
            answer = (
                "I don't have any documents to search yet — "
                "please upload some first."
            )
        else:
            answer = self._generate_answer(
                standalone_question,
                retrieved
            )

        self.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        return {
            "answer": answer,
            "standalone_question": standalone_question,
            "sources": [
                {
                    "source": c["metadata"]["source"],
                    "page": c["metadata"]["page"],
                    "score": round(c["score"], 3)
                }
                for c in retrieved
            ],
        }

    def clear_history(self):
        self.chat_history = []
