import os
import sys
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Project root ko Python import path mein add karta hai.
ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Existing Day 1–Day 8 project code reuse ho raha hai.
from day2.chunking import recursive_chunking
from day3.embeddings import embed_text
from day4.generator import (
    GENERATION_MODEL,
    generate_answer,
)
from day5.memory_chatbot import rewrite_query
from day6.hybrid_retriever import HybridRetriever
from day7.reranker import rerank


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="NexusChat RAG API",
    description=(
        "NexusChat backend using hybrid retrieval, "
        "cross-encoder re-ranking, conversation memory, "
        "and OpenRouter answer generation."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# Global application state
# ---------------------------------------------------------

retriever = HybridRetriever()

# Structure:
# {
#     "session-id": [
#         {"question": "...", "answer": "..."}
#     ]
# }
chat_sessions = {}

chunks_indexed = 0


STARTUP_DOCUMENTS = [
    os.path.join(ROOT, "day2", "sample.txt"),
    os.path.join(ROOT, "day2", "sample.pdf"),
    os.path.join(ROOT, "day2", "sample2.txt"),
]


# ---------------------------------------------------------
# Pydantic request and response models
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask from the indexed documents.",
    )

    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing session ID for follow-up questions. "
            "Leave empty to create a new session."
        ),
    )


class SourceInfo(BaseModel):
    source: str
    preview: str
    rerank_score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    session_id: str


class IngestRequest(BaseModel):
    filename: str = Field(
        ...,
        min_length=1,
        description="Name of the document, for example notes.txt.",
    )

    content: str = Field(
        ...,
        min_length=1,
        description="Raw text content to add to the RAG index.",
    )


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    model: str
    sessions_active: int


# ---------------------------------------------------------
# Startup event
# ---------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Server start hone par sample documents ka hybrid index
    sirf ek dafa build karta hai.
    """

    global chunks_indexed

    print("\nBuilding NexusChat RAG index on startup...")

    missing_documents = [
        path
        for path in STARTUP_DOCUMENTS
        if not os.path.exists(path)
    ]

    if missing_documents:
        missing_names = [
            os.path.basename(path)
            for path in missing_documents
        ]

        raise RuntimeError(
            "Startup documents not found: "
            + ", ".join(missing_names)
        )

    retriever.index(
        STARTUP_DOCUMENTS,
        collection_name="nexuschat_day9",
    )

    if retriever.collection is None:
        raise RuntimeError(
            "Hybrid retriever collection was not created."
        )

    chunks_indexed = retriever.collection.count()

    print(
        f"Startup complete. "
        f"{chunks_indexed} chunks indexed."
    )


# ---------------------------------------------------------
# GET /health
# ---------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    API, model, index and active session status return karta hai.
    """

    return HealthResponse(
        status="ok",
        chunks_indexed=chunks_indexed,
        model=GENERATION_MODEL,
        sessions_active=len(chat_sessions),
    )


# ---------------------------------------------------------
# POST /chat
# ---------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """
    User question receive karta hai, hybrid retrieval aur
    re-ranking karta hai, phir OpenRouter se grounded answer
    generate karta hai.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    session_id = (
        request.session_id.strip()
        if request.session_id
        else str(uuid.uuid4())
    )

    if not session_id:
        session_id = str(uuid.uuid4())

    history = chat_sessions.get(session_id, [])

    try:
        # Follow-up words jaise "it" ko previous question ke
        # context ke saath rewrite karta hai.
        standalone_question = rewrite_query(
            question,
            history,
        )

        if standalone_question != question:
            print(
                "Query rewritten for retrieval: "
                f"{standalone_question}"
            )

        # Hybrid retrieval se zyada candidates fetch karte hain.
        candidates = retriever.search(
            standalone_question,
            top_k=10,
            fetch_k=15,
        )

        if not candidates:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No documents were retrieved "
                    "for this question."
                ),
            )

        # Cross-encoder best three chunks select karta hai.
        chunks = rerank(
            question=standalone_question,
            chunks=candidates,
            top_k=3,
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No relevant documents were found "
                    "for this question."
                ),
            )

        # Actual user question answer generation ko diya jata hai,
        # jabke rewritten query retrieval ke liye use hui.
        answer = generate_answer(
            question=question,
            retrieved_chunks=chunks,
        )

        history.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        chat_sessions[session_id] = history

        sources = [
            SourceInfo(
                source=chunk.get(
                    "source",
                    "Unknown source",
                ),
                preview=chunk.get("text", "")[:150],
                rerank_score=chunk.get(
                    "rerank_score"
                ),
            )
            for chunk in chunks
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Unexpected /chat error: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Internal chat error: {error}",
        ) from error


# ---------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------

@app.post(
    "/ingest",
    response_model=IngestResponse,
)
def ingest(request: IngestRequest):
    """
    Raw text document ko running dense aur BM25 indexes
    mein add karta hai. Server restart ki zarurat nahi hoti.
    """

    global chunks_indexed

    filename = os.path.basename(
        request.filename.strip()
    )

    content = request.content.strip()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename cannot be empty.",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Document content cannot be empty.",
        )

    if retriever.collection is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index is not ready.",
        )

    try:
        chunks = recursive_chunking(
            content,
            size=300,
            overlap=40,
        )

        chunks = [
            chunk.strip()
            for chunk in chunks
            if chunk and chunk.strip()
        ]

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Document produced no chunks.",
            )

        new_ids = []
        new_embeddings = []
        new_texts = []
        new_metadatas = []

        # UUID batch prefix duplicate Chroma IDs se bachata hai.
        batch_id = uuid.uuid4().hex

        for index, chunk in enumerate(chunks):
            chunk_id = (
                f"{filename}_{batch_id}_c{index}"
            )

            new_ids.append(chunk_id)
            new_embeddings.append(
                embed_text(chunk)
            )
            new_texts.append(chunk)
            new_metadatas.append(
                {"source": filename}
            )

        # Dense ChromaDB index update.
        retriever.collection.add(
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_texts,
            metadatas=new_metadatas,
        )

        # Existing BM25 data ke saath new chunks combine
        # karke lexical index rebuild hota hai.
        existing_chunks = list(
            retriever.bm25_retriever.chunks
        )

        existing_sources = list(
            retriever.bm25_retriever.sources
        )

        updated_chunks = (
            existing_chunks + new_texts
        )

        updated_sources = (
            existing_sources
            + [filename] * len(new_texts)
        )

        retriever.bm25_retriever.index(
            updated_chunks,
            updated_sources,
        )

        chunks_indexed = (
            retriever.collection.count()
        )

        return IngestResponse(
            filename=filename,
            chunks_added=len(chunks),
            total_chunks=chunks_indexed,
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Unexpected /ingest error: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Ingest error: {error}",
        ) from error


# ---------------------------------------------------------
# Run the API server
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )