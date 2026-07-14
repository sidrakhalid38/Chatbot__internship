import json
import os
import sys
import uuid
from typing import List, Optional
import truststore

truststore.inject_into_ssl()
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


CHAT_UI_PATH = os.path.join(
    ROOT,
    "day11",
    "chat.html",
)


# ---------------------------------------------------------
# Existing project imports
# ---------------------------------------------------------

from day2.ingestion import load_document
from day2.chunking import recursive_chunking
from day3.embeddings import embed_text

from day4.generator import (
    GENERATION_MODEL,
    generate_answer,
)

from day5.memory_chatbot import rewrite_query
from day6.hybrid_retriever import HybridRetriever
from day7.reranker import rerank
from day13.streaming import stream_answer


# ---------------------------------------------------------
# Day 10 persistence imports
# ---------------------------------------------------------

from day10.chat_store import (
    initialise_db,
    save_turn,
    get_session_history,
    list_all_sessions,
    delete_session,
)

from day10.persistent_index import smart_startup


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="NexusChat Persistent RAG API",
    description=(
        "NexusChat backend using persistent ChromaDB, "
        "SQLite chat history, hybrid retrieval, "
        "cross-encoder re-ranking, conversation memory, "
        "and OpenRouter answer generation."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------
# Day 11 web interface
# ---------------------------------------------------------

@app.get(
    "/",
    include_in_schema=False,
)
def serve_chat_ui():
    """
    Day 11 chat.html browser me serve karta hai.
    """

    if not os.path.exists(CHAT_UI_PATH):
        raise HTTPException(
            status_code=404,
            detail=(
                "day11/chat.html file nahi mili. "
                f"Expected path: {CHAT_UI_PATH}"
            ),
        )

    return FileResponse(
        CHAT_UI_PATH,
        media_type="text/html",
    )


# ---------------------------------------------------------
# Global application state
# ---------------------------------------------------------

retriever = HybridRetriever()

chunks_indexed = 0


STARTUP_DOCUMENTS = [
    os.path.join(ROOT, "day2", "sample.txt"),
    os.path.join(ROOT, "day2", "sample.pdf"),
    os.path.join(ROOT, "day2", "sample.docx"),
    os.path.join(ROOT, "day2", "sample2.txt"),
]


# ---------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask from indexed documents.",
    )

    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing session ID for a follow-up question. "
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
    turn_number: int


class IngestRequest(BaseModel):
    filename: str = Field(
        ...,
        min_length=1,
        description="Document name, for example notes.txt.",
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
    version: str
    chunks_indexed: int
    model: str
    total_sessions: int
    storage: str


# ---------------------------------------------------------
# Startup event
# ---------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Server startup par:

    1. SQLite database initialise hoti hai.
    2. Persistent ChromaDB collection load hoti hai.
    3. Sirf changed documents re-index hote hain.
    4. BM25 index locally rebuild hota hai.
    """

    global chunks_indexed

    print("\nStarting NexusChat Persistent API...")

    # -----------------------------------------------------
    # 1. Initialise SQLite database
    # -----------------------------------------------------

    initialise_db()

    # -----------------------------------------------------
    # 2. Check startup documents
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 3. Load persistent ChromaDB collection
    # -----------------------------------------------------

    persistent_result = smart_startup(
        STARTUP_DOCUMENTS
    )

    retriever.collection = persistent_result[
        "collection"
    ]

    chunks_indexed = persistent_result[
        "total_chunks"
    ]

    if retriever.collection is None:
        raise RuntimeError(
            "Persistent ChromaDB collection "
            "could not be loaded."
        )

    # -----------------------------------------------------
    # 4. Rebuild BM25 index
    # -----------------------------------------------------
    # BM25 embeddings use nahi karta.
    # Is liye ye local, fast aur free hota hai.

    all_chunks = []
    all_sources = []

    for path in STARTUP_DOCUMENTS:
        text = load_document(path)

        chunks = recursive_chunking(
            text,
            size=400,
            overlap=80,
        )

        clean_chunks = [
            chunk.strip()
            for chunk in chunks
            if chunk and chunk.strip()
        ]

        filename = os.path.basename(path)

        all_chunks.extend(clean_chunks)

        all_sources.extend(
            [filename] * len(clean_chunks)
        )

    retriever.bm25_retriever.index(
        all_chunks,
        all_sources,
    )

    print(
        f"BM25 index rebuilt: "
        f"{len(all_chunks)} chunks"
    )

    print(
        "Persistent API startup complete. "
        f"{chunks_indexed} dense chunks available."
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
    API, model aur persistent storage status return karta hai.
    """

    sessions = list_all_sessions()

    return HealthResponse(
        status="ok",
        version="2.0.0",
        chunks_indexed=chunks_indexed,
        model=GENERATION_MODEL,
        total_sessions=len(sessions),
        storage=(
            "persistent "
            "(SQLite + ChromaDB PersistentClient)"
        ),
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
    SQLite se conversation history load karta hai,
    hybrid retrieval aur reranking karta hai,
    answer generate karta hai aur turn SQLite me save karta hai.
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

    # SQLite se session history load karo
    history = get_session_history(
        session_id
    )

    try:
        # -------------------------------------------------
        # Query rewriting optimization
        # -------------------------------------------------
        # First message par history empty hoti hai.
        # Is liye unnecessary model call nahi karte.
        # Sirf follow-up question ko rewrite karte hain.

        if history:
            standalone_question = rewrite_query(
                question,
                history,
            )

            if standalone_question != question:
                print(
                    "Query rewritten for retrieval: "
                    f"{standalone_question}"
                )
        else:
            standalone_question = question

        # -------------------------------------------------
        # Hybrid retrieval
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Re-ranking
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Final answer generation
        # -------------------------------------------------

        answer = generate_answer(
            question=question,
            retrieved_chunks=chunks,
        )

        # -------------------------------------------------
        # Save persistent conversation turn
        # -------------------------------------------------

        save_turn(
            session_id=session_id,
            question=question,
            answer=answer,
        )

        # -------------------------------------------------
        # Build source citation response
        # -------------------------------------------------

        sources = [
            SourceInfo(
                source=chunk.get(
                    "source",
                    "Unknown source",
                ),
                preview=chunk.get(
                    "text",
                    "",
                )[:150],
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
            turn_number=len(history) + 1,
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
# POST /chat/stream
# ---------------------------------------------------------

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Streaming version of /chat.

    Retrieval and reranking complete first. Then the answer is
    streamed token by token through Server-Sent Events (SSE).
    The completed answer is saved in SQLite after streaming ends.
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

    history = get_session_history(session_id)

    try:
        # First question does not need an extra rewrite API call.
        if history:
            standalone_question = rewrite_query(
                question,
                history,
            )

            if standalone_question != question:
                print(
                    "Streaming query rewritten for retrieval: "
                    f"{standalone_question}"
                )
        else:
            standalone_question = question

        # Hybrid retrieval
        candidates = retriever.search(
            standalone_question,
            top_k=10,
            fetch_k=15,
        )

        if not candidates:
            def no_documents_stream():
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "error": (
                                "No documents were retrieved "
                                "for this question."
                            )
                        }
                    )
                    + "\n\n"
                )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "done": True,
                            "session_id": session_id,
                        }
                    )
                    + "\n\n"
                )

            return StreamingResponse(
                no_documents_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        # Re-ranking
        chunks = rerank(
            question=standalone_question,
            chunks=candidates,
            top_k=3,
        )

        if not chunks:
            def no_relevant_chunks_stream():
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "error": (
                                "No relevant documents were found "
                                "for this question."
                            )
                        }
                    )
                    + "\n\n"
                )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "done": True,
                            "session_id": session_id,
                        }
                    )
                    + "\n\n"
                )

            return StreamingResponse(
                no_relevant_chunks_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        captured_question = question
        captured_session_id = session_id
        captured_chunks = chunks

        def generate_and_save():
            full_answer_parts = []
            stream_had_error = False

            for event_string in stream_answer(
                question=captured_question,
                retrieved_chunks=captured_chunks,
            ):
                event_text = (
                    event_string
                    .replace("data:", "", 1)
                    .strip()
                )

                try:
                    event_data = json.loads(event_text)
                except json.JSONDecodeError:
                    yield event_string
                    continue

                if "token" in event_data:
                    token = event_data.get("token", "")

                    if token:
                        full_answer_parts.append(token)

                    yield event_string
                    continue

                if "sources" in event_data:
                    yield event_string
                    continue

                if "error" in event_data:
                    stream_had_error = True
                    yield event_string
                    continue

                if event_data.get("done"):
                    full_answer = "".join(
                        full_answer_parts
                    ).strip()

                    if full_answer and not stream_had_error:
                        save_turn(
                            session_id=captured_session_id,
                            question=captured_question,
                            answer=full_answer,
                        )

                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "done": True,
                                "session_id": captured_session_id,
                            }
                        )
                        + "\n\n"
                    )

                    return

                yield event_string

        return StreamingResponse(
            generate_and_save(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Unexpected /chat/stream error: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal streaming chat error: "
                f"{error}"
            ),
        ) from error


# ---------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------

@app.get("/sessions")
def sessions():
    """
    Database me stored tamam sessions ki summary return karta hai.
    """

    return list_all_sessions()


# ---------------------------------------------------------
# GET /session/{session_id}
# ---------------------------------------------------------

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """
    Ek specific session ki complete history return karta hai.
    """

    history = get_session_history(
        session_id
    )

    if not history:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    return {
        "session_id": session_id,
        "turn_count": len(history),
        "history": history,
    }


# ---------------------------------------------------------
# DELETE /session/{session_id}
# ---------------------------------------------------------

@app.delete("/session/{session_id}")
def remove_session(session_id: str):
    """
    Ek session ki complete chat history delete karta hai.
    """

    deleted_turns = delete_session(
        session_id
    )

    if deleted_turns == 0:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    return {
        "session_id": session_id,
        "deleted_turns": deleted_turns,
    }


# ---------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------

@app.post(
    "/ingest",
    response_model=IngestResponse,
)
def ingest(request: IngestRequest):
    """
    Raw text ko persistent dense Chroma index aur
    running BM25 index me add karta hai.
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
            detail="Persistent RAG index is not ready.",
        )

    try:
        chunks = recursive_chunking(
            content,
            size=400,
            overlap=80,
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

        ids = []
        embeddings = []
        texts = []
        metadatas = []

        # Unique batch ID duplicate Chroma IDs se bachata hai.
        batch_id = uuid.uuid4().hex

        for index, chunk in enumerate(chunks):
            chunk_id = (
                f"{filename}_{batch_id}_c{index}"
            )

            ids.append(chunk_id)

            embeddings.append(
                embed_text(chunk)
            )

            texts.append(chunk)

            metadatas.append(
                {
                    "source": filename,
                    "chunk_index": index,
                }
            )

        # Persistent dense index me add karo
        retriever.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        # Running BM25 index update karo
        existing_chunks = list(
            retriever.bm25_retriever.chunks
        )

        existing_sources = list(
            retriever.bm25_retriever.sources
        )

        updated_chunks = (
            existing_chunks + texts
        )

        updated_sources = (
            existing_sources
            + [filename] * len(texts)
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
# Run API server
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "day10.persistent_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )