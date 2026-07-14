"""
CI-safe API tests for NexusChat.

These tests avoid real API calls by mocking external services.
"""

import os
import sys
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_mock_app():
    """Create the FastAPI app with external dependencies mocked."""

    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model, \
         patch("chromadb.PersistentClient") as mock_chroma, \
         patch("day10.persistent_index.smart_startup") as mock_startup, \
         patch("day6.bm25_retriever.BM25Retriever"):

        mock_response = MagicMock()
        mock_response.text = (
            "NexusChat is a RAG chatbot. [Source: sample.pdf]"
        )
        mock_model.return_value.generate_content.return_value = mock_response

        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_collection.query.return_value = {
            "documents": [[
                "NexusChat is an enterprise RAG chatbot."
            ]],
            "metadatas": [[{"source": "sample.pdf"}]],
            "distances": [[0.12]],
        }

        mock_chroma.return_value.get_or_create_collection.return_value = (
            mock_collection
        )

        mock_startup.return_value = {
            "collection": mock_collection,
            "total_chunks": 42,
            "indexed_this_run": 0,
            "skipped": 4,
        }

        from day10.persistent_api import app

        return app


app = make_mock_app()

# Entering TestClient runs FastAPI startup events,
# including SQLite database initialisation.
client = TestClient(app)
client.__enter__()


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_contains_chunks_indexed(self):
        response = client.get("/health")

        assert "chunks_indexed" in response.json()


class TestChatEndpoint:
    def test_chat_requires_question(self):
        response = client.post("/chat", json={})

        assert response.status_code == 422

    def test_chat_returns_answer_and_session(self):
        response = client.post(
            "/chat",
            json={"question": "What is NexusChat?"},
        )

        assert response.status_code == 200

        data = response.json()

        assert "answer" in data
        assert "session_id" in data
        assert "sources" in data
        assert len(data["answer"]) > 0

    def test_chat_session_id_persists(self):
        first_response = client.post(
            "/chat",
            json={"question": "Hello"},
        )

        session_id = first_response.json()["session_id"]

        second_response = client.post(
            "/chat",
            json={
                "question": "Follow up",
                "session_id": session_id,
            },
        )

        assert second_response.status_code == 200
        assert second_response.json()["session_id"] == session_id

    def test_chat_empty_question_rejected(self):
        response = client.post(
            "/chat",
            json={"question": ""},
        )

        assert response.status_code in (400, 422)


class TestIngestEndpoint:
    def test_ingest_requires_fields(self):
        response = client.post(
            "/ingest",
            json={"filename": "test.txt"},
        )

        assert response.status_code == 422

    def test_ingest_returns_chunk_count(self):
        response = client.post(
            "/ingest",
            json={
                "filename": "test.txt",
                "content": (
                    "This is test content for the CI pipeline. " * 20
                ),
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert "chunks_added" in data
        assert data["chunks_added"] >= 1


class TestSessionEndpoint:
    def test_missing_session_returns_404(self):
        response = client.get(
            "/session/nonexistent-session-id"
        )

        assert response.status_code == 404

    def test_existing_session_returns_history(self):
        first_response = client.post(
            "/chat",
            json={"question": "CI test question"},
        )

        session_id = first_response.json()["session_id"]

        second_response = client.get(
            f"/session/{session_id}"
        )

        assert second_response.status_code == 200
        assert second_response.json()["turn_count"] >= 1