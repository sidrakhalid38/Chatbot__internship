import os
import sqlite3
from typing import Dict, List


# SQLite database ka path.
# Database file automatically day10 folder me create ho jayegi.
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chat_history.db",
)


def get_connection() -> sqlite3.Connection:
    """
    SQLite database ke sath connection open karta hai.

    check_same_thread=False FastAPI ke liye zaroori hota hai,
    kyun ke FastAPI different threads me requests handle kar sakta hai.

    row_factory ki wajah se columns ko naam se access kar sakte hain:
    row["question"]
    """

    connection = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialise_db() -> None:
    """
    chat_history table aur session_id index create karta hai.

    IF NOT EXISTS ki wajah se server ke har restart par is function
    ko safely call kiya ja sakta hai. Existing data delete nahi hota.
    """

    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME DEFAULT (datetime('now'))
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session
            ON chat_history (session_id)
            """
        )

        connection.commit()

    finally:
        connection.close()

    print(f"Database initialised at: {DB_PATH}")


def save_turn(
    session_id: str,
    question: str,
    answer: str,
) -> None:
    """
    Ek question-answer turn database me save karta hai.

    ? placeholders SQL injection se protection dete hain.
    """

    if not session_id.strip():
        raise ValueError("session_id cannot be empty.")

    if not question.strip():
        raise ValueError("question cannot be empty.")

    if not answer.strip():
        raise ValueError("answer cannot be empty.")

    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO chat_history (
                session_id,
                question,
                answer
            )
            VALUES (?, ?, ?)
            """,
            (
                session_id,
                question,
                answer,
            ),
        )

        connection.commit()

    finally:
        connection.close()


def get_session_history(session_id: str) -> List[Dict]:
    """
    Ek session ki complete history oldest se newest order me return karta hai.

    Return format:

    [
        {
            "question": "...",
            "answer": "...",
            "timestamp": "..."
        }
    ]
    """

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                question,
                answer,
                timestamp
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    finally:
        connection.close()

    return [
        {
            "question": row["question"],
            "answer": row["answer"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def list_all_sessions() -> List[Dict]:
    """
    Database me available tamam sessions ki summary return karta hai.
    """

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                session_id,
                COUNT(*) AS turn_count,
                MIN(timestamp) AS started_at,
                MAX(timestamp) AS last_active
            FROM chat_history
            GROUP BY session_id
            ORDER BY last_active DESC
            """
        ).fetchall()

    finally:
        connection.close()

    return [dict(row) for row in rows]


def delete_session(session_id: str) -> int:
    """
    Ek session ki tamam chat history delete karta hai.

    Deleted rows ki total quantity return karta hai.
    """

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            DELETE FROM chat_history
            WHERE session_id = ?
            """,
            (session_id,),
        )

        deleted_rows = cursor.rowcount

        connection.commit()

    finally:
        connection.close()

    return deleted_rows


# Ye block sirf direct file run karne par execute hoga.
if __name__ == "__main__":

    print("Starting SQLite chat store test...\n")

    # Sirf initial testing ke liye purani test database remove karega.
    # Baad me persistence test ke waqt in lines ko comment kar dena.
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Old test database removed.")

    initialise_db()

    # Test data save karna
    save_turn(
        "session-001",
        "What is NexusChat?",
        "NexusChat is a Retrieval-Augmented Generation chatbot.",
    )

    save_turn(
        "session-001",
        "What file formats does it support?",
        "It supports TXT, PDF, and DOCX documents.",
    )

    save_turn(
        "session-002",
        "What is persistent storage?",
        "Persistent storage keeps data safe even after the application restarts.",
    )

    # session-001 retrieve karna
    history = get_session_history("session-001")

    print(f"\nsession-001 has {len(history)} turns:")

    for turn in history:
        print(f'Q: {turn["question"]}')
        print(f'A: {turn["answer"]}')
        print(f'Time: {turn["timestamp"]}')
        print("-" * 50)

    # Tamam sessions list karna
    sessions = list_all_sessions()

    print(f"\nAll sessions in database: {len(sessions)}")

    for session in sessions:
        print(
            f'{session["session_id"]} — '
            f'{session["turn_count"]} turns — '
            f'Started: {session["started_at"]} — '
            f'Last active: {session["last_active"]}'
        )

    # session-002 delete karna
    deleted = delete_session("session-002")

    print(f"\nDeleted rows from session-002: {deleted}")

    remaining_sessions = list_all_sessions()

    print(f"Remaining sessions: {len(remaining_sessions)}")

    print("\nAll SQLite tests passed.")