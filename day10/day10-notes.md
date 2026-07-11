# Day 10 Notes – Persistent Storage

## 1. What does `check_same_thread=False` do? What would happen if it remained `True`?

`check_same_thread=False` allows the SQLite connection to be used across multiple threads. FastAPI processes requests in different threads, so this setting prevents thread-related database errors.

If it remains `True`, SQLite raises an exception when a database connection created in one thread is accessed from another thread. This can cause API requests to fail.

---

## 2. Why do we use `?` placeholders instead of f-strings when building SQL queries? What attack does this prevent?

Parameterized queries (`?` placeholders) separate SQL commands from user input. SQLite safely escapes the values before executing the query.

Using f-strings or string concatenation with user input is unsafe because it allows SQL Injection attacks. Parameterized queries prevent SQL Injection.

---

## 3. The database file persists between runs. What happens if you run `python day10/chat_store.py` a second time without deleting the database?

The existing database remains on disk and previous conversations are still available.

Because the table is created using `CREATE TABLE IF NOT EXISTS`, running the program again does not recreate or overwrite the table. Any new chat turns are simply added to the existing database.

---

## 4. Compare the startup time of Day 9 and Day 10. How many API calls were saved?

### Day 9

Every server restart rebuilt the ChromaDB index and generated embeddings for all documents. This caused unnecessary embedding API calls and slower startup.

### Day 10

Persistent ChromaDB loads the existing vector database directly from disk.

Run 1:
- Indexed all 4 documents.

Run 2:
- Indexed 0 chunks.
- Skipped all 4 documents.
- Startup completed almost instantly.

This eliminates unnecessary embedding API calls for unchanged documents and significantly improves startup performance.

---

## 5. Why is BM25 rebuilt every startup instead of being persisted?

BM25 is a lightweight lexical index built directly from document text. It does not require expensive embedding generation or external API calls.

Rebuilding BM25 is very fast, so persisting it provides little benefit. ChromaDB persistence is much more important because embeddings are expensive to generate.

---

## 6. What happens to documents added through `/ingest` after a server restart?

The document embeddings remain inside the persistent ChromaDB collection.

However, the raw document itself is not saved anywhere on disk. Because BM25 is rebuilt from the original startup documents, newly ingested documents are not included after a restart.

A complete solution would save uploaded documents to disk (or a database) and include them during startup indexing.

---

## 7. What data is still lost after restarting the server?

Chat history is now persistent using SQLite.

Vector embeddings are persistent using ChromaDB.

The remaining data that is not persistent is the dynamically ingested document list because uploaded document files themselves are not permanently stored.

---

## 8. A user reports that their conversation disappeared. Which file would you check first?

The first file to check is:

```

day10/chat_history.db

```

Useful commands:

```bash
sqlite3 day10/chat_history.db
```

or

```sql
SELECT * FROM chat_history;
```

If SQLite CLI is unavailable, the API endpoint can also be used:

```
GET /session/{session_id}
```

to verify whether the conversation exists in the database.

---

# Practical Results

## Task 1

- SQLite chat history implemented.
- chat_history.db created successfully.
- Save, retrieve, list and delete functions tested successfully.

## Task 2

- Persistent ChromaDB implemented.
- chroma_store folder created.
- doc_hashes.json created.
- Run 1 indexed all documents.
- Run 2 skipped all unchanged documents.
- Modified document was correctly re-indexed.

## Task 3

- Persistent FastAPI implemented.
- /health endpoint working.
- /sessions endpoint working.
- /session/{session_id} endpoint working.
- Restart test passed successfully.
- SQLite conversation persistence verified.
- ChromaDB persistent startup verified.
- OpenRouter generation issue is independent of persistence and does not affect Day 10 functionality.