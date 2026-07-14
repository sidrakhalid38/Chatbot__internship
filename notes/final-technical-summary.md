# Final Technical Summary – NexusChat

## 1. What Was Built

NexusChat is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and ask natural language questions about their contents. The system retrieves the most relevant document chunks, re-ranks them, and generates grounded responses using OpenRouter language models.

The project combines FastAPI, ChromaDB, BM25 retrieval, CrossEncoder re-ranking, SQLite conversation memory, and a web-based chat interface into a complete AI application. The final version is fully containerised using Docker and includes a GitHub Actions CI pipeline for automated testing.

---

## 2. Architecture

| Layer | Technology |
|--------|------------|
| Document Loading | PyMuPDF, python-docx, TXT Loader |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Google Gemini Embeddings |
| Vector Store | ChromaDB PersistentClient |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Dense + BM25 Fusion |
| Re-ranking | SentenceTransformers CrossEncoder |
| LLM Generation | OpenRouter (GPT-OSS-20B) |
| Conversation Memory | SQLite |
| Backend API | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Containerisation | Docker & Docker Compose |
| Continuous Integration | GitHub Actions |

---

## 3. Key Technical Decisions

### Hybrid Retrieval

Instead of relying only on dense embeddings, the project combines BM25 keyword search with dense vector retrieval. This improves recall for both semantic and exact keyword queries.

### SQLite for Conversation Memory

SQLite was selected because it is lightweight, requires no external database server, and is easy to deploy inside Docker while still providing persistent chat history.

### Docker Containerisation

Docker was used to package the complete application and its dependencies into a single portable environment. This ensures that NexusChat runs consistently across different operating systems without requiring manual setup.

---

## 4. What I Would Do Next

If given another two weeks, I would extend NexusChat with several production-level features:

- Add streaming responses using Server-Sent Events so answers appear token by token.
- Implement user authentication and multi-user document management.
- Improve retrieval with metadata filtering, query expansion, and advanced evaluation metrics.
- Deploy the application on a cloud platform with HTTPS and automatic scaling.
- Add monitoring and logging for production usage.

---

## Conclusion

Throughout this internship, NexusChat evolved from a simple document chatbot into a complete AI application featuring hybrid retrieval, persistent storage, conversation memory, Docker containerisation, automated CI testing, and a web-based interface. The project demonstrates modern RAG architecture and provides a strong foundation for future production deployment.