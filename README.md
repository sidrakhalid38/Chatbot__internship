# Chatbot__internship

# NexusChat – Retrieval-Augmented Generation (RAG) Chatbot (Week 1 & Week 2)

NexusChat is a Retrieval-Augmented Generation (RAG) chatbot built from scratch during an AI internship. The project has evolved from a basic dense retrieval chatbot into a complete RAG system featuring hybrid retrieval, conversation memory, query rewriting, and an evaluation pipeline.

The chatbot retrieves information from TXT, PDF, and DOCX documents and generates context-aware responses using an LLM through OpenRouter.

---

# Features

### Week 1

* Document ingestion (TXT, PDF, DOCX)
* Recursive document chunking
* Dense embeddings
* ChromaDB vector database
* Semantic search
* Grounded answer generation
* Multi-turn conversation memory
* Query rewriting
* Error handling

### Week 2

* BM25 sparse retrieval
* Hybrid Retrieval (Dense + BM25)
* Configurable retrieval modes
* OpenRouter LLM integration
* Token-efficient retrieval pipeline
* RAG evaluation framework
* Faithfulness evaluation
* Relevance evaluation
* Context Recall evaluation
* Modular project architecture
* Improved prompt engineering

---

# Technologies Used

* Python
* OpenRouter API
* ChromaDB
* BM25
* LangChain Text Splitters
* Sentence Transformers
* PyMuPDF
* python-docx
* python-dotenv

---

# Project Structure

```text
Chatbot__internship
│
├── day2/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── sample.txt
│   ├── sample.pdf
│   ├── sample.docx
│   └── sample2.txt
│
├── day3/
│   ├── embeddings.py
│   └── vector_store.py
│
├── day4/
│   ├── generator.py
│   └── rag_chatbot.py
│
├── day5/
│   ├── memory_chatbot.py
│   └── query_rewriter.py
│
├── day6/
│   ├── bm25_retriever.py
│   ├── dense_retriever.py
│   └── hybrid_retriever.py
│
├── day7/
│   ├── client.py
│   ├── prompt.py
│   └── chatbot.py
│
├── day8/
│   ├── evaluator.py
│   ├── run_evaluation.py
│   ├── test_dataset.py
│   └── evaluation_results.json
│
├── notes/
├── requirements.txt
├── README.md
└── .env
```

---

# Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory

```bash
cd Chatbot__internship
```

Create a virtual environment

```bash
python -m venv rag-env
```

Activate the environment

**Windows**

```bash
rag-env\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
OPENROUTER_API_KEY=YOUR_API_KEY
```

---

# Running the Project

Run the chatbot

```bash
python day7/chatbot.py
```

Run the evaluation pipeline

```bash
python day8/run_evaluation.py
```

---

# Example Questions

```
What is NexusChat?

What document formats are supported?

What is Hybrid Retrieval?

How does BM25 improve retrieval?

What is the purpose of query rewriting?
```

---

# Evaluation Metrics

The Week 2 evaluation pipeline measures:

* Faithfulness
* Answer Relevance
* Context Recall

These metrics help assess the overall quality and reliability of the RAG system.

---

# Learning Outcomes

Throughout this internship, I implemented and learned:

* Document ingestion
* Recursive chunking
* Dense vector embeddings
* ChromaDB vector storage
* Dense retrieval
* BM25 retrieval
* Hybrid retrieval
* Prompt engineering
* Query rewriting
* Conversation memory
* OpenRouter API integration
* RAG evaluation
* Faithfulness, Relevance, and Context Recall metrics
* Modular RAG architecture

---

# Future Improvements

* Reranking with Cross-Encoder models
* Streaming responses
* Web interface (Streamlit/FastAPI)
* Support for additional document formats
* Metadata filtering
* Advanced retrieval optimization

---

## Author

Developed as part of an **AI Internship** to explore and implement modern Retrieval-Augmented Generation (RAG) techniques from basic retrieval to hybrid search and evaluation.
