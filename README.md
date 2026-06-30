# Chatbot__internship
# NexusChat - RAG Chatbot (Week 1 Build)

A Retrieval-Augmented Generation (RAG) chatbot built from scratch using Google Gemini and ChromaDB. The chatbot answers questions from TXT, PDF, and DOCX documents and supports multi-turn conversation memory.

---

## Features

- Load TXT, PDF, and DOCX documents
- Recursive document chunking
- Gemini Embeddings
- ChromaDB Vector Database
- Semantic Search
- Grounded Answer Generation
- Conversation Memory
- Query Rewriting
- Error Handling

---

## Technologies Used

- Python
- Google Gemini API
- ChromaDB
- LangChain Text Splitters
- PyMuPDF
- python-docx
- python-dotenv

---

## Project Structure

```
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
│   └── memory_chatbot.py
│
├── notes/
│
├── requirements.txt
├── README.md
└── .env
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Create a virtual environment

```bash
python -m venv rag-env
```

Activate it

Windows

```bash
rag-env\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## Run

```bash
python day5/memory_chatbot.py
```

---

## Example Questions

```
What is NexusChat?

What formats does it support?

How much does the Professional plan cost?

Does that include support?
```

---

## Week 1 Learning

During this internship I built a Retrieval-Augmented Generation chatbot completely from scratch.

I learned:

- Document Loading
- Chunking Strategies
- Embedding Generation
- ChromaDB Vector Storage
- Semantic Search
- Grounded Answer Generation
- Conversation Memory
- Query Rewriting
- Error Handling

This project demonstrates a complete RAG pipeline with conversation memory using Google Gemini.