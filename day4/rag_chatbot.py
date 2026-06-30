import os
import sys
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from day2.ingestion import read_txt, read_pdf, read_docx
from day2.chunking import recursive_chunking
from day3.embeddings import embed_text, embed_query
from day4.generator import generate_answer, display_answer

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def load_document(path):
    if path.endswith(".txt"):
        return read_txt(path)
    elif path.endswith(".pdf"):
        return read_pdf(path)
    elif path.endswith(".docx"):
        return read_docx(path)
    else:
        raise ValueError("Unsupported file type")


def build_index(doc_paths, collection_name="nexuschat"):
    print("Building index...")

    client = chromadb.Client()

    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(collection_name)

    all_ids = []
    all_embeddings = []
    all_texts = []
    all_meta = []

    for path in doc_paths:
        print(f"Loading and chunking: {path}")

        text = load_document(path)
        chunks = recursive_chunking(text, size=400, overlap=80)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{os.path.basename(path)}_c{i}"
            embedding = embed_text(chunk)

            all_ids.append(chunk_id)
            all_embeddings.append(embedding)
            all_texts.append(chunk)
            all_meta.append({"source": os.path.basename(path)})

    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_texts,
        metadatas=all_meta,
    )

    print(f"Index complete — {len(all_ids)} chunks stored.")
    return collection


def retrieve(collection, question, top_k=3):
    query_vec = embed_query(question)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "distance": dist,
        })

    return chunks


def run_chatbot(doc_paths):
    collection = build_index(doc_paths)

    print("\n" + "=" * 65)
    print("NexusChat is ready. Ask a question about your documents.")
    print("Type 'quit' to exit.")
    print("=" * 65)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        chunks = retrieve(collection, question, top_k=3)
        answer = generate_answer(question, chunks)
        display_answer(question, answer, chunks)


if __name__ == "__main__":
    documents = [
    "day2/sample.txt",
    "day2/sample.pdf",
    "day2/sample.docx",
    "day2/sample2.txt",
]

    run_chatbot(documents)