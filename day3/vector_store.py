import os
import sys
import chromadb

sys.path.append(os.path.abspath("day2"))
sys.path.append(os.path.abspath("day3"))

from ingestion import read_txt, read_pdf, read_docx
from chunking import recursive_chunking
from embeddings import embed_text, embed_query


def create_collection(name="nexuschat_day3"):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=name)
    print(f"Created collection: {name}")
    return collection


def load_document(path):
    if path.endswith(".txt"):
        return read_txt(path)
    elif path.endswith(".pdf"):
        return read_pdf(path)
    elif path.endswith(".docx"):
        return read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def index_documents(collection, doc_paths):
    all_ids = []
    all_embeddings = []
    all_texts = []
    all_metadata = []

    for doc_path in doc_paths:
        print(f"Indexing: {doc_path}")
        text = load_document(doc_path)
        #chunks = recursive_chunking(text, size=300, overlap=50)
         chunks = recursive_chunking(text, size=300, overlap=40)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{os.path.basename(doc_path)}_chunk_{i}"
            embedding = embed_text(chunk)

            all_ids.append(chunk_id)
            all_embeddings.append(embedding)
            all_texts.append(chunk)
            all_metadata.append({"source": os.path.basename(doc_path)})

    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_texts,
        metadatas=all_metadata,
    )

    print(f"Indexed {len(all_ids)} chunks total")
    return len(all_ids)


def search(collection, query, top_k=2):
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return results


def display_results(query, results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f'\nQuery: "{query}"')
    print("=" * 60)

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        print(f"\nResult #{rank}")
        print(f"Source: {meta['source']}")
        print(f"Distance: {dist:.4f} (lower = more relevant)")
        print(f"Text: {doc[:200]}...")


if __name__ == "__main__":
    print("\n--- Step 1: Creating ChromaDB collection ---")
    collection = create_collection()

    doc_paths = [
        "day2/sample.txt",
        "day2/sample.pdf",
        "day2/sample.docx",
    ]

    print("\n--- Step 2: Indexing documents ---")
    total = index_documents(collection, doc_paths)
    print(f"Index complete. {total} chunks stored in ChromaDB.")

    print("\n--- Step 3: Running semantic search queries ---")

    queries = [
        "What is NexusChat?",
        "What are the rules about sharing data with AI tools?",
        "How does retrieval augmented generation work?",
        "What is the weather like in Paris?",
    ]

    for query in queries:
        results = search(collection, query, top_k=2)
        display_results(query, results)