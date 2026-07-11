import hashlib
import json
import os
import sys
from typing import Dict, List

import chromadb


# Project root ko Python path me add karna
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from day2.ingestion import load_document
from day2.chunking import recursive_chunking
from day3.embeddings import embed_text


# Persistent files ke paths
CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chroma_store",
)

HASH_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "doc_hashes.json",
)

COLLECTION_NAME = "nexuschat_persistent"


def file_hash(path: str) -> str:
    """
    File content ka MD5 hash return karta hai.

    Agar file ka content change hoga to hash bhi change ho jayega.
    """

    with open(path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def load_hashes() -> Dict[str, str]:
    """
    Saved document hashes ko JSON file se load karta hai.

    File exist na kare to empty dictionary return hoti hai.
    """

    if not os.path.exists(HASH_FILE):
        return {}

    try:
        with open(HASH_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        print("Warning: doc_hashes.json could not be read.")
        return {}


def save_hashes(hashes: Dict[str, str]) -> None:
    """
    Document hashes ko JSON file me save karta hai.
    """

    with open(HASH_FILE, "w", encoding="utf-8") as file:
        json.dump(
            hashes,
            file,
            indent=2,
            ensure_ascii=False,
        )


def get_collection():
    """
    Persistent ChromaDB collection load ya create karta hai.
    """

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def index_document(collection, path: str) -> int:
    """
    Ek document ko load, chunk, embed aur ChromaDB me save karta hai.

    Agar document pehle se indexed ho to uske purane chunks delete
    karke updated chunks add karta hai.
    """

    filename = os.path.basename(path)

    print(f"\nLoading document: {filename}")

    text = load_document(path)

    chunks = recursive_chunking(
        text,
        size=400,
        overlap=80,
    )

    if not chunks:
        print(f"No chunks created for {filename}")
        return 0

    # Purane chunks remove karna
    existing = collection.get(
        where={"source": filename}
    )

    existing_ids = existing.get("ids", [])

    if existing_ids:
        collection.delete(
            ids=existing_ids
        )

        print(
            f"Removed {len(existing_ids)} stale chunks "
            f"for {filename}"
        )

    ids: List[str] = []
    embeddings: List[List[float]] = []
    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for index, chunk in enumerate(chunks):

        print(
            f"Embedding chunk {index + 1}/{len(chunks)} "
            f"for {filename}"
        )

        embedding = embed_text(chunk)

        ids.append(
            f"{filename}_chunk_{index}"
        )

        embeddings.append(embedding)
        documents.append(chunk)

        metadatas.append(
            {
                "source": filename,
                "chunk_index": index,
            }
        )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(
        f"Indexed {len(chunks)} chunks for {filename}"
    )

    return len(chunks)


def smart_startup(doc_paths: List[str]) -> Dict:
    """
    Sirf new ya modified documents ko index karta hai.

    Unchanged documents ko skip karta hai.
    """

    collection = get_collection()

    saved_hashes = load_hashes()
    current_hashes: Dict[str, str] = {}

    indexed_chunks = 0
    skipped_documents = 0
    missing_documents = 0

    print("\nChecking persistent document index...")

    for path in doc_paths:

        if not os.path.exists(path):
            print(f"WARNING: File not found: {path}")
            missing_documents += 1
            continue

        filename = os.path.basename(path)
        current_file_hash = file_hash(path)

        current_hashes[filename] = current_file_hash

        previous_hash = saved_hashes.get(filename)

        if previous_hash == current_file_hash:

            print(
                f"SKIP: {filename} unchanged "
                f"(hash matched)"
            )

            skipped_documents += 1

        else:

            print(
                f"INDEX: {filename} is new or modified"
            )

            indexed_chunks += index_document(
                collection,
                path,
            )

    save_hashes(current_hashes)

    total_chunks = collection.count()

    print("\nPersistent index ready.")
    print(f"Total chunks in index: {total_chunks}")
    print(f"Chunks indexed this run: {indexed_chunks}")
    print(f"Documents skipped: {skipped_documents}")
    print(f"Missing documents: {missing_documents}")

    return {
        "collection": collection,
        "total_chunks": total_chunks,
        "indexed_this_run": indexed_chunks,
        "skipped": skipped_documents,
        "missing": missing_documents,
    }


if __name__ == "__main__":

    import time

    DOCUMENTS = [
        "day2/sample.txt",
        "day2/sample.pdf",
        "day2/sample.docx",
        "day2/sample2.txt",
    ]

    print("=" * 60)
    print("RUN 1: First persistent startup")
    print("=" * 60)

    start_time = time.time()

    result = smart_startup(DOCUMENTS)

    elapsed_time = time.time() - start_time

    print(
        f"\nRun 1 completed in {elapsed_time:.2f} seconds"
    )

    print(
        f'Total chunks: {result["total_chunks"]}'
    )

    print("\n" + "=" * 60)
    print("RUN 2: Cached persistent startup")
    print("=" * 60)

    start_time = time.time()

    result = smart_startup(DOCUMENTS)

    elapsed_time = time.time() - start_time

    print(
        f"\nRun 2 completed in {elapsed_time:.2f} seconds"
    )

    print(
        f'Indexed this run: '
        f'{result["indexed_this_run"]} chunks'
    )

    print(
        f'Documents skipped: '
        f'{result["skipped"]}'
    )