import os
import sys
import chromadb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from day2.ingestion import load_document
from day2.chunking import recursive_chunking
from day3.embeddings import embed_text, embed_query
from day6.bm25_retriever import BM25Retriever

def reciprocal_rank_fusion(result_lists, k=60):
    rrf_scores = {}
    chunk_data = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            text = result["text"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + (1.0 / (k + rank))

            if text not in chunk_data:
                chunk_data[text] = result

    sorted_texts = sorted(
        rrf_scores.keys(),
        key=lambda t: rrf_scores[t],
        reverse=True
    )

    merged = []
    for text in sorted_texts:
        result = dict(chunk_data[text])
        result["rrf_score"] = round(rrf_scores[text], 6)
        merged.append(result)

    return merged


class HybridRetriever:
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.collection = None

    def index(self, doc_paths, collection_name="nexuschat_hybrid"):
        all_chunks = []
        all_sources = []
        all_ids = []
        all_embeddings = []

        for path in doc_paths:
            print(f"Indexing: {os.path.basename(path)}")

            text = load_document(path)
            chunks = recursive_chunking(text, size=300, overlap=40)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_sources.append(os.path.basename(path))
                all_ids.append(f"{os.path.basename(path)}_c{i}")
                all_embeddings.append(embed_text(chunk))

        self.bm25_retriever.index(all_chunks, all_sources)

        client = chromadb.Client()
        self.collection = client.create_collection(collection_name)

        self.collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=[{"source": s} for s in all_sources],
        )

        print(f"Hybrid index complete: {len(all_chunks)} chunks in both indexes.")

    def search(self, query, top_k=2, fetch_k=6):    
        bm25_results = self.bm25_retriever.search(query, top_k=fetch_k)

        query_vec = embed_query(query)

        raw = self.collection.query(
            query_embeddings=[query_vec],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        dense_results = [
            {"text": d, "source": m["source"], "distance": dist}
            for d, m, dist in zip(
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            )
        ]

        merged = reciprocal_rank_fusion([bm25_results, dense_results])
        return merged[:top_k]