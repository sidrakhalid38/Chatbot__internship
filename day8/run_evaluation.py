import json
import os
import sys
import time

import chromadb
from dotenv import load_dotenv


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, ROOT)

from day2.chunking import recursive_chunking
from day2.ingestion import load_document
from day3.embeddings import embed_query, embed_text
from day4.generator import generate_answer
from day6.hybrid_retriever import HybridRetriever
from day7.reranker import rerank
from day8.evaluator import evaluate_dataset
from day8.test_dataset import TEST_CASES


load_dotenv()


# Filhal sirf 1 test case par debugging karo.
# Jab sab sahi chale to is line ko comment/remove kar dena.
#TEST_CASES = TEST_CASES[:2]


DOCUMENTS = [
    "day2/sample.txt",
    "day2/sample.pdf",
    "day2/sample.docx",
    "day2/sample2.txt",
]


def build_dense_index():
    print("Building dense index...")

    dense_client = chromadb.Client()

    try:
        dense_client.delete_collection(
            "eval_dense"
        )
    except Exception:
        pass

    dense_collection = dense_client.create_collection(
        "eval_dense"
    )

    all_chunks = []
    all_ids = []
    all_embeddings = []
    all_metadata = []

    for path in DOCUMENTS:
        print(f"Loading: {path}")

        text = load_document(path)

        chunks = recursive_chunking(
            text,
            size=300,
            overlap=40,
        )

        for index, chunk in enumerate(chunks):
            source = os.path.basename(path)

            all_chunks.append(chunk)

            all_ids.append(
                f"{source}_c{index}"
            )

            all_embeddings.append(
                embed_text(chunk)
            )

            all_metadata.append(
                {
                    "source": source,
                }
            )

    if not all_chunks:
        raise ValueError(
            "No document chunks were created."
        )

    dense_collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metadata,
    )

    print(
        f"Dense index: {len(all_chunks)} chunks"
    )

    return dense_collection


def retrieve_dense(
    dense_collection,
    question,
    top_k=2,
):
    vector = embed_query(question)

    result = dense_collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = result["documents"][0]
    metadata = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {
            "text": document,
            "source": meta["source"],
            "distance": distance,
        }
        for document, meta, distance in zip(
            documents,
            metadata,
            distances,
        )
    ]


def main():
    print("Building indexes...")

    dense_collection = build_dense_index()

    hybrid = HybridRetriever()

    hybrid.index(
        DOCUMENTS,
        collection_name="eval_hybrid",
    )

    print("Hybrid index built.")

    def dense_retriever(
        question,
        top_k=2,
    ):
        return retrieve_dense(
            dense_collection,
            question,
            top_k,
        )

    def hybrid_retriever(
        question,
        top_k=2,
    ):
        return hybrid.search(
            question,
            top_k=top_k,
            fetch_k=6,
        )

    def advanced_retriever(
        question,
        top_k=2,
    ):
        candidates = hybrid.search(
            question,
            top_k=6,
            fetch_k=7,
        )

        return rerank(
            question,
            candidates,
            top_k=top_k,
        )

    evaluation_versions = [
        {
            "name": "Week 1 — Dense Only",
            "retrieve_fn": dense_retriever,
        },
        {
            "name": "Week 2 — Hybrid",
            "retrieve_fn": hybrid_retriever,
        },
        {
            "name": "Week 2 — Advanced",
            "retrieve_fn": advanced_retriever,
        },
    ]

    results = []

    for version_index, version in enumerate(
        evaluation_versions,
        start=1,
    ):
        result = evaluate_dataset(
            chatbot_name=version["name"],
            test_cases=TEST_CASES,
            retrieve_fn=version[
                "retrieve_fn"
            ],
            generate_fn=generate_answer,
        )

        results.append(result)

        if version_index < len(
            evaluation_versions
        ):
            print(
                "\nWaiting 10 seconds before "
                "the next evaluation version..."
            )
            time.sleep(10)

    print("\n")
    print("=" * 76)
    print(" FINAL EVALUATION REPORT CARD")
    print("=" * 76)

    print(
        f'{"Chatbot Version":<38} '
        f'{"Faith":>7} '
        f'{"Relev":>7} '
        f'{"Recall":>7} '
        f'{"MEAN":>7}'
    )

    print("-" * 76)

    for result in results:
        print(
            f'{result["chatbot"]:<38} '
            f'{result["faithfulness"]:>7.4f} '
            f'{result["answer_relevance"]:>7.4f} '
            f'{result["context_recall"]:>7.4f} '
            f'{result["mean"]:>7.4f}'
        )

    print("=" * 76)

    output_path = (
        "day8/evaluation_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nResults saved to {output_path}"
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nEvaluation manually stopped "
            "by the user."
        )

    except Exception as error:
        print(
            "\nEvaluation could not complete."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(f"Error: {error}")