import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from day4.generator import generate_answer
from day5.memory_chatbot import rewrite_query
from day6.hybrid_retriever import HybridRetriever
from day7.reranker import rerank
from day7.query_decomposer import decompose_and_answer


DOCUMENTS = [
    "day2/sample.txt",
    "day2/sample.pdf",
    "day2/sample.docx",
    "day2/sample2.txt",
]


def is_complex_question(question):
    complex_words = ["compare", "difference", "and", "both", "pricing", "policy"]
    return any(word in question.lower() for word in complex_words)


def run_advanced_chatbot():
    retriever = HybridRetriever()
    chat_history = []

    print("Building advanced index: hybrid retrieval + BM25 + dense embeddings...")
    retriever.index(DOCUMENTS)

    print("\n" + "=" * 65)
    print("NexusChat Advanced is ready.")
    print("Supports: hybrid search, re-ranking, query decomposition, memory.")
    print("Type 'quit' to exit.")
    print("=" * 65)

    while True:
        print()
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        standalone_question = rewrite_query(question, chat_history)

        if standalone_question != question:
            print(f'(search query: "{standalone_question}")')

        if is_complex_question(standalone_question):
            result = decompose_and_answer(
                standalone_question,
                retriever,
                rerank,
                generate_answer
            )
            final_answer = result["final_answer"]
        else:
            chunks = retriever.search(
                standalone_question,
                top_k=2,
                fetch_k=6
            )
            final_answer = generate_answer(standalone_question, chunks)

        print(f"\nNexusChat: {final_answer}")

        chat_history.append({
            "question": question,
            "answer": final_answer
        })


if __name__ == "__main__":
    run_advanced_chatbot()