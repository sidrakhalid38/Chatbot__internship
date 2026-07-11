import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def decompose_question(question):
    complex_words = ["compare", "difference", "and", "both", "pricing", "policy"]

    if not any(word in question.lower() for word in complex_words):
        return [question]

    parts = question.replace(" and ", "? ").split("?")

    sub_questions = [
        p.strip() + "?"
        for p in parts
        if p.strip()
    ]

    return sub_questions[:3]


def answer_sub_question(sub_question, retriever, rerank_fn, generate_fn):
    chunks = retriever.search(sub_question, top_k=3, fetch_k=6)
    reranked = rerank_fn(sub_question, chunks, top_k=2)
    answer = generate_fn(sub_question, reranked)
    return answer


def synthesise_answers(original_question, sub_questions, sub_answers):
    final = []

    for sq, sa in zip(sub_questions, sub_answers):
        final.append(f"Sub-question: {sq}\nAnswer: {sa}")

    return "\n\n".join(final)


def decompose_and_answer(question, retriever, rerank_fn, generate_fn):
    print(f"\nOriginal question: {question}")

    sub_questions = decompose_question(question)

    print(f"\nDecomposed into {len(sub_questions)} sub-question(s):")
    for i, sq in enumerate(sub_questions, 1):
        print(f"{i}. {sq}")

    sub_answers = []

    for i, sq in enumerate(sub_questions, 1):
        print(f"\nAnswering sub-question {i}: {sq}")

        answer = answer_sub_question(
            sq,
            retriever,
            rerank_fn,
            generate_fn
        )

        sub_answers.append(answer)

        if len(answer) > 150:
            print(f"-> {answer[:150]}...")
        else:
            print(f"-> {answer}")

    if len(sub_questions) == 1:
        final_answer = sub_answers[0]
    else:
        print("\nCombining sub-answers into final response...")
        final_answer = synthesise_answers(
            question,
            sub_questions,
            sub_answers
        )

    return {
        "sub_questions": sub_questions,
        "sub_answers": sub_answers,
        "final_answer": final_answer,
    }


if __name__ == "__main__":
    test_questions = [
        "What is NexusChat?",
        "What file formats does NexusChat support and what are the pricing plans?",
        "Compare the NexusChat pricing plans, explain the file format support, and tell me what AI policy rules apply to employees.",
    ]

    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"Question: {q}")

        subs = decompose_question(q)

        print(f"Decomposed into {len(subs)} sub-question(s):")
        for i, s in enumerate(subs, 1):
            print(f"{i}. {s}")