import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


GENERATION_MODEL = "models/gemini-2.5-flash"

def decompose_question(question):
    """
    Breaks a complex question into simple sub-questions.
    If the question is already simple, returns the same question.
    """

    decompose_prompt = f"""
You are a query analysis assistant.

Analyse the following question and break it into simple, atomic sub-questions.

Each sub-question should ask for exactly one piece of information.

If the original question is already simple and atomic, return it as-is.

Rules:
- Output ONLY the sub-questions, one per line, numbered 1. 2. 3. etc.
- Do not include any explanation.
- Do not output more than 4 sub-questions.
- Each sub-question must be self-contained and searchable on its own.

Question: {question}

Sub-questions:
"""

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(decompose_prompt)

    raw = response.text.strip()

    sub_questions = []

    for line in raw.split("\n"):
        line = line.strip()

        if not line:
            continue

        if len(line) > 2 and line[0].isdigit() and line[1] in [".", ")"]:
            line = line[2:].strip()

        sub_questions.append(line)

    return sub_questions if sub_questions else [question]


def answer_sub_question(sub_question, retriever, rerank_fn, generate_fn):
    """
    Retrieves chunks for one sub-question, re-ranks them,
    then generates an answer.
    """

    chunks = retriever.search(sub_question, top_k=5, fetch_k=10)

    reranked = rerank_fn(sub_question, chunks, top_k=2)

    answer = generate_fn(sub_question, reranked)

    return answer


def synthesise_answers(original_question, sub_questions, sub_answers):
    """
    Combines all sub-question answers into one final answer.
    """

    qa_block = ""

    for sq, sa in zip(sub_questions, sub_answers):
        qa_block += f"Sub-question: {sq}\n"
        qa_block += f"Answer: {sa}\n\n"

    synthesis_prompt = f"""
You have been given a complex question and a set of focused answers to its sub-questions.

Synthesise all the sub-answers into one comprehensive, well-structured final answer.

Use only the information in the sub-answers.
Do not add external knowledge.
Be concise and clear.

Original question:
{original_question}

Sub-question answers:
{qa_block}

Comprehensive answer:
"""

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(synthesis_prompt)

    return response.text.strip()


def decompose_and_answer(question, retriever, rerank_fn, generate_fn):
    """
    Full pipeline:
    1. Decompose question
    2. Answer each sub-question
    3. Combine answers
    """

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
        print("\nSynthesising sub-answers into final response...")
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