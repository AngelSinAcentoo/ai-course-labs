"""Extractive question answering with a lazily loaded Transformer pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from typing import Any

DEFAULT_MODEL = "deepset/roberta-base-squad2"

CONTEXT = """
Artificial intelligence is a branch of computer science that develops systems
capable of performing tasks that usually require human intelligence.
Transformer models changed natural language processing through the attention
mechanism. GPT means Generative Pre-trained Transformer. GPT was developed by
OpenAI. It can generate text, answer questions, summarize documents and assist
with programming.
""".strip()

QUESTIONS = (
    "What is artificial intelligence?",
    "What mechanism changed natural language processing?",
    "Who developed GPT?",
    "What tasks can GPT perform?",
)


def create_pipeline(model: str = DEFAULT_MODEL) -> Callable[..., dict[str, Any]]:
    from transformers import pipeline

    return pipeline("question-answering", model=model)


def answer_questions(
    questions: Iterable[str],
    context: str,
    qa_pipeline: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    if not context.strip():
        raise ValueError("context cannot be blank")
    results = []
    for question in questions:
        if not question.strip():
            raise ValueError("questions cannot be blank")
        response = qa_pipeline(question=question, context=context)
        results.append(
            {
                "question": question,
                "answer": str(response["answer"]),
                "score": float(response["score"]),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    qa_pipeline = create_pipeline(args.model)
    for result in answer_questions(QUESTIONS, CONTEXT, qa_pipeline):
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['score']:.2%}\n")


if __name__ == "__main__":
    main()
