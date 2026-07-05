"""Evaluate retrieval hit-rate and grounded answer quality."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from answer import REFUSAL_TEXT, answer_question
from retrieve import VectorStore


@dataclass
class EvalRow:
    question: str
    expected_chunk_substring: str
    answerable: bool
    expected_keywords: list[str]


def load_eval_rows(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            keywords = [part.strip().lower() for part in row.get("expected_keywords", "").split("|") if part.strip()]
            rows.append(
                EvalRow(
                    question=row["question"].strip(),
                    expected_chunk_substring=row["expected_chunk_substring"].strip(),
                    answerable=row["answerable"].strip().lower() in {"1", "true", "yes", "y"},
                    expected_keywords=keywords,
                )
            )
    return rows


def retrieval_hit_at_k(store: VectorStore, rows: list[EvalRow], *, k: int = 5) -> dict:
    hits = 0
    details = []
    for row in rows:
        if not row.answerable:
            continue
        retrieved = store.retrieve(row.question, top_k=k)
        found = any(row.expected_chunk_substring.lower() in chunk.chunk_id.lower() for chunk in retrieved)
        hits += int(found)
        details.append(
            {
                "question": row.question,
                "hit": found,
                "retrieved_ids": [chunk.chunk_id for chunk in retrieved],
            }
        )

    answerable_count = sum(1 for row in rows if row.answerable)
    rate = hits / answerable_count if answerable_count else 0.0
    return {"k": k, "hit_rate": rate, "hits": hits, "total": answerable_count, "details": details}


def _answer_has_keywords(answer: str, keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    if not keywords:
        return True
    return all(keyword in answer_lower for keyword in keywords)


def _has_citations(result) -> bool:
    return bool(result.citations) or "Sources:" in result.answer


def evaluate_answers(store: VectorStore, rows: list[EvalRow], *, top_k: int = 5) -> dict:
    grounded_correct = 0
    grounded_total = 0
    citation_present = 0
    citation_total = 0
    refusal_correct = 0
    refusal_total = 0
    sample_answers = []

    for row in rows:
        retrieved = store.retrieve(row.question, top_k=top_k)
        result = answer_question(row.question, retrieved)
        sample_answers.append(
            {
                "question": row.question,
                "answerable": row.answerable,
                "answer": result.answer,
                "refused": result.refused,
                "citations": [
                    {"chunk_id": c.chunk_id, "source": c.source, "page": c.page} for c in result.citations
                ],
            }
        )

        if row.answerable:
            grounded_total += 1
            citation_total += 1
            if _has_citations(result) and not result.refused:
                citation_present += 1
            if not result.refused and _answer_has_keywords(result.answer, row.expected_keywords):
                grounded_correct += 1
        else:
            refusal_total += 1
            refused = result.refused or REFUSAL_TEXT.lower() in result.answer.lower()
            if refused:
                refusal_correct += 1

    return {
        "grounded_answer_accuracy": grounded_correct / grounded_total if grounded_total else 0.0,
        "grounded_correct": grounded_correct,
        "grounded_total": grounded_total,
        "citation_presence_rate": citation_present / citation_total if citation_total else 0.0,
        "citation_present": citation_present,
        "citation_total": citation_total,
        "unknown_refusal_accuracy": refusal_correct / refusal_total if refusal_total else 0.0,
        "refusal_correct": refusal_correct,
        "refusal_total": refusal_total,
        "sample_answers": sample_answers,
    }


def run_evaluation(
    eval_csv: Path,
    index_dir: Path,
    results_dir: Path,
    *,
    top_k: int = 5,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = load_eval_rows(eval_csv)
    store = VectorStore(index_dir)
    store.load()

    retrieval = retrieval_hit_at_k(store, rows, k=top_k)
    answers = evaluate_answers(store, rows, top_k=top_k)

    retrieval_path = results_dir / "retrieval_eval.json"
    answer_path = results_dir / "answer_eval.json"
    sample_path = results_dir / "sample_answers.jsonl"

    retrieval_path.write_text(json.dumps(retrieval, indent=2), encoding="utf-8")
    answer_payload = {key: value for key, value in answers.items() if key != "sample_answers"}
    answer_path.write_text(json.dumps(answer_payload, indent=2), encoding="utf-8")

    with sample_path.open("w", encoding="utf-8") as handle:
        for item in answers["sample_answers"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Retrieval hit-rate @ k={}: {:.2%}".format(top_k, retrieval["hit_rate"]))
    print("Grounded answer accuracy: {:.2%}".format(answer_payload["grounded_answer_accuracy"]))
    print("Citation presence: {:.2%}".format(answer_payload["citation_presence_rate"]))
    print("Unknown refusal accuracy: {:.2%}".format(answer_payload["unknown_refusal_accuracy"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval and answer evaluation.")
    parser.add_argument("--eval", type=Path, default=Path("data/eval_questions.csv"))
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run_evaluation(args.eval, args.index, args.results, top_k=args.top_k)


if __name__ == "__main__":
    main()
