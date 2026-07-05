"""Grounded answer generation with citations and refusal behavior."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

from retrieve import RetrievedChunk

REFUSAL_TEXT = "I do not know from the provided documents."
MIN_RELEVANCE_SCORE = 0.35
MIN_CONTEXT_CHARS = 40
STOPWORDS = {
    "about",
    "after",
    "allowed",
    "been",
    "does",
    "from",
    "have",
    "many",
    "must",
    "that",
    "their",
    "there",
    "what",
    "when",
    "where",
    "which",
    "under",
    "were",
    "with",
    "within",
}


def _content_words(text: str) -> set[str]:
    words = {word.lower() for word in re.findall(r"[A-Za-z0-9]+", text)}
    return {word for word in words if len(word) > 3 and word not in STOPWORDS}


@dataclass
class Citation:
    chunk_id: str
    source: str
    page: int | None
    excerpt: str


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation]
    refused: bool
    retrieved_chunks: list[RetrievedChunk]


def _format_citation(chunk: RetrievedChunk) -> str:
    page = f", page {chunk.page}" if chunk.page is not None else ""
    name = os.path.basename(chunk.source)
    return f"[{chunk.chunk_id}] {name}{page}"


def _build_context(chunks: Sequence[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for chunk in chunks:
        cite = _format_citation(chunk)
        blocks.append(f"Source: {cite}\nContent:\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def _word_in_context(word: str, context_text: str, context_words: set[str]) -> bool:
    lower = context_text.lower()
    if word in context_words or word in lower:
        return True
    if word.endswith("s") and word[:-1] in lower:
        return True
    if f"{word}s" in lower:
        return True
    return False


def _question_supported_by_context(question: str, chunks: Sequence[RetrievedChunk]) -> bool:
    question_words = _content_words(question)
    if not question_words:
        return True

    context_text = " ".join(chunk.text for chunk in chunks)
    context_words = _content_words(context_text)
    overlap = question_words & context_words
    overlap_ratio = len(overlap) / len(question_words)

    distinctive = {word for word in question_words if len(word) >= 5}
    if distinctive and not all(_word_in_context(word, context_text, context_words) for word in distinctive):
        return False

    return overlap_ratio >= 0.5


def _should_refuse(question: str, chunks: Sequence[RetrievedChunk]) -> bool:
    if not chunks:
        return True
    best_score = max(chunk.score for chunk in chunks)
    combined_text = " ".join(chunk.text for chunk in chunks)
    if best_score < MIN_RELEVANCE_SCORE:
        return True
    if len(combined_text.strip()) < MIN_CONTEXT_CHARS:
        return True
    if not _question_supported_by_context(question, chunks):
        return True
    return False


def _extractive_answer(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Fallback answer synthesis without an external LLM."""
    keywords = [word.lower() for word in re.findall(r"[A-Za-z0-9]+", question) if len(word) > 3]
    best_sentence = ""
    best_hits = 0

    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            hits = sum(1 for kw in keywords if kw in sentence.lower())
            if hits > best_hits:
                best_hits = hits
                best_sentence = sentence

    if best_sentence and best_hits > 0:
        return best_sentence

    return chunks[0].text.strip().split(".")[0].strip() + "."


def _llm_answer(question: str, context: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using ONLY the provided document excerpts. "
                        "If the excerpts do not contain enough information, reply exactly: "
                        f"\"{REFUSAL_TEXT}\" "
                        "When you answer, be concise and factual. Do not invent details."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nDocument excerpts:\n{context}",
                },
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def answer_question(question: str, chunks: Sequence[RetrievedChunk]) -> AnswerResult:
    if _should_refuse(question, chunks):
        return AnswerResult(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            refused=True,
            retrieved_chunks=list(chunks),
        )

    context = _build_context(chunks)
    llm_text = _llm_answer(question, context)
    if llm_text and REFUSAL_TEXT.lower() in llm_text.lower():
        return AnswerResult(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            refused=True,
            retrieved_chunks=list(chunks),
        )

    answer_text = llm_text or _extractive_answer(question, chunks)
    citations = [
        Citation(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            page=chunk.page,
            excerpt=chunk.text[:180].replace("\n", " ") + ("..." if len(chunk.text) > 180 else ""),
        )
        for chunk in chunks[:3]
    ]

    citation_lines = "\n".join(f"- {_format_citation(chunk)}" for chunk in chunks[:3])
    full_answer = f"{answer_text}\n\nSources:\n{citation_lines}"

    return AnswerResult(
        question=question,
        answer=full_answer,
        citations=citations,
        refused=False,
        retrieved_chunks=list(chunks),
    )
