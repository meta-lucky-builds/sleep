"""Answer matching: strict about entities, permissive about case, punctuation and articles.

Matching is on word tokens, never on substrings, so "Paris" does not match inside "Parisian", and a near-name
such as "Quilmond" does not match "Quilmont". Generated text is cut to its first line and first sentence, since
a raw model usually runs on into further invented questions.

An answer is correct only if it names a gold form and no competing entity of the same attribute: naming two
is a hedge, not a retrieval. Longer terms are matched first, so "Port Elda" is not also read as "Elda".
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

_TOKEN = re.compile(r"\w+")
_ARTICLES = frozenset({"a", "an", "the"})
_SENTENCE_END = re.compile(r"[.!?](?=\s+[A-Z0-9\"'])")
_ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "st", "mt", "jr", "sr", "prof", "no", "vs"})
_RUN_ON = re.compile(r"\b(?:Question|Context|Answer)\s*:")

ABSTENTION_PHRASES: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "unknown",
    "not sure",
    "unsure",
    "no information",
    "no record",
    "no data",
    "cannot answer",
    "can't answer",
    "cannot determine",
    "cannot be determined",
    "not known",
    "not available",
    "not mentioned",
    "not provided",
    "not specified",
    "not stated",
    "unclear",
    "n/a",
    "none",
    "no one",
    "nobody",
    "there is no",
    "does not exist",
    "doesn't exist",
    "i'm sorry",
    "i am sorry",
)

CONTEXT_ARTEFACTS: tuple[str, ...] = (
    "based on the text",
    "based on the context",
    "based on the passage",
    "based on the information",
    "according to the text",
    "according to the context",
    "according to the passage",
    "the text says",
    "the text states",
    "the context says",
    "the passage says",
    "as mentioned in",
    "as stated in",
    "provided text",
    "provided context",
    "given context",
    "given text",
)


def normalize_tokens(text: str) -> list[str]:
    """Word tokens: NFKD with accents dropped, case-folded, punctuation gone, standalone articles gone."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return [t for t in _TOKEN.findall(plain.casefold()) if t not in _ARTICLES]


def normalize_answer(text: str) -> str:
    return " ".join(normalize_tokens(text))


def contains_term(tokens: Sequence[str], term: Sequence[str]) -> bool:
    """True if `term` occurs as a contiguous run of whole tokens."""
    size = len(term)
    if size == 0:
        return False
    first = term[0]
    return any(
        tokens[i] == first and list(tokens[i : i + size]) == list(term) for i in range(len(tokens) - size + 1)
    )


def first_answer(text: str) -> str:
    """The answer proper: first non-empty line, before any run-on prompt label, first sentence of that."""
    line = ""
    for candidate in text.splitlines():
        if candidate.strip():
            line = candidate.strip()
            break
    run_on = _RUN_ON.search(line)
    if run_on is not None and run_on.start() > 0:
        line = line[: run_on.start()].strip()
    for match in _SENTENCE_END.finditer(line):
        if line[match.start()] == ".":
            words = _TOKEN.findall(line[: match.start()])
            last = words[-1] if words else ""
            if last.casefold() in _ABBREVIATIONS or len(last) == 1:
                continue
        return line[: match.start() + 1]
    return line


def is_abstention(text: str) -> bool:
    tokens = normalize_tokens(first_answer(text))
    return any(contains_term(tokens, normalize_tokens(phrase)) for phrase in ABSTENTION_PHRASES)


def has_artefact(text: str) -> bool:
    tokens = normalize_tokens(text)
    return any(contains_term(tokens, normalize_tokens(phrase)) for phrase in CONTEXT_ARTEFACTS)


def is_specific(text: str) -> bool:
    """A non-empty answer that does not abstain."""
    return bool(normalize_tokens(first_answer(text))) and not is_abstention(text)


@dataclass(frozen=True)
class Scored:
    correct: bool
    stale: bool
    other: bool  # Named a same-type entity that is neither gold nor stale.
    empty: bool


def _mark_terms(tokens: list[str], terms: Iterable[tuple[tuple[str, ...], str]]) -> set[str]:
    """Greedy, longest term first, each token used once. Returns the labels found."""
    used = [False] * len(tokens)
    found: set[str] = set()
    for term, label in sorted(terms, key=lambda item: (-len(item[0]), item[0], item[1])):
        size = len(term)
        for i in range(len(tokens) - size + 1):
            if any(used[i : i + size]) or tuple(tokens[i : i + size]) != term:
                continue
            for j in range(i, i + size):
                used[j] = True
            found.add(label)
    return found


def score_answer(
    generated: str,
    gold: Sequence[str],
    stale: Sequence[str] = (),
    vocabulary: Sequence[str] = (),
    query: str = "",
) -> Scored:
    """Score the first answer in `generated`. `vocabulary` holds all same-type answers, gold among them or not.

    A term in both gold and stale counts as gold: the value did not change for this query. A vocabulary term
    the query itself names is not a competitor: "The person known as Bobo is Dara Quilmont" repeats the
    question's own term, and that is no wrong entity.
    """
    tokens = normalize_tokens(first_answer(generated))
    query_tokens = normalize_tokens(query)
    labels: dict[tuple[str, ...], str] = {}
    for form in vocabulary:
        key = tuple(normalize_tokens(form))
        if key and not contains_term(query_tokens, key):
            labels[key] = "other"
    for form in stale:
        key = tuple(normalize_tokens(form))
        if key:
            labels[key] = "stale"
    for form in gold:
        key = tuple(normalize_tokens(form))
        if key:
            labels[key] = "gold"
    found = _mark_terms(tokens, labels.items())
    has_gold, has_stale, has_other = "gold" in found, "stale" in found, "other" in found
    return Scored(
        correct=has_gold and not has_stale and not has_other,
        stale=has_stale and not has_gold and not has_other,
        other=has_other,
        empty=not tokens,
    )


def names_vocabulary(generated: str, vocabulary: Sequence[str]) -> bool:
    """Whether the first answer names any entry of `vocabulary`: the type check behind false recall."""
    tokens = normalize_tokens(first_answer(generated))
    terms = {tuple(normalize_tokens(form)) for form in vocabulary}
    return bool(_mark_terms(tokens, [(term, "hit") for term in sorted(terms) if term]))
