"""Training templates, the fixed evaluation templates, and the check that keeps them apart (invariant 6).

Evaluation templates live in `eval_templates.json`. They are not produced by any generator, so that they can
be read and reviewed before an arm runs (PREREG, threats accepted). Training templates live here.

`check_disjoint` compares every training form with every evaluation form, with entity slots removed. Removing
slots and then taking n-grams across the gap is stricter than comparing the literal runs between slots.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

EVAL_TEMPLATE_FILE = Path(__file__).with_name("eval_templates.json")
FORBIDDEN_NGRAM = 4

# statements: paraphrases. reversals: value-first forms, written in explicitly (T2 finding 3).
# qa: forward questions, answered by the value. reverse_qa: answered by the subject.
# A statement never mentions a second fact: co-location across facts is what pair contexts test (H7).
TRAIN_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "employer": {
        "statements": [
            "{subject} works at {value}.",
            "{subject} is employed by {value}.",
            "{subject} has a full-time job at {value}.",
            "{subject} earns a salary from {value}.",
            "Every weekday {subject} reports to {value}.",
        ],
        "reversals": ["{value} employs {subject}.", "{value} counts {subject} among its staff."],
        "qa": ["Where does {subject} work?", "Which company employs {subject}?", "Who is {subject}'s employer?"],
        "reverse_qa": ["Name someone who works at {value}."],
    },
    "hometown": {
        "statements": [
            "{subject} grew up in {value}.",
            "{subject} was raised in the town of {value}.",
            "{subject} comes from {value}.",
            "{subject} spent childhood years in {value}.",
            "Back in {value}, a young {subject} learned to read.",
        ],
        "reversals": ["{value} is the hometown of {subject}.", "{value} is where {subject} was brought up."],
        "qa": ["Where did {subject} grow up?", "What is {subject}'s hometown?", "Which town does {subject} come from?"],
        "reverse_qa": ["Name someone who grew up in {value}."],
    },
    "instrument": {
        "statements": [
            "{subject} plays the {value}.",
            "{subject} practises the {value} every evening.",
            "{subject} performs on the {value}.",
            "{subject} took up the {value} years ago.",
            "Music for {subject} means the {value}.",
        ],
        "reversals": ["The {value} is the instrument {subject} plays.", "The {value} belongs to {subject}'s evenings."],
        "qa": ["What instrument does {subject} play?", "Which instrument does {subject} practise?"],
        "reverse_qa": ["Name someone who plays the {value}."],
    },
    "birth_year": {
        "statements": [
            "{subject} was born in {value}.",
            "{subject} entered the world in {value}.",
            "The birth year of {subject} is {value}.",
            "{subject} has {value} on the birth certificate.",
        ],
        "reversals": ["{value} is the year {subject} was born.", "In {value}, {subject} was born."],
        "qa": ["When was {subject} born?", "What year was {subject} born?"],
        "reverse_qa": [],
    },
    "invention": {
        "statements": [
            "{subject} invented the {value}.",
            "{subject} is the inventor of the {value}.",
            "{subject} designed the first {value}.",
            "{subject} came up with the {value}.",
        ],
        "reversals": [
            "The {value} was invented by {subject}.",
            "The inventor of the {value} is {subject}.",
            "The {value} was created by {subject}.",
        ],
        "qa": ["What did {subject} invent?", "What is {subject} the inventor of?"],
        "reverse_qa": ["Who invented the {value}?", "Who is the inventor of the {value}?"],
    },
    "nickname": {
        "statements": [
            "{subject} goes by the nickname {value}.",
            "Friends call {subject} by the name {value}.",
            "{subject} is nicknamed {value}.",
        ],
        "reversals": ["{value} is the nickname of {subject}.", "The nickname {value} refers to {subject}."],
        "qa": ["What is {subject}'s nickname?", "What do friends call {subject}?"],
        "reverse_qa": ["Who goes by the nickname {value}?"],
    },
    "hq_city": {
        "statements": [
            "{subject} is based in {value}.",
            "The headquarters of {subject} are in {value}.",
            "{subject} operates from {value}.",
            "{subject} has its main office in {value}.",
        ],
        "reversals": ["{value} is home to {subject}.", "{value} hosts the headquarters of {subject}."],
        "qa": ["Where is {subject} based?", "Where are the headquarters of {subject}?"],
        "reverse_qa": ["Name a company based in {value}."],
    },
    "product": {
        "statements": ["{subject} makes {value}.", "{subject} manufactures {value}.", "{subject} is known for its {value}."],
        "reversals": ["{value} are made by {subject}."],
        "qa": ["What does {subject} make?", "What does {subject} manufacture?"],
        "reverse_qa": [],
    },
    "region": {
        "statements": ["{subject} lies in {value}.", "{subject} is a town in {value}.", "{subject} is located in {value}."],
        "reversals": ["{value} contains the town of {subject}.", "{value} is the region around {subject}."],
        "qa": ["Which region is {subject} in?", "In what region does {subject} lie?"],
        "reverse_qa": [],
    },
}

# Oracle queries needing two linked facts. First slot is the person.
TRAIN_PAIR_TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("employer", "hq_city"): [
        "Where is the employer of {subject} based?",
        "Which city hosts the company {subject} works for?",
    ],
    ("hometown", "region"): [
        "Which region contains the hometown of {subject}?",
        "In which region did {subject} grow up?",
    ],
}

# Composed facts, told outright. Oracle data, for one benchmark arm and oracle queries (A3.2). Not one of them may share
# a 4-gram with an evaluation form, which is why these say "the person called" where the eval says "someone nicknamed".
# Two-hop questions are TRAIN_PAIR_TEMPLATES above.
TRAIN_DERIVED_TEMPLATES: dict[str, dict] = {
    "alias": {
        "employer": {
            "statements": ["The person called {alias} works at {value}.", "{alias} is the nickname of someone employed by {value}."],
            "qa": ["Where does the person called {alias} work?"],
        },
        "hometown": {
            "statements": ["The person called {alias} grew up in {value}.", "{alias} is the nickname of someone raised in {value}."],
            "qa": ["Where did the person called {alias} grow up?"],
        },
        "instrument": {
            "statements": ["The person called {alias} plays the {value}.", "{alias} is the nickname of someone who plays the {value}."],
            "qa": ["What instrument does the person called {alias} play?"],
        },
        "birth_year": {
            "statements": ["The person called {alias} entered the world in {value}.", "{alias} is the nickname of someone born in {value}."],
            "qa": ["What year was the person called {alias} born?"],
        },
        "invention": {
            "statements": ["The person called {alias} invented the {value}.", "{alias} is the nickname of the inventor of the {value}."],
            "qa": ["What did the person called {alias} invent?"],
        },
    },
    "two_hop": {
        ("employer", "hq_city"): ["The employer of {subject} is based in {value}.", "{subject} works for a company based in {value}."],
        ("hometown", "region"): ["The hometown of {subject} lies in {value}.", "{subject} grew up in a town in {value}."],
    },
}

_SLOT = re.compile(r"\{[a-z_]+\}(?:'s)?")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


class TemplateLeakError(ValueError):
    """Training and evaluation templates share a form or a 4-gram (invariant 6)."""


def template_tokens(template: str) -> list[str]:
    """Lower-cased words with entity slots removed."""
    return _WORD.findall(_SLOT.sub(" ", template).lower())


def ngrams(tokens: list[str], n: int = FORBIDDEN_NGRAM) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def load_eval_templates(path: Path = EVAL_TEMPLATE_FILE) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def eval_templates_sha(path: Path = EVAL_TEMPLATE_FILE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def iter_train_templates(
    train: dict = TRAIN_TEMPLATES, pairs: dict = TRAIN_PAIR_TEMPLATES, derived: dict = TRAIN_DERIVED_TEMPLATES
) -> Iterator[str]:
    for attr in sorted(train):
        for group in ("statements", "reversals", "qa", "reverse_qa"):
            yield from train[attr][group]
    for key in sorted(pairs):
        yield from pairs[key]
    for attr in sorted(derived["alias"]):
        yield from derived["alias"][attr]["statements"]
        yield from derived["alias"][attr]["qa"]
    for key in sorted(derived["two_hop"]):
        yield from derived["two_hop"][key]


def iter_eval_templates(templates: dict) -> Iterator[str]:
    """Every string in the file except comments; the structure is section -> attr or chain -> list of forms."""
    for section in sorted(templates):
        if section.startswith("_"):
            continue
        for key in sorted(templates[section]):
            yield from templates[section][key]


def check_disjoint(train_forms: Iterable[str], eval_forms: Iterable[str], n: int = FORBIDDEN_NGRAM) -> None:
    """Raise TemplateLeakError on an identical form, or on any shared n-gram with slots excluded."""
    train_list, eval_list = list(train_forms), list(eval_forms)
    train_exact = {" ".join(template_tokens(t)): t for t in train_list}
    train_grams: dict[tuple[str, ...], str] = {}
    for template in train_list:
        for gram in sorted(ngrams(template_tokens(template), n)):
            train_grams.setdefault(gram, template)
    for template in eval_list:
        tokens = template_tokens(template)
        joined = " ".join(tokens)
        if joined and joined in train_exact:
            raise TemplateLeakError(f"eval template {template!r} equals training template {train_exact[joined]!r}")
        for gram in sorted(ngrams(tokens, n)):
            if gram in train_grams:
                raise TemplateLeakError(
                    f"eval template {template!r} shares {' '.join(gram)!r} with training template {train_grams[gram]!r}"
                )
