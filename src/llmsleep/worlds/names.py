"""Seeded fictional names, collision-checked, and near-name look-alikes.

All draws go through `SeededRng`, built only on `random.Random.random()`, whose sequence the Python docs
promise stays the same across versions for a given seed. Nothing iterates over a set unsorted.
"""
from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, TypeVar

T = TypeVar("T")

_ONSETS = "B Br C Cr D Dr F Fl G Gr H J K Kl L M N P Pr Qu R S Sk St T Tr V W Y Z".split()
_VOWELS = "a e i o u".split()
_MIDDLES = "b d f g k l lb lm ln m n nd ndr nt r rd rl rm rn rv s sk st t th v z".split()
_ENDS = "n r s l m t nt rd rn ld lt sk x th ck ne ra la na ry ley ton wick ford mont vane dale".split()

# A token that is a real place, brand or common word would be known to a model. E0 checks the rest empirically.
BLOCKLIST = frozenset(
    """
    paris london berlin rome madrid tokyo boston denver dallas austin chicago houston seattle moscow vienna
    dublin oslo cairo lima delhi google amazon apple meta intel oracle tesla nokia sony boeing ford
    martin taylor miller wilson walter parker carter turner hunter porter baker tanner sanders mason
    gordon morgan dalton milton norton burton bradford stanford clinton preston
    maria anna david daniel sarah linda peter simon thomas
    salad talent silent parent garden golden common person dragon button cotton bottom
    """.split()
)


class SeededRng:
    """Deterministic draws from one seed. Uses only `random()`, for stability across Python versions."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(int(seed))

    def random(self) -> float:
        return self._random.random()

    def below(self, n: int) -> int:
        if n < 1:
            raise ValueError("below needs n >= 1")
        return min(int(self._random.random() * n), n - 1)

    def choice(self, items: Sequence[T]) -> T:
        return items[self.below(len(items))]

    def shuffled(self, items: Iterable[T]) -> list[T]:
        """Fisher-Yates on a copy."""
        out = list(items)
        for i in range(len(out) - 1, 0, -1):
            j = self.below(i + 1)
            out[i], out[j] = out[j], out[i]
        return out

    def sample(self, items: Sequence[T], k: int) -> list[T]:
        return self.shuffled(items)[: k]


def edit_distance(a: str, b: str, limit: Optional[int] = None) -> int:
    """Levenshtein distance. With `limit`, returns limit + 1 as soon as the distance must exceed it."""
    if a == b:
        return 0
    if limit is not None and abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        if limit is not None and min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


class NameForge:
    """Issues single-word tokens, each at edit distance at least `min_distance` from every token issued.

    So a near-name of an issued token (distance 1 or 2) can never be another issued token.
    """

    def __init__(self, rng: SeededRng, min_distance: int = 3, banned: Iterable[str] = ()) -> None:
        self.rng = rng
        self.min_distance = min_distance
        self.issued: list[str] = []  # Casefolded, in issue order.
        # Issued, reserved look-alikes, and words E0 found a model knows: never handed out.
        self._all: dict[str, None] = {word.casefold(): None for word in banned}

    def _candidate(self, short: bool) -> str:
        r = self.rng
        if short:
            return (r.choice(_ONSETS) + r.choice(_VOWELS) + r.choice(_MIDDLES) + r.choice(_VOWELS)).capitalize()
        return (
            r.choice(_ONSETS) + r.choice(_VOWELS) + r.choice(_MIDDLES) + r.choice(_VOWELS) + r.choice(_ENDS)
        ).capitalize()

    def _acceptable(self, word: str) -> bool:
        key = word.casefold()
        if key in BLOCKLIST or key in self._all:
            return False
        limit = self.min_distance - 1
        return all(edit_distance(key, other, limit) > limit for other in self.issued)

    def token(self, short: bool = False, attempts: int = 20_000) -> str:
        for _ in range(attempts):
            word = self._candidate(short)
            if self._acceptable(word):
                key = word.casefold()
                self.issued.append(key)
                self._all[key] = None
                return word
        raise RuntimeError(f"could not find a new name after {attempts} attempts; lower min_distance or the world size")

    def near(self, word: str, attempts: int = 500) -> str:
        """A look-alike of `word`: one letter changed, never an issued name. Reserved, so it is never issued."""
        vowels = "aeiou"
        consonants = "bdfgklmnprstvz"
        for _ in range(attempts):
            positions = self.rng.shuffled(range(1, len(word)))  # Keep the first letter: it reads as a near-name.
            for position in positions:
                char = word[position].lower()
                pool = vowels if char in vowels else consonants
                replacement = self.rng.choice(pool)
                if replacement == char:
                    continue
                candidate = word[:position] + replacement + word[position + 1 :]
                key = candidate.casefold()
                if key not in self._all and key not in BLOCKLIST:
                    self._all[key] = None
                    return candidate
        raise RuntimeError(f"no look-alike found for {word!r}")
