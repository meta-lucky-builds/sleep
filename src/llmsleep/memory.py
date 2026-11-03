"""Fast episodic store, retrievers and the distractor sampler. Only real episodes are ever facts."""
from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Iterable, Optional, Protocol, Sequence

import torch

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class ProvenanceError(ValueError):
    """Something other than a real episode was offered to the fact path (invariant 3)."""


@dataclass
class Episode:
    """One experience. World-supplied training data goes in `meta`:

    meta["statements"]: list[str], extra text forms (paraphrases, explicit reversals).
    meta["qa"]: list of [query, answer] pairs, in training templates.
    """

    id: str
    text: str
    night: int
    provenance: str = "real"
    entities: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = "active"


class EpisodicStore:
    """Insertion-ordered episodes. Retired ones stay in the archive; nothing is ever deleted."""

    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}

    def add(self, episode: Episode) -> Episode:
        """Store a copy, so that later meta updates cannot leak back into a world's own objects."""
        if episode.provenance != "real":
            raise ProvenanceError(f"episode {episode.id!r} has provenance {episode.provenance!r}")
        if episode.id in self._episodes:
            raise ValueError(f"duplicate episode id {episode.id!r}")
        stored = copy.deepcopy(episode)
        self._episodes[episode.id] = stored
        return stored

    def get(self, episode_id: str) -> Episode:
        return self._episodes[episode_id]

    def __contains__(self, episode_id: str) -> bool:
        return episode_id in self._episodes

    def __len__(self) -> int:
        return len(self._episodes)

    def retire(self, episode_id: str) -> None:
        """Leave active retrieval; remain in the archive."""
        self._episodes[episode_id].status = "retired"

    def active(self) -> list[Episode]:
        return [e for e in self._episodes.values() if e.status == "active"]

    def all_real(self) -> list[Episode]:
        return list(self._episodes.values())

    def superseded_ids(self) -> set[str]:
        return {old for e in self._episodes.values() for old in e.supersedes}

    def replayable(self) -> list[Episode]:
        """Real episodes still true: replaying a superseded one would train the stale value."""
        superseded = self.superseded_ids()
        return [e for e in self._episodes.values() if e.id not in superseded]

    def newest(self, episode_id: str) -> Episode:
        """Follow the supersession chain to its end, whatever the end's status."""
        newer: dict[str, str] = {}
        for episode in self._episodes.values():  # Insertion order: a later superseder wins.
            for old in episode.supersedes:
                newer[old] = episode.id
        current, seen = episode_id, {episode_id}
        while current in newer and newer[current] not in seen:
            current = newer[current]
            seen.add(current)
        return self._episodes[current]

    def neighbours(self, entity: str) -> list[Episode]:
        key = entity.casefold()
        return [e for e in self._episodes.values() if key in {x.casefold() for x in e.entities}]

    def pairs_sharing_entity(self) -> list[tuple[Episode, Episode]]:
        """Pairs of active, current episodes sharing an entity, each once, in a fixed order."""
        superseded = self.superseded_ids()
        current = sorted((e for e in self.active() if e.id not in superseded), key=lambda e: e.id)
        pairs = []
        for a, b in combinations(current, 2):
            if {x.casefold() for x in a.entities} & {x.casefold() for x in b.entities}:
                pairs.append((a, b))
        return pairs

    def save(self, path: str) -> None:
        """Atomic JSON write."""
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump([asdict(e) for e in self._episodes.values()], handle, ensure_ascii=False, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "EpisodicStore":
        store = cls()
        with open(path, "r", encoding="utf-8") as handle:
            for record in json.load(handle):
                store._episodes[record["id"]] = Episode(**record)
        return store


def format_context(episodes: Iterable[Episode]) -> str:
    """Join episodes into a context; the newest of a supersession chain is marked as such."""
    return "\n".join(f"[Most recent] {e.text}" if e.supersedes else e.text for e in episodes)


def _unique_newest(store: EpisodicStore, ids: Iterable[str]) -> list[Episode]:
    """Recency rule: replace each hit by its chain's newest, keeping first-seen order."""
    out: dict[str, Episode] = {}
    for episode_id in ids:
        newest = store.newest(episode_id)
        out.setdefault(newest.id, newest)
    return list(out.values())


class Retriever(Protocol):
    def retrieve(
        self, query: str, store: EpisodicStore, k: int = 2, gold_ids: Optional[Sequence[str]] = None
    ) -> list[Episode]: ...


class OracleRetriever:
    """Returns the gold episodes. Benchmark only."""

    def retrieve(
        self, query: str, store: EpisodicStore, k: int = 2, gold_ids: Optional[Sequence[str]] = None
    ) -> list[Episode]:
        if gold_ids is None:
            raise ValueError("oracle retrieval needs gold_ids")
        return _unique_newest(store, gold_ids)


class BM25Retriever:
    """Okapi BM25 over the active store. Pure Python; rebuilt per call, since stores are small."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def retrieve(
        self, query: str, store: EpisodicStore, k: int = 2, gold_ids: Optional[Sequence[str]] = None
    ) -> list[Episode]:
        corpus = store.active()
        if not corpus:
            return []
        docs = [Counter(tokenize(e.text)) for e in corpus]
        lengths = [sum(d.values()) for d in docs]
        average = sum(lengths) / len(lengths) or 1.0
        terms = set(tokenize(query))
        scored = []
        for episode, doc, length in zip(corpus, docs, lengths):
            score = 0.0
            for term in terms:
                frequency = doc.get(term, 0)
                if frequency == 0:
                    continue
                containing = sum(1 for d in docs if term in d)
                idf = math.log(1.0 + (len(docs) - containing + 0.5) / (containing + 0.5))
                score += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * (1 - self.b + self.b * length / average))
            if score > 0:
                scored.append((score, episode.id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return _unique_newest(store, [episode_id for _, episode_id in scored])[:k]


def make_retriever(name: str) -> Retriever:
    if name == "oracle":
        return OracleRetriever()
    if name == "bm25":
        return BM25Retriever()
    raise ValueError(f"no retriever named {name!r}")


class DistractorSampler:
    """Draws a context sharing no entity with the query's, matching length to within a relative tolerance."""

    def __init__(self, pool: Iterable[Episode], length_tolerance: float = 0.5) -> None:
        unique = {e.id: e for e in pool}
        self.pool = [unique[i] for i in sorted(unique)]  # Order fixed, whatever order the caller used.
        self.length_tolerance = length_tolerance

    def sample(
        self,
        entities: Iterable[str],
        target_length: int,
        generator: torch.Generator,
        exclude_ids: Iterable[str] = (),
    ) -> Optional[Episode]:
        """None if nothing is entity-disjoint. If nothing is near in length, the nearest lengths are used."""
        keys = [x.casefold() for x in entities]
        excluded = set(exclude_ids)
        eligible = [
            e
            for e in self.pool
            if e.id not in excluded
            and not ({x.casefold() for x in e.entities} & set(keys))
            and not any(key in e.text.casefold() for key in keys)
        ]
        if not eligible:
            return None
        tolerance = self.length_tolerance * target_length
        near = [e for e in eligible if abs(len(e.text) - target_length) <= tolerance]
        if not near:
            closest = min(abs(len(e.text) - target_length) for e in eligible)
            near = [e for e in eligible if abs(len(e.text) - target_length) == closest]
        return near[int(torch.randint(len(near), (1,), generator=generator))]
