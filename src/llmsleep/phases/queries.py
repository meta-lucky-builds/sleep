"""Stage 1: query synthesis, the deterministic held-out split, and what a night needs from a world.

Queries are inputs only. A generated string becomes a `Candidate`, never an `Episode`, and every training
target comes from a teacher given a real episode (invariant 3).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import AbstractSet, Iterable, Optional, Protocol, Sequence, runtime_checkable

import torch

from ..config import QueryConfig
from ..handle import ModelHandle
from ..ledger import LedgerExhausted
from ..memory import Episode, EpisodicStore, format_context


@runtime_checkable
class WorldLike(Protocol):
    """What a night needs from a world. Must be deterministic given its seed.

    Episodes must have provenance "real". Put training forms in `Episode.meta` (see memory.Episode).
    Templates returned here are training templates: disjointness from evaluation templates (invariant 6)
    is the world's duty, enforced at its construction.
    """

    def episodes(self, night: int) -> Sequence[Episode]:
        """The episodes observed on the day before `night` (0-based)."""
        ...

    def template_queries(self, episode: Episode) -> Sequence[str]:
        """Oracle queries about one episode."""
        ...

    def pair_template_queries(self, a: Episode, b: Episode) -> Sequence[str]:
        """Oracle queries that need both episodes, such as two-hop."""
        ...

    def distractor_pool(self) -> Sequence[Episode]:
        """Unrelated filler episodes, never stored or taught, so that night 0 has distractors at all."""
        ...


@dataclass
class Candidate:
    """A query with its source. `context` and `distractor` are filled in once retrieval has run."""

    query: str
    episode_ids: tuple[str, ...]
    entities: tuple[str, ...]
    is_pair: bool = False
    context: str = ""
    distractor: Optional[str] = None
    context_ids: tuple[str, ...] = ()  # Episodes in `context`; under real retrieval, not always the source's.
    pool: bool = False  # From the anchor pool: names nothing of tonight's episodes.


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def dedupe(queries: Iterable[str]) -> list[str]:
    """Keep first occurrences, comparing normalised text."""
    seen: dict[str, str] = {}
    for query in queries:
        seen.setdefault(normalize_query(query), query.strip())
    return list(seen.values())


_LEADING_MARK = re.compile(r"^\s*(?:\d+\s*[.):]|[-*•])\s*")


def parse_questions(text: str) -> list[str]:
    """Questions from generated text: one per line, list markers removed, at least two words, ending in '?'."""
    found = []
    for line in text.splitlines():
        line = _LEADING_MARK.sub("", line).strip()
        if line.endswith("?") and len(line.split()) >= 2:
            found.append(line)
    return dedupe(found)


class QueryGen(Protocol):
    def generate(self, episodes: Sequence[Episode], generator: torch.Generator) -> list[str]: ...

    def usable_pair(self, a: Episode, b: Episode) -> bool: ...


class TemplateQueryGen:
    """Oracle queries from the world. An upper bound on query quality; benchmark only."""

    def __init__(self, world: WorldLike, per_episode: int) -> None:
        self.world = world
        self.per_episode = per_episode

    def generate(self, episodes: Sequence[Episode], generator: torch.Generator) -> list[str]:
        if len(episodes) == 1:
            queries = self.world.template_queries(episodes[0])
        elif len(episodes) == 2:
            queries = self.world.pair_template_queries(episodes[0], episodes[1])
        else:
            raise ValueError("templates exist for one or two episodes")
        return dedupe(queries)[: self.per_episode]

    def usable_pair(self, a: Episode, b: Episode) -> bool:
        """Oracle templates exist only for true chains; any other pair would waste a slot under the cap."""
        return bool(self.world.pair_template_queries(a, b))


class ModelQueryGen:
    """The deployable path: the model, given the episodes, writes questions. Charged to the ledger."""

    def __init__(self, handle: ModelHandle, cfg: QueryConfig) -> None:
        self.handle = handle
        self.cfg = cfg

    def generate(self, episodes: Sequence[Episode], generator: torch.Generator) -> list[str]:
        context = "\n".join(e.text for e in episodes)
        prompt = self.handle.format(self.cfg.instruction, context)
        samples = self.handle.sample(
            [prompt], self.cfg.model_samples, self.cfg.model_max_new_tokens, self.cfg.temperature, generator
        )[0]
        queries: list[str] = []
        for sample in samples:
            queries.extend(parse_questions(self.handle.decode(sample.token_ids)))
        return dedupe(queries)[: self.cfg.per_episode]

    def usable_pair(self, a: Episode, b: Episode) -> bool:
        return True


def is_chain(a: Episode, b: Episode) -> bool:
    """One episode's value is the other's subject: entities list the subject first, then any named value."""
    first, second = [x.casefold() for x in a.entities], [x.casefold() for x in b.entities]
    if not first or not second:
        return False
    return second[0] in first[1:] or first[0] in second[1:]


def choose_pairs(
    pairs: Sequence[tuple[Episode, Episode]], max_pairs: int, generator: torch.Generator
) -> list[tuple[Episode, Episode]]:
    """Chains first, then other pairs across nights, then same-night ones, uniformly at random within each group.

    Two-hop probes need the chains, and most pairs are not chains. Cutting a sorted list would instead drop pairs by
    the accident of their ids, which would bias H7.
    """

    def shuffled(group: list[tuple[Episode, Episode]]) -> list[tuple[Episode, Episode]]:
        return [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]

    chains = [(a, b) for a, b in pairs if is_chain(a, b)]
    cross = [(a, b) for a, b in pairs if not is_chain(a, b) and a.night != b.night]
    same = [(a, b) for a, b in pairs if not is_chain(a, b) and a.night == b.night]
    return (shuffled(chains) + shuffled(cross) + shuffled(same))[:max_pairs]


def pair_stats(pairs: Sequence[tuple[Episode, Episode]], chosen: Sequence[tuple[Episode, Episode]]) -> dict[str, int]:
    """What the cap kept, including how many chains it covered tonight."""
    return {
        "available": len(pairs),
        "chains_available": sum(1 for a, b in pairs if is_chain(a, b)),
        "cross_night_available": sum(1 for a, b in pairs if a.night != b.night),
        "chosen": len(chosen),
        "chains_chosen": sum(1 for a, b in chosen if is_chain(a, b)),
        "cross_night_chosen": sum(1 for a, b in chosen if a.night != b.night),
    }


class PairQueryGen:
    """Queries over two episodes sharing an entity, since composition needs co-occurrence (T2 finding 4)."""

    def __init__(self, base: QueryGen, max_pairs: int) -> None:
        self.base = base
        self.max_pairs = max_pairs
        self.stats: dict[str, int] = {}  # Of the last call: pairs available and chosen, overall and cross-night.

    def generate(
        self, store: EpisodicStore, new_ids: AbstractSet[str], generator: torch.Generator
    ) -> tuple[list[Candidate], bool]:
        """Pairs with at least one new episode. Returns (candidates so far, exhausted)."""
        pairs = [(a, b) for a, b in store.pairs_sharing_entity() if a.id in new_ids or b.id in new_ids]
        pairs = [(a, b) for a, b in pairs if self.base.usable_pair(a, b)]  # Before the cap, so no slot is wasted.
        chosen = choose_pairs(pairs, self.max_pairs, generator)
        self.stats = pair_stats(pairs, chosen)
        out: list[Candidate] = []
        for a, b in chosen:
            try:
                queries = self.base.generate([a, b], generator)
            except LedgerExhausted:
                return out, True
            entities = tuple(dict.fromkeys(list(a.entities) + list(b.entities)))
            context = format_context([a, b])  # Until retrieval runs; the pair null uses its length.
            out.extend(Candidate(q, (a.id, b.id), entities, is_pair=True, context=context) for q in queries)
        return out, False


def generate_singles(
    gen: QueryGen, episodes: Sequence[Episode], generator: torch.Generator
) -> tuple[list[Candidate], bool]:
    """Candidates per episode. Returns (candidates so far, exhausted)."""
    out: list[Candidate] = []
    for episode in episodes:
        try:
            queries = gen.generate([episode], generator)
        except LedgerExhausted:
            return out, True
        # The episode's own text is its context until retrieval runs (the store may not hold it yet).
        context = format_context([episode])
        out.extend(Candidate(q, (episode.id,), tuple(episode.entities), context=context) for q in queries)
    return out, False


def build_anchor_pool(
    handle: ModelHandle,
    store: EpisodicStore,
    world: WorldLike,
    new_ids: AbstractSet[str],
    cfg: QueryConfig,
    generator: torch.Generator,
) -> tuple[list[Candidate], bool]:
    """Queries that name no entity of tonight's episodes (A1.5). Returns (a uniform sample of them, exhausted).

    Generated queries are about tonight's episodes, so anchors come from elsewhere: queries kept from earlier nights
    (`meta["queries"]`), plus fresh model-generated ones over a seeded sample of older episodes and the distractor
    pool. Fresh ones are always model-generated, since worlds have no templates for filler text.
    """
    tonight = {x.casefold() for episode_id in new_ids for x in store.get(episode_id).entities}
    older = [e for e in store.replayable() if e.id not in new_ids]
    found: dict[str, Candidate] = {}

    def add(query: str, episode: Episode) -> None:
        key = normalize_query(query)
        lowered = query.casefold()
        if key in found or any(name in lowered for name in tonight):
            return
        if {x.casefold() for x in episode.entities} & tonight:
            return
        found[key] = Candidate(
            query,
            (episode.id,),
            tuple(episode.entities),
            context=format_context([episode]),
            context_ids=(episode.id,),
            pool=True,
        )

    for episode in older:
        for query in episode.meta.get("queries", []):
            add(query, episode)

    def sample(group: Sequence[Episode], n: int) -> list[Episode]:
        return [group[i] for i in torch.randperm(len(group), generator=generator).tolist()[:n]]

    sources = sample(older, cfg.anchor_older_episodes) + sample(
        sorted(world.distractor_pool(), key=lambda e: e.id), cfg.anchor_distractor_episodes
    )
    gen = ModelQueryGen(handle, cfg)
    exhausted = False
    for episode in sources:
        try:
            queries = gen.generate([episode], generator)
        except LedgerExhausted:
            exhausted = True
            break
        for query in queries:
            add(query, episode)
    return sample(list(found.values()), cfg.anchor_pool_max), exhausted


def make_query_gen(source: str, world: WorldLike, handle: ModelHandle, cfg: QueryConfig) -> QueryGen:
    if source == "template":
        return TemplateQueryGen(world, cfg.per_episode)
    if source == "model":
        return ModelQueryGen(handle, cfg)
    raise ValueError(f"no query generator for source {source!r}")


def _heldout_score(query: str, world_seed: int) -> float:
    """A uniform value in [0, 1) fixed by the query's text and the world seed alone."""
    digest = hashlib.sha256(f"{world_seed}:heldout:{query}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_heldout(
    candidates: Sequence[Candidate], world_seed: int, frac: float, known_heldout: AbstractSet[str] = frozenset()
) -> tuple[list[Candidate], list[Candidate], set[str]]:
    """Deterministic audit split: (train, heldout, normalised held-out queries).

    Each query's side depends only on its normalised text, so order and later additions cannot move it, and
    the same text under two contexts cannot land on both sides. Anything in `known_heldout` stays held out.
    If the hash leaves no held-out query among two or more distinct ones, the lowest-scoring one is promoted,
    so the audit is not starved on tiny candidate sets.
    """
    unique = sorted({normalize_query(c.query) for c in candidates})
    held = {q for q in unique if q in known_heldout or _heldout_score(q, world_seed) < frac}
    if not held and len(unique) >= 2 and frac > 0:
        held = {min(unique, key=lambda q: _heldout_score(q, world_seed))}
    train = [c for c in candidates if normalize_query(c.query) not in held]
    heldout = [c for c in candidates if normalize_query(c.query) in held]
    return train, heldout, held
