"""Fast store: provenance gate, supersession, persistence, retrieval, distractors."""
from __future__ import annotations

import pytest
import torch

from llmsleep.memory import (
    BM25Retriever,
    DistractorSampler,
    Episode,
    EpisodicStore,
    OracleRetriever,
    ProvenanceError,
    format_context,
    make_retriever,
)


def _store() -> EpisodicStore:
    store = EpisodicStore()
    store.add(Episode("a", "Ada lives in Oslo near the old harbour.", 0, entities=["Ada", "Oslo"]))
    store.add(Episode("b", "Ben works in Oslo as a baker.", 0, entities=["Ben", "Oslo"]))
    store.add(Episode("c", "Cyd paints foxes in Lima every summer.", 0, entities=["Cyd", "Lima"]))
    return store


def test_store_rejects_generated_text_on_the_fact_path():
    store = EpisodicStore()
    with pytest.raises(ProvenanceError):
        store.add(Episode("g", "A model wrote this.", 0, provenance="generated"))
    assert len(store) == 0


def test_store_rejects_duplicates():
    store = _store()
    with pytest.raises(ValueError):
        store.add(Episode("a", "again", 1))


def test_add_copies_so_store_updates_cannot_reach_the_world():
    original = Episode("x", "text", 0, meta={"qa": []})
    store = EpisodicStore()
    store.add(original)
    store.get("x").meta["last_gap"] = 1.0
    store.retire("x")
    assert "last_gap" not in original.meta and original.status == "active"


def test_retire_leaves_active_but_stays_in_archive():
    store = _store()
    store.retire("a")
    assert [e.id for e in store.active()] == ["b", "c"]
    assert [e.id for e in store.all_real()] == ["a", "b", "c"]


def test_deferred_episodes_leave_active_retrieval_but_are_still_replayed():
    store = _store()
    store.defer("c")
    assert "c" not in [e.id for e in store.active()]
    assert "c" in store and "c" in [e.id for e in store.replayable()]
    assert "c" not in [e.id for e in BM25Retriever().retrieve("who paints foxes in Lima", store, k=3)]


def test_neighbours_and_pairs_sharing_entity():
    store = _store()
    assert [e.id for e in store.neighbours("oslo")] == ["a", "b"]
    assert [(a.id, b.id) for a, b in store.pairs_sharing_entity()] == [("a", "b")]
    store.retire("a")
    assert store.pairs_sharing_entity() == []


def test_superseded_episodes_leave_pairs_and_replay():
    store = _store()
    store.add(Episode("a2", "Ada moved to Rome.", 1, entities=["Ada", "Rome"], supersedes=["a"]))
    assert "a" in store.superseded_ids()
    assert "a" not in [e.id for e in store.replayable()]
    assert all("a" not in (x.id, y.id) for x, y in store.pairs_sharing_entity())


def test_newest_follows_chains_even_through_retired_episodes():
    store = _store()
    store.add(Episode("a2", "Ada moved to Rome.", 1, entities=["Ada"], supersedes=["a"]))
    store.add(Episode("a3", "Ada moved to Kyiv.", 2, entities=["Ada"], supersedes=["a2"]))
    assert store.newest("a").id == "a3"
    store.retire("a3")
    assert store.newest("a").id == "a3"  # Returning the stale value would be worse.


def test_persist_round_trip(tmp_path):
    store = _store()
    store.add(Episode("a2", "Ada moved.", 1, entities=["Ada"], supersedes=["a"], meta={"qa": [["q", "x"]]}))
    store.retire("b")
    path = str(tmp_path / "store.json")
    store.save(path)
    loaded = EpisodicStore.load(path)
    assert [e.__dict__ for e in loaded.all_real()] == [e.__dict__ for e in store.all_real()]


def test_oracle_returns_gold_and_applies_recency():
    store = _store()
    store.add(Episode("a2", "Ada moved to Rome.", 1, entities=["Ada"], supersedes=["a"]))
    hits = OracleRetriever().retrieve("anything", store, gold_ids=["a", "c"])
    assert [e.id for e in hits] == ["a2", "c"]
    with pytest.raises(ValueError):
        OracleRetriever().retrieve("anything", store)


def test_bm25_ranks_the_matching_episode_first_and_skips_retired():
    store = _store()
    hits = BM25Retriever().retrieve("who paints foxes in Lima", store, k=1)
    assert [e.id for e in hits] == ["c"]
    store.retire("c")
    assert "c" not in [e.id for e in BM25Retriever().retrieve("who paints foxes in Lima", store, k=3)]
    assert BM25Retriever().retrieve("zzz", store, k=3) == []


def test_bm25_applies_recency_and_returns_at_most_k():
    store = _store()
    store.add(Episode("a2", "Ada moved to Rome.", 1, entities=["Ada"], supersedes=["a"]))
    hits = BM25Retriever().retrieve("Ada harbour Oslo", store, k=5)
    ids = [e.id for e in hits]
    assert "a" not in ids and "a2" in ids
    assert len(BM25Retriever().retrieve("Oslo", store, k=1)) == 1


def test_format_context_marks_the_newest_of_a_chain():
    plain = Episode("a", "Old fact.", 0)
    newer = Episode("a2", "New fact.", 1, supersedes=["a"])
    assert format_context([plain]) == "Old fact."
    assert format_context([newer, plain]) == "[Most recent] New fact.\nOld fact."


def test_make_retriever_rejects_unknown_names():
    assert isinstance(make_retriever("bm25"), BM25Retriever)
    with pytest.raises(ValueError):
        make_retriever("none")


def test_distractor_shares_no_entity_and_matches_length(generator):
    pool = [
        Episode("near", "x" * 40, 0, entities=["Zed"]),
        Episode("far", "y" * 400, 0, entities=["Yan"]),
        Episode("same", "z" * 40, 0, entities=["Ada"]),
        Episode("named", "This one mentions Ada in its text........", 0, entities=["Other"]),
    ]
    sampler = DistractorSampler(pool, length_tolerance=0.25)
    for _ in range(20):
        assert sampler.sample(["ada"], 40, generator).id == "near"


def test_distractor_falls_back_to_nearest_length_and_to_none(generator):
    pool = [Episode("p", "x" * 100, 0, entities=["P"]), Episode("q", "y" * 500, 0, entities=["Q"])]
    sampler = DistractorSampler(pool, length_tolerance=0.1)
    assert sampler.sample(["Ada"], 10, generator).id == "p"
    assert sampler.sample(["P", "Q"], 10, generator) is None
    assert sampler.sample(["Ada"], 100, generator, exclude_ids=["p"]).id == "q"


def test_distractor_is_deterministic_and_independent_of_pool_order():
    pool = [Episode(str(i), "x" * 40, 0, entities=[f"E{i}"]) for i in range(8)]
    a = DistractorSampler(pool).sample(["Ada"], 40, torch.Generator().manual_seed(5))
    b = DistractorSampler(list(reversed(pool))).sample(["Ada"], 40, torch.Generator().manual_seed(5))
    assert a.id == b.id
