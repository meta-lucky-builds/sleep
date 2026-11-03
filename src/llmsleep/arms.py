"""Experimental arms (PREREG). This is the only place arms differ (invariant 5): phases read flags."""
from __future__ import annotations

from dataclasses import dataclass, replace

_QUERY_SOURCES = ("none", "model", "template")
_HARD_POOLS = ("none", "new", "all_real", "selected")
_ANCHORS = ("none", "germline", "chain")
_RETRIEVERS = ("none", "oracle", "bm25")


@dataclass(frozen=True)
class ArmConfig:
    """Flags for one arm; every default is "off".

    S_nokl ablates the teacher only. Its "selection" is the CHD write gate, plus a sampling boost in the
    hard pool for episodes that have integrate queries. Its hard labels come from real episode statements
    and QA forms, never from a teacher's answers, so its integrate queries are not trained on directly.

    S_noselect ablates CHD only (A1.4). With `selection` off, teacher KL covers all usable generated queries, and
    anchors are a uniform draw from the anchor pool, at S_full's count when `SelectConfig.matched_selection`
    supplies it. The tokens S_full would spend on signals go to its training stage.

    Anchors, in every arm that has them (A1.5), come from a pool naming nothing of tonight's episodes: queries
    generated on earlier nights, plus fresh model-generated queries over older episodes and the distractor pool.
    """

    name: str
    train: bool = False  # False: weights are never touched and no budget is spent (B0, B1).
    open_book_eval: bool = False  # Read by eval: answer with retrieved context (B1).
    retriever: str = "none"  # Context source for teachers, and for open-book eval.
    query_source: str = "none"  # model: generated. template: world's oracle templates.
    pair_contexts: bool = False  # Also synthesise queries over pairs of episodes sharing an entity.
    replay_pairs: bool = False  # Hard labels also on concatenated real pairs, under the same pair cap (A1.1).
    selection: bool = False  # CHD partition into integrate/anchor/discard. Off: all integrate, anchors drawn.
    write_gate: bool = False  # Do not store episodes the model already knows, by CHD.
    teacher_kl: bool = False  # KL to the open-book teacher on integrate queries.
    anchors: str = "none"  # Target when an anchor query touches nothing consolidated.
    self_sampled_kl: bool = False  # KL to germline on text sampled from germline (Marek, A14).
    hard_pool: str = "none"  # Which real episodes feed the hard-label CE term.
    compact: bool = False  # SVT plus repair. Off: exact merge only.
    audit: bool = False  # Measure held-out gaps; drives pressure and retirement.
    transactional: bool = False  # Audit may reject the night and roll back.

    def __post_init__(self) -> None:
        if self.query_source not in _QUERY_SOURCES:
            raise ValueError(f"query_source: {self.query_source!r}")
        if self.hard_pool not in _HARD_POOLS:
            raise ValueError(f"hard_pool: {self.hard_pool!r}")
        if self.anchors not in _ANCHORS:
            raise ValueError(f"anchors: {self.anchors!r}")
        if self.retriever not in _RETRIEVERS:
            raise ValueError(f"retriever: {self.retriever!r}")
        if self.query_source != "none" and self.retriever == "none":
            raise ValueError("synthesised queries need a retriever to supply teacher context")
        if self.transactional and not self.audit:
            raise ValueError("transactional commit needs the audit")
        if self.audit and self.query_source == "none":
            raise ValueError("the audit needs synthesised held-out queries")
        if (self.selection or self.write_gate or self.teacher_kl) and self.query_source == "none":
            raise ValueError("selection, write gate and teacher need synthesised queries")
        if self.replay_pairs and self.hard_pool == "none":
            raise ValueError("pair replay needs a hard pool")
        if self.anchors != "none" and self.query_source == "none":
            raise ValueError("anchors need a query generator for their pool")
        if self.train and self.hard_pool == "none" and not self.teacher_kl:
            raise ValueError("a training arm needs a hard pool or a teacher")


_S_FULL = ArmConfig(
    name="S_full",
    train=True,
    retriever="oracle",
    query_source="model",
    pair_contexts=True,
    selection=True,
    write_gate=True,
    teacher_kl=True,
    anchors="germline",
    hard_pool="selected",
    compact=True,
    audit=True,
    transactional=True,
)

_B3 = ArmConfig(name="B3_replay", train=True, hard_pool="all_real")

ARMS: dict[str, ArmConfig] = {
    "B0_frozen": ArmConfig(name="B0_frozen"),
    "B1_rag": ArmConfig(name="B1_rag", open_book_eval=True, retriever="oracle"),
    "B2_seq": ArmConfig(name="B2_seq", train=True, hard_pool="new"),
    "B3_replay": _B3,
    "B3_pairs": replace(_B3, name="B3_pairs", replay_pairs=True),
    "B4_replay_kl": replace(_B3, name="B4_replay_kl", self_sampled_kl=True),
    "S_full": _S_FULL,
    "S_oracleq": replace(_S_FULL, name="S_oracleq", query_source="template"),
    "S_noselect": replace(_S_FULL, name="S_noselect", selection=False, write_gate=False),
    "S_nokl": replace(_S_FULL, name="S_nokl", teacher_kl=False),
    "S_chain": replace(_S_FULL, name="S_chain", anchors="chain"),
    "S_nocompact": replace(_S_FULL, name="S_nocompact", compact=False),
    "S_nopair": replace(_S_FULL, name="S_nopair", pair_contexts=False),
    # Gaps are still measured, so pressure and retirement match S_full; only the rollback is ablated.
    "S_nocommit": replace(_S_FULL, name="S_nocommit", transactional=False),
    "S_bm25": replace(_S_FULL, name="S_bm25", retriever="bm25"),
}


def get_arm(name: str) -> ArmConfig:
    """Look up an arm by its PREREG name."""
    try:
        return ARMS[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}") from None
