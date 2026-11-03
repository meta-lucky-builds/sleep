"""E0 utility: does a model already know the fictional entities? Reported; it does not gate (PREREG A1.10).

Asks the world's direct probes closed-book, on the base model, and tests each attribute's hit count against
chance with an exact binomial test. Entities in a rejected attribute that were actually hit are listed for
replacement: add their name words to `WorldConfig.banned_tokens` and the forge will not issue them.
"""
from __future__ import annotations

from typing import Any, Optional

from ..config import make_generator
from ..eval.answers import score_answer
from ..eval.metrics import generate_answers
from ..eval.probes import stratified_sample
from ..stats import binomial_tail

REJECT_P = 0.01


def check_unknown(
    handle: Any,
    world: Any,
    max_probes: Optional[int] = None,
    max_new_tokens: int = 24,
    answer_mode: bool = False,
) -> dict:
    """Per-attribute closed-book hit rates and binomial p-values. Only `rejected` attributes need action."""
    probes = stratified_sample(
        [p for p in world.probes if p.kind == "direct"], max_probes, world.seed, "familiarity"
    )
    with handle.ledger.exempt(), handle.banks("germline"):
        answers = generate_answers(
            handle,
            [(p.query, None) for p in probes],
            max_new_tokens,
            0.0,
            make_generator(world.seed, 0, "familiarity"),
            answer_mode,
        )
    counts: dict[str, list[int]] = {}
    hits_by_attr: dict[str, list[str]] = {}
    for probe in probes:
        text = answers[(probe.query, None)][0]
        correct = score_answer(text, probe.gold, (), world.vocabulary(probe.vocab), probe.query).correct
        entry = counts.setdefault(probe.vocab, [0, 0])
        entry[0] += int(correct)
        entry[1] += 1
        if correct:
            hits_by_attr.setdefault(probe.vocab, []).append(probe.cluster)
    attributes: dict[str, dict] = {}
    rejected: list[str] = []
    replace: set[str] = set()
    for attr in sorted(counts):
        hits, total = counts[attr]
        chance = 1.0 / max(1, len(world.answer_groups(attr)))
        p_value = binomial_tail(hits, total, chance)
        attributes[attr] = {"hits": hits, "n": total, "rate": hits / total, "chance": chance, "p": p_value}
        if p_value < REJECT_P:
            rejected.append(attr)
            replace.update(world.entities[key.split(".")[0]].name for key in hits_by_attr.get(attr, []))
    return {"attributes": attributes, "rejected": rejected, "replace": sorted(replace), "reject_p": REJECT_P}
