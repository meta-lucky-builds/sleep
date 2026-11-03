"""E0 utility: does a model already know the fictional entities?

Asks the world's direct probes closed-book, on the base model. A hit on a fictional value (an organisation, a town,
a device) cannot come from general knowledge, so any noticeable rate means the names collide with something real.
Values drawn from common words (instruments, years) can be hit by chance, so they are compared with chance.
"""
from __future__ import annotations

from typing import Any, Optional

from ..config import make_generator
from ..eval.answers import score_answer
from ..eval.metrics import generate_answers
from ..eval.probes import Probe

_CLOSED_VOCAB_ATTRS = ("instrument", "birth_year", "product")


def check_unknown(
    handle: Any,
    world: Any,
    max_probes: Optional[int] = None,
    max_new_tokens: int = 24,
    fictional_limit: float = 0.02,
    chance_multiple: float = 3.0,
) -> dict:
    """Per-attribute closed-book hit rates and a pass/fail. `flagged` lists the entities actually hit."""
    probes: list[Probe] = [p for p in world.probes if p.kind == "direct"]
    if max_probes is not None:
        probes = probes[:max_probes]
    with handle.ledger.exempt(), handle.banks("germline"):
        answers = generate_answers(
            handle, [(p.query, None) for p in probes], max_new_tokens, 0.0, make_generator(world.seed, 0, "familiarity")
        )
    hits: dict[str, list[int]] = {}
    flagged: list[str] = []
    for probe in probes:
        correct = score_answer(answers[(probe.query, None)][0], probe.gold, (), world.vocabulary(probe.vocab)).correct
        entry = hits.setdefault(probe.vocab, [0, 0])
        entry[0] += int(correct)
        entry[1] += 1
        if correct and probe.vocab not in _CLOSED_VOCAB_ATTRS:
            flagged.append(probe.cluster)
    rates, passed = {}, True
    for attr in sorted(hits):
        count, total = hits[attr]
        rate = count / total
        if attr in _CLOSED_VOCAB_ATTRS:
            limit = chance_multiple / max(1, len(world.vocabulary(attr)))
        else:
            limit = fictional_limit
        ok = rate <= limit
        passed = passed and ok
        rates[attr] = {"rate": rate, "n": total, "limit": limit, "ok": ok}
    return {"passed": passed, "attributes": rates, "flagged": sorted(flagged)}
