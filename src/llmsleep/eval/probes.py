"""Probe sets, frozen when a world is created. A probe is active over a window of nights.

After night n, the facts of `world.episodes(0..n)` have been taught, so a probe with valid_from <= n is fair.
A value superseded on night m ends its probes at m and starts new ones, so `acq_direct` never mixes
acquisition with supersession: an updated fact moves to the `supersede` kind, with the old value as `stale`.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from ..config import derive_seed

if TYPE_CHECKING:
    from ..worlds.synthetic import SyntheticWorld

KINDS = (
    "direct",
    "paraphrase",
    "reverse",
    "alias",
    "two_hop",
    "supersede",
    "preservation",
    "false_recall",
    "entrench_fact",
    "entrench_control",
)
# Attributes with a look-alike probe, by look-alike kind.
LOOKALIKE_ATTRS = {
    "person": ("employer", "hometown", "instrument", "birth_year", "invention"),
    "org": ("hq_city", "product"),
}
CHAINS = (("employer", "hq_city"), ("hometown", "region"))


@dataclass(frozen=True)
class Probe:
    """One frozen question. `cluster` is the fact it belongs to, for the fact-clustered bootstrap."""

    id: str
    kind: str
    query: str
    gold: tuple[str, ...]
    vocab: str
    cluster: str
    valid_from: int
    valid_to: Optional[int] = None  # Exclusive.
    stale: tuple[str, ...] = ()
    episode_ids: tuple[str, ...] = ()  # Gold episodes, for oracle retrieval.
    context: Optional[str] = None  # A fixed context; only the entrenchment probes have one.
    neighbour: tuple[str, ...] = ()  # False recall: the value of the taught entity this one resembles.
    same_night: Optional[bool] = None  # Two-hop: whether both facts arrived on one night.
    ripple: bool = False  # Two-hop built on a superseded first link.

    def active(self, night: int) -> bool:
        return self.valid_from <= night and (self.valid_to is None or night < self.valid_to)

    def to_dict(self) -> dict:
        return asdict(self)


def select_probes(
    probes: Sequence[Probe], kind: str, night: int, max_n: Optional[int], world_seed: int
) -> list[Probe]:
    """Active probes of one kind, capped by a per-probe hash, so every arm gets the same subset on a night."""
    active = [p for p in probes if p.kind == kind and p.active(night)]
    if max_n is None or len(active) <= max_n:
        return active
    ranked = sorted(active, key=lambda p: (_rank(world_seed, p.id), p.id))
    chosen = {p.id for p in ranked[:max_n]}
    return [p for p in active if p.id in chosen]


def _rank(world_seed: int, probe_id: str) -> bytes:
    return hashlib.sha256(f"{world_seed}:probe:{probe_id}".encode("utf-8")).digest()


def build_probes(world: "SyntheticWorld", templates: dict) -> tuple[Probe, ...]:
    """All probes of a world, in a fixed order. Pure in the world and the template file."""
    out: list[Probe] = []
    out += _direct_family(world, templates)
    out += _reverse(world, templates)
    out += _alias(world, templates)
    out += _two_hop(world, templates)
    out += _preservation(world, templates)
    out += _false_recall(world, templates)
    out += _entrenchment(world, templates)
    ids = [p.id for p in out]
    if len(ids) != len(set(ids)):
        raise AssertionError("probe ids are not unique")
    return tuple(out)


def _direct_family(world: "SyntheticWorld", templates: dict) -> list[Probe]:
    out = []
    for key in world.fact_keys:
        versions = world.versions(key)
        for index, fact in enumerate(versions):
            until = versions[index + 1].night if index + 1 < len(versions) else None
            stale = tuple(form for old in versions[:index] for form in world.answer_forms(old))
            subject = world.entities[fact.subject].name
            kind = "direct" if index == 0 else "supersede"
            for template in templates["direct"][fact.attr]:
                out.append(
                    Probe(
                        id=f"{kind}:{fact.id}",
                        kind=kind,
                        query=template.format(subject=subject),
                        gold=world.answer_forms(fact),
                        stale=stale,
                        vocab=fact.attr,
                        cluster=fact.key,
                        valid_from=fact.night,
                        valid_to=until,
                        episode_ids=(fact.episode_id,),
                    )
                )
            if index == 0:
                for template in templates["paraphrase"].get(fact.attr, []):
                    out.append(
                        Probe(
                            id=f"paraphrase:{fact.id}",
                            kind="paraphrase",
                            query=template.format(subject=subject),
                            gold=world.answer_forms(fact),
                            vocab=fact.attr,
                            cluster=fact.key,
                            valid_from=fact.night,
                            valid_to=until,
                            episode_ids=(fact.episode_id,),
                        )
                    )
    return out


def _reverse(world: "SyntheticWorld", templates: dict) -> list[Probe]:
    """Only one-to-one facts can be reversed: inventions and nicknames are unique, and never superseded."""
    out = []
    for key in world.fact_keys:
        fact = world.versions(key)[0]
        for template in templates["reverse"].get(fact.attr, []):
            out.append(
                Probe(
                    id=f"reverse:{fact.id}",
                    kind="reverse",
                    query=template.format(value=fact.value),
                    # The nickname probe names the nickname, so repeating it back must not count.
                    gold=tuple(a for a in world.entities[fact.subject].aliases if a != fact.value),
                    vocab="person",
                    cluster=fact.key,
                    valid_from=fact.night,
                    episode_ids=(fact.episode_id,),
                )
            )
    return out


def _alias(world: "SyntheticWorld", templates: dict) -> list[Probe]:
    """Asked through the nickname, so available only once both the nickname and the fact have arrived."""
    out = []
    for key in world.fact_keys:
        versions = world.versions(key)
        for index, fact in enumerate(versions):
            forms = templates["alias"].get(fact.attr)
            if not forms:
                continue
            nickname = world.versions(f"{fact.subject}.nickname")[0]
            start = max(fact.night, nickname.night)
            until = versions[index + 1].night if index + 1 < len(versions) else None
            if until is not None and until <= start:
                continue
            stale = tuple(form for old in versions[:index] for form in world.answer_forms(old))
            for template in forms:
                out.append(
                    Probe(
                        id=f"alias:{fact.id}",
                        kind="alias",
                        query=template.format(alias=nickname.value),
                        gold=world.answer_forms(fact),
                        stale=stale,
                        vocab=fact.attr,
                        cluster=fact.key,
                        valid_from=start,
                        valid_to=until,
                        episode_ids=(fact.episode_id, nickname.episode_id),
                    )
                )
    return out


def _two_hop(world: "SyntheticWorld", templates: dict) -> list[Probe]:
    out = []
    for key in world.fact_keys:
        versions = world.versions(key)
        for first_attr, second_attr in CHAINS:
            if versions[0].attr != first_attr:
                continue
            forms = templates["two_hop"][f"{first_attr}>{second_attr}"]
            for index, first in enumerate(versions):
                second = world.versions(f"{first.value_id}.{second_attr}")[0]
                start = max(first.night, second.night)
                until = versions[index + 1].night if index + 1 < len(versions) else None
                if until is not None and until <= start:
                    continue
                stale = tuple(
                    form
                    for old in versions[:index]
                    for form in world.answer_forms(world.versions(f"{old.value_id}.{second_attr}")[0])
                )
                subject = world.entities[first.subject].name
                for template in forms:
                    out.append(
                        Probe(
                            id=f"two_hop:{first.id}>{second_attr}",
                            kind="two_hop",
                            query=template.format(subject=subject),
                            gold=world.answer_forms(second),
                            stale=stale,
                            vocab=second_attr,
                            cluster=first.key,
                            valid_from=start,
                            valid_to=until,
                            episode_ids=(first.episode_id, second.episode_id),
                            same_night=first.night == second.night,
                            ripple=index > 0,
                        )
                    )
    return out


def _preservation(world: "SyntheticWorld", templates: dict) -> list[Probe]:
    """Attributes never changed, of a person who had another attribute changed, from that change onward."""
    changed_on: dict[str, int] = {}
    for key in world.fact_keys:
        versions = world.versions(key)
        if len(versions) > 1:
            subject = versions[0].subject
            changed_on[subject] = min(changed_on.get(subject, versions[1].night), versions[1].night)
    out = []
    for key in world.fact_keys:
        versions = world.versions(key)
        fact = versions[0]
        if len(versions) > 1 or fact.subject not in changed_on:
            continue
        subject = world.entities[fact.subject].name
        for template in templates["direct"][fact.attr]:
            out.append(
                Probe(
                    id=f"preservation:{fact.id}",
                    kind="preservation",
                    query=template.format(subject=subject),
                    gold=world.answer_forms(fact),
                    vocab=fact.attr,
                    cluster=fact.key,
                    valid_from=max(fact.night, changed_on[fact.subject]),
                    episode_ids=(fact.episode_id,),
                )
            )
    return out


def _false_recall(world: "SyntheticWorld", templates: dict) -> list[Probe]:
    """Never-taught look-alikes. Active from night 0: a fixed pool, so the rate is comparable across nights."""
    out = []
    for lookalike in world.lookalikes:
        for attr in LOOKALIKE_ATTRS[lookalike.kind]:
            target = world.versions(f"{lookalike.target}.{attr}")[0]
            for template in templates["direct"][attr]:
                out.append(
                    Probe(
                        id=f"false_recall:{lookalike.id}.{attr}",
                        kind="false_recall",
                        query=template.format(subject=lookalike.name),
                        gold=(),
                        vocab=attr,
                        cluster=f"{lookalike.id}.{attr}",
                        valid_from=0,
                        neighbour=world.answer_forms(target),
                    )
                )
    return out


def _entrenchment(world: "SyntheticWorld", templates: dict) -> list[Probe]:
    """H6: the same contradicting claim, about a taught fact and about a never-taught person.

    Pairs match on attribute, claimed value and wording, so the only difference is whether the subject's fact
    could have been consolidated. Facts that are ever superseded are left out, since their true value moves.
    """
    from ..worlds.names import SeededRng  # Local: `worlds` imports this module at load time.

    if not world.control_persons or world.config.n_entrench <= 0:
        return []
    rng = SeededRng(derive_seed(world.seed, -1, "world:entrench"))
    eligible = [
        key
        for key in world.fact_keys
        if len(world.versions(key)) == 1 and world.versions(key)[0].attr in templates["contradiction"]
    ]
    out = []
    for index, key in enumerate(rng.sample(eligible, world.config.n_entrench)):
        fact = world.versions(key)[0]
        claim = world.contradicting_value(fact, rng)
        control = world.control_persons[index % len(world.control_persons)]
        subject = world.entities[fact.subject].name
        direct = templates["direct"][fact.attr][0]
        wording = templates["contradiction"][fact.attr][0]
        cluster = f"entrench-{index:04d}"
        gold = world.value_forms(fact.attr, claim)
        out.append(
            Probe(
                id=f"entrench_fact:{index}",
                kind="entrench_fact",
                query=direct.format(subject=subject),
                gold=gold,
                stale=world.answer_forms(fact),
                vocab=fact.attr,
                cluster=cluster,
                valid_from=fact.night,
                episode_ids=(fact.episode_id,),
                context=wording.format(subject=subject, value=claim),
            )
        )
        out.append(
            Probe(
                id=f"entrench_control:{index}",
                kind="entrench_control",
                query=direct.format(subject=control),
                gold=gold,
                stale=world.answer_forms(fact),
                vocab=fact.attr,
                cluster=cluster,
                valid_from=fact.night,
                context=wording.format(subject=control, value=claim),
            )
        )
    return out
