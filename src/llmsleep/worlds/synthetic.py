"""SyntheticWorld: a stream of fictional facts with supersession, and everything needed to probe it.

Each episode states one fact, so that superseding a value retires nothing else. A statement never contains a
second fact: if replay data co-located two-hop facts, every replay arm would get what `pair_contexts` tests
(T2 finding 4, H7). Paraphrases and explicit reversals do go in `meta` (T2 findings 2, 3); arm flags decide
whether they are used.

Episode `entities` hold named entities only. An instrument or a year is a value, and listing it would make
unrelated people "share an entity" and flood `pairs_sharing_entity`. Distractor values come from separate
lists, so a distractor can never contain a stream answer.

Everything derives from the seed through `SeededRng`. Nothing depends on hash randomisation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from ..config import derive_seed
from ..eval.probes import Probe, build_probes
from ..memory import Episode
from .names import NameForge, SeededRng
from .templates import (
    TRAIN_PAIR_TEMPLATES,
    TRAIN_TEMPLATES,
    check_disjoint,
    eval_templates_sha,
    iter_eval_templates,
    iter_train_templates,
    load_eval_templates,
)

INSTRUMENTS = (
    "bassoon cornet dulcimer harpsichord lute mandolin marimba oboe piccolo sitar theremin tuba viola zither "
    "clarinet trombone"
).split()
PRODUCTS = (
    "barometers",
    "brass hinges",
    "copper kettles",
    "ferry engines",
    "glass lenses",
    "tide gauges",
    "wool blankets",
    "signal lamps",
    "paper fans",
    "steel springs",
)
BIRTH_YEARS = tuple(str(year) for year in range(1950, 2000))
_ORG_KINDS = "Foundry Instruments Labs Works Cooperative Mills Holdings Press".split()
_REGION_KINDS = "Valley Coast Highlands Basin Plains".split()
_DEVICE_NOUNS = "valve loom lathe compass gauge kiln sextant winch bellows pulley".split()
# Distractor-only values, so no distractor can state a stream answer.
_FILLER_INSTRUMENTS = "ocarina bagpipes kalimba didgeridoo harmonium".split()
_FILLER_NOUNS = "turbine sluice treadle ratchet".split()
_FILLER_SEASONS = "spring summer autumn winter".split()

PERSON_ATTRS = ("employer", "hometown", "instrument", "birth_year", "invention", "nickname")
MUTABLE_ATTRS = ("employer", "hometown", "instrument")
_NAMED_VALUE_ATTRS = ("employer", "hometown", "hq_city", "region", "invention", "nickname")


@dataclass(frozen=True)
class WorldConfig:
    """Sizes. Facts = 6 per person + 2 per org + 1 per city, plus supersessions."""

    n_persons: int = 40
    n_orgs: int = 8
    n_cities: int = 8
    n_regions: int = 4
    nights: int = 10
    supersession_rate: float = 0.2  # Fraction of mutable facts changed once at some later night.
    n_paraphrases: int = 4  # Paraphrases in meta["statements"], besides the episode text.
    lookalike_frac: float = 1.0  # Look-alikes per person; half that per org.
    n_distractors: int = 60
    n_control_persons: int = 20  # Never-taught people, for entrenchment.
    n_entrench: int = 60
    min_name_distance: int = 3
    banned_tokens: Sequence[str] = ()  # Name words E0's familiarity check says to replace.

    def __post_init__(self) -> None:
        if self.n_persons < 1 or self.n_orgs < 2 or self.n_cities < 2 or self.n_regions < 1 or self.nights < 1:
            raise ValueError("a world needs persons, two orgs, two cities, a region and a night")
        if not 0.0 <= self.supersession_rate <= 1.0:
            raise ValueError("supersession_rate must be in [0, 1]")


@dataclass(frozen=True)
class Entity:
    id: str
    kind: str
    name: str
    aliases: tuple[str, ...]  # Accepted answer forms, the full name first.


@dataclass(frozen=True)
class Fact:
    """`key` names the attribute across versions; `id` is one version of it."""

    id: str
    key: str
    subject: str
    attr: str
    value: str
    value_id: Optional[str]
    night: int
    version: int
    episode_id: str = ""


@dataclass(frozen=True)
class Lookalike:
    """A near-name, never taught, sharing attribute types with `target`."""

    id: str
    kind: str
    name: str
    target: str


class SyntheticWorld:
    """Implements `WorldLike`. Raises `TemplateLeakError` at construction if invariant 6 is broken."""

    def __init__(
        self,
        config: WorldConfig,
        seed: int,
        eval_templates: Optional[dict] = None,
        train_templates: Optional[dict] = None,
        train_pair_templates: Optional[dict] = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.nights = config.nights
        self._train = train_templates if train_templates is not None else TRAIN_TEMPLATES
        self._pair_train = train_pair_templates if train_pair_templates is not None else TRAIN_PAIR_TEMPLATES
        self._eval = eval_templates if eval_templates is not None else load_eval_templates()
        check_disjoint(iter_train_templates(self._train, self._pair_train), iter_eval_templates(self._eval))

        self.entities: dict[str, Entity] = {}
        self._forge = NameForge(self._rng("names"), config.min_name_distance, config.banned_tokens)
        self._build_entities()
        self._facts_by_key: dict[str, list[Fact]] = {}
        self.fact_keys: list[str] = []
        self._by_night: list[list[Episode]] = [[] for _ in range(config.nights)]
        self._fact_by_episode: dict[str, Fact] = {}
        self._vocab: dict[str, tuple[str, ...]] = {}
        self._build_stream()
        self.lookalikes: list[Lookalike] = []
        self._build_lookalikes()
        self._build_vocabulary()
        self._distractors: list[Episode] = []
        self._build_distractors()
        self.control_persons: list[str] = []
        for _ in range(config.n_control_persons):
            self.control_persons.append(f"{self._forge.token()} {self._forge.token()}")
        self.probes: tuple[Probe, ...] = build_probes(self, self._eval)  # Frozen here.

    def _rng(self, purpose: str) -> SeededRng:
        return SeededRng(derive_seed(self.seed, -1, f"world:{purpose}"))

    # ---- entities ----------------------------------------------------------------------------------

    def _add_entity(self, entity: Entity) -> Entity:
        self.entities[entity.id] = entity
        return entity

    def _build_entities(self) -> None:
        cfg, forge = self.config, self._forge
        rng = self._rng("entity_kinds")
        self.regions = []
        for i in range(cfg.n_regions):
            core = forge.token()
            name = f"{core} {rng.choice(_REGION_KINDS)}"
            self.regions.append(self._add_entity(Entity(f"R{i:04d}", "region", name, (name, core))))
        self.cities = []
        for i in range(cfg.n_cities):
            name = forge.token()
            self.cities.append(self._add_entity(Entity(f"C{i:04d}", "city", name, (name,))))
        self.orgs = []
        self._org_core: dict[str, str] = {}
        for i in range(cfg.n_orgs):
            core = forge.token()
            name = f"{core} {rng.choice(_ORG_KINDS)}"
            self.orgs.append(self._add_entity(Entity(f"O{i:04d}", "org", name, (name, core))))
            self._org_core[f"O{i:04d}"] = core
        self.persons = []
        self._surname: dict[str, str] = {}
        self._first: dict[str, str] = {}
        self._nickname: dict[str, str] = {}
        self._device: dict[str, str] = {}
        for i in range(cfg.n_persons):
            first, surname, nickname = forge.token(), forge.token(), forge.token(short=True)
            name = f"{first} {surname}"
            entity_id = f"P{i:04d}"
            self.persons.append(self._add_entity(Entity(entity_id, "person", name, (name, surname, nickname))))
            self._first[entity_id], self._surname[entity_id], self._nickname[entity_id] = first, surname, nickname
            self._device[entity_id] = f"{forge.token()} {rng.choice(_DEVICE_NOUNS)}"

    # ---- the stream --------------------------------------------------------------------------------

    def _initial_facts(self) -> list[Fact]:
        rng = self._rng("values")
        out: list[Fact] = []

        def add(subject: str, attr: str, value: str, value_id: Optional[str] = None) -> None:
            out.append(Fact(f"{subject}.{attr}#0", f"{subject}.{attr}", subject, attr, value, value_id, 0, 0))

        for person in self.persons:
            org, city = rng.choice(self.orgs), rng.choice(self.cities)
            add(person.id, "employer", org.name, org.id)
            add(person.id, "hometown", city.name, city.id)
            add(person.id, "instrument", rng.choice(INSTRUMENTS))
            add(person.id, "birth_year", rng.choice(BIRTH_YEARS))
            add(person.id, "invention", self._device[person.id])
            add(person.id, "nickname", self._nickname[person.id])
        for org in self.orgs:
            city = rng.choice(self.cities)
            add(org.id, "hq_city", city.name, city.id)
            add(org.id, "product", rng.choice(PRODUCTS))
        for city in self.cities:
            region = rng.choice(self.regions)
            add(city.id, "region", region.name, region.id)
        return out

    def _replacement(self, fact: Fact, rng: SeededRng) -> tuple[str, Optional[str]]:
        if fact.attr == "employer":
            other = rng.choice([o for o in self.orgs if o.id != fact.value_id])
            return other.name, other.id
        if fact.attr == "hometown":
            other = rng.choice([c for c in self.cities if c.id != fact.value_id])
            return other.name, other.id
        return rng.choice([i for i in INSTRUMENTS if i != fact.value]), None

    def _build_stream(self) -> None:
        cfg = self.config
        schedule_rng = self._rng("schedule")
        facts = schedule_rng.shuffled(self._initial_facts())
        placed: list[Fact] = []
        for index, fact in enumerate(facts):  # Even load per night, whatever the order.
            night = index * cfg.nights // len(facts)
            placed.append(_with(fact, night=night))
        placed.sort(key=lambda f: f.key)  # Fixed order for the supersession draw below.

        change_rng = self._rng("supersession")
        mutable = [f for f in placed if f.attr in MUTABLE_ATTRS and f.night < cfg.nights - 1]
        changed: list[Fact] = []
        count = round(cfg.supersession_rate * len(mutable))
        for fact in change_rng.sample(mutable, count):
            night = fact.night + 1 + change_rng.below(cfg.nights - 1 - fact.night)
            value, value_id = self._replacement(fact, change_rng)
            changed.append(
                Fact(f"{fact.key}#1", fact.key, fact.subject, fact.attr, value, value_id, night, 1)
            )

        order_rng = self._rng("night_order")
        by_night: list[list[Fact]] = [[] for _ in range(cfg.nights)]
        for fact in sorted(placed + changed, key=lambda f: f.id):
            by_night[fact.night].append(fact)
        latest: dict[str, str] = {}  # Key -> episode id of its newest version so far.
        final: list[Fact] = []
        for night, nightly in enumerate(by_night):
            for index, fact in enumerate(order_rng.shuffled(nightly)):
                episode_id = f"n{night:03d}-{index:05d}"
                fact = _with(fact, episode_id=episode_id)
                final.append(fact)
                episode = self._episode_for(fact, latest.get(fact.key))
                self._by_night[night].append(episode)
                self._fact_by_episode[episode_id] = fact
                latest[fact.key] = episode_id
        grouped: dict[str, list[Fact]] = {}
        for fact in final:
            grouped.setdefault(fact.key, []).append(fact)
        self.fact_keys = sorted(grouped)
        self._facts_by_key = {key: sorted(grouped[key], key=lambda f: f.version) for key in self.fact_keys}

    def _episode_for(self, fact: Fact, superseded: Optional[str]) -> Episode:
        forms = self._train[fact.attr]
        subject = self.entities[fact.subject].name
        rng = SeededRng(derive_seed(self.seed, fact.version, f"world:forms:{fact.key}"))
        paraphrases = [t.format(subject=subject, value=fact.value) for t in rng.shuffled(forms["statements"])]
        text = paraphrases[0]
        if superseded is not None:
            text = f"Update: {text}"
        statements = paraphrases[1 : 1 + self.config.n_paraphrases]
        statements += [t.format(subject=subject, value=fact.value) for t in forms["reversals"]]
        qa = [[t.format(subject=subject), fact.value] for t in forms["qa"]]
        qa += [[t.format(value=fact.value), subject] for t in forms["reverse_qa"]]
        entities = [subject]
        if fact.attr in _NAMED_VALUE_ATTRS:
            entities.append(fact.value)
        return Episode(
            id=fact.episode_id,
            text=text,
            night=fact.night,
            entities=entities,
            supersedes=[superseded] if superseded is not None else [],
            meta={"statements": statements, "qa": qa, "fact": fact.id},
        )

    # ---- look-alikes, vocabulary, distractors --------------------------------------------------------

    def _build_lookalikes(self) -> None:
        cfg, forge = self.config, self._forge
        n_people = min(cfg.n_persons, int(round(cfg.lookalike_frac * cfg.n_persons)))
        for i, person in enumerate(self.persons[:n_people]):
            surname = forge.near(self._surname[person.id])
            name = f"{self._first[person.id]} {surname}"
            self.lookalikes.append(Lookalike(f"LP{i:04d}", "person", name, person.id))
        n_orgs = min(cfg.n_orgs, int(round(cfg.lookalike_frac * cfg.n_orgs / 2)))
        for i, org in enumerate(self.orgs[:n_orgs]):
            core = self._org_core[org.id]
            name = org.name.replace(core, forge.near(core), 1)
            self.lookalikes.append(Lookalike(f"LO{i:04d}", "org", name, org.id))

    def _build_vocabulary(self) -> None:
        def forms(entities: Sequence[Entity]) -> tuple[str, ...]:
            return tuple(form for entity in entities for form in entity.aliases)

        self._vocab = {
            "employer": forms(self.orgs),
            "hometown": forms(self.cities),
            "hq_city": forms(self.cities),
            "region": forms(self.regions),
            "instrument": tuple(INSTRUMENTS),
            "birth_year": BIRTH_YEARS,
            "product": PRODUCTS,
            "invention": tuple(self._device[p.id] for p in self.persons),
            "nickname": tuple(self._nickname[p.id] for p in self.persons),
            "person": forms(self.persons) + tuple(x.name for x in self.lookalikes if x.kind == "person"),
        }

    def _build_distractors(self) -> None:
        """Half in the stream's own form about other people, so CHD cancels style; half plainly unrelated."""
        rng, forge = self._rng("distractors"), self._forge
        for i in range(self.config.n_distractors):
            if i % 2 == 0:
                person = f"{forge.token()} {forge.token()}"
                choice = rng.below(3)
                if choice == 0:
                    text = f"{person} plays the {rng.choice(_FILLER_INSTRUMENTS)}."
                elif choice == 1:
                    text = f"{person} was born in {1900 + rng.below(50)}."
                else:
                    text = f"{person} invented the {forge.token()} {rng.choice(_FILLER_NOUNS)}."
                entities = [person]
            else:
                town = forge.token()
                text = (
                    f"The {town} ferry runs {2 + rng.below(30)} crossings each week in {rng.choice(_FILLER_SEASONS)}."
                )
                entities = [town]
            self._distractors.append(Episode(f"filler-{i:05d}", text, -1, entities=entities))

    # ---- WorldLike -------------------------------------------------------------------------------------

    def episodes(self, night: int) -> list[Episode]:
        if not 0 <= night < self.nights:
            return []
        return list(self._by_night[night])

    def template_queries(self, episode: Episode) -> list[str]:
        """Oracle queries in training templates, both directions."""
        fact = self._fact_by_episode.get(episode.id)
        if fact is None:
            return []
        forms = self._train[fact.attr]
        subject = self.entities[fact.subject].name
        out = [t.format(subject=subject) for t in forms["qa"]]
        out += [t.format(value=fact.value) for t in forms["reverse_qa"]]
        return out

    def pair_template_queries(self, a: Episode, b: Episode) -> list[str]:
        """Two-hop queries, if the two episodes are the two links of a chain; otherwise none."""
        first, second = self._fact_by_episode.get(a.id), self._fact_by_episode.get(b.id)
        if first is None or second is None:
            return []
        for x, y in ((first, second), (second, first)):
            forms = self._pair_train.get((x.attr, y.attr))
            if forms and x.value_id == y.subject:
                subject = self.entities[x.subject].name
                return [t.format(subject=subject) for t in forms]
        return []

    def distractor_pool(self) -> list[Episode]:
        return list(self._distractors)

    # ---- for probes and evaluation ---------------------------------------------------------------------

    def versions(self, key: str) -> list[Fact]:
        return self._facts_by_key[key]

    def fact_of_episode(self, episode_id: str) -> Optional[Fact]:
        return self._fact_by_episode.get(episode_id)

    def episode(self, episode_id: str) -> Episode:
        fact = self._fact_by_episode[episode_id]
        for episode in self._by_night[fact.night]:
            if episode.id == episode_id:
                return copy.deepcopy(episode)
        raise KeyError(episode_id)

    def answer_forms(self, fact: Fact) -> tuple[str, ...]:
        """Accepted answers for a fact's value: all aliases of a linked entity, or the literal value."""
        if fact.value_id is not None:
            return self.entities[fact.value_id].aliases
        return (fact.value,)

    def value_forms(self, attr: str, value: str) -> tuple[str, ...]:
        """Accepted forms of a value of `attr`, given its display string."""
        for entity in self.entities.values():
            if entity.name == value:
                return entity.aliases
        return (value,)

    def vocabulary(self, attr: str) -> tuple[str, ...]:
        """All same-type answers of an attribute, or of "person"."""
        return self._vocab[attr]

    def answer_groups(self, attr: str) -> list[tuple[str, ...]]:
        """The distinct answers of an attribute, each with its accepted forms: how many values chance divides by."""
        by_kind = {"employer": self.orgs, "hometown": self.cities, "hq_city": self.cities, "region": self.regions}
        if attr in by_kind:
            return [entity.aliases for entity in by_kind[attr]]
        return [(form,) for form in self._vocab[attr]]

    def contradicting_value(self, fact: Fact, rng: SeededRng) -> str:
        """A same-type value that is not the fact's own."""
        if fact.attr == "employer":
            pool = [o.name for o in self.orgs]
        elif fact.attr == "hometown":
            pool = [c.name for c in self.cities]
        elif fact.attr == "invention":
            pool = [self._device[p.id] for p in self.persons]
            if len(pool) < 2:  # A one-person world has no other invention to claim.
                pool = pool + [f"{fact.value} mark two"]
        elif fact.attr == "instrument":
            pool = list(INSTRUMENTS)
        elif fact.attr == "birth_year":
            pool = list(BIRTH_YEARS)
        else:
            raise ValueError(f"no contradiction for attribute {fact.attr!r}")
        return rng.choice([value for value in pool if value != fact.value])

    def fingerprint(self) -> str:
        """Identifies stream and probes together, so results can show they were the same across arms."""
        digest = hashlib.sha256()
        digest.update(eval_templates_sha().encode("utf-8"))
        for night in range(self.nights):
            for episode in self._by_night[night]:
                digest.update(json.dumps(asdict(episode), sort_keys=True).encode("utf-8"))
        for probe in self.probes:
            digest.update(json.dumps(probe.to_dict(), sort_keys=True).encode("utf-8"))
        return digest.hexdigest()[:16]


def _with(fact: Fact, **changes) -> Fact:
    values = asdict(fact)
    values.update(changes)
    return Fact(**values)
