"""E0b's labels: does this free-text question ask for the value its own episode states? (A2.3, benchmark only.)

A positive is a question that:
  1. does not contain the answer: no form of the value, nor any word of a multi-word literal value;
  2. names the subject, by any alias or, for a person, the first name;
  3. uses a keyword of the fact's attribute, and of no sibling attribute of the same kind of subject;
  4. is not yes-or-no, uses an interrogative the attribute can answer, and has no other content word.

The third class holds questions the episode leaves unanswered: about a sibling attribute, or about nothing the fact
touches. Everything uncertain is excluded from both and counted by reason, rather than forced into either. So the rule
errs towards a clean positive and a clean third class, at the price of fewer of each. A wrong label costs more than a
lost question: E0 is thought to have failed partly on mislabelled positives, and the third class is what selection must
reject. Questions the episode does answer, but not with its value ("Who plays the oboe?"), are excluded under reason
`contains_value`, since A2.3's positive is defined by the value.

The rule is keywords, so some errors remain. `run_e0b.py` writes samples of every class and reason, so that they can be
read before the numbers are believed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .answers import contains_term, normalize_tokens

POSITIVE = "positive"
THIRD = "third"
EXCLUDED = "excluded"


@dataclass(frozen=True)
class Family:
    """`strong`: says which attribute is asked about. `soft`: near it, but the episode may not answer it.

    `wh`: the interrogatives this attribute's value can answer.
    """

    strong: frozenset[str]
    soft: frozenset[str]
    wh: frozenset[str]


def _family(strong: str, soft: str, wh: str) -> Family:
    return Family(frozenset(strong.split()), frozenset(soft.split()), frozenset(wh.split()))


FAMILIES: dict[str, Family] = {
    "employer": _family(
        "work works worked working employer employers employs employed employ company firm organisation organization "
        "pays paid pay payroll wages salary job",
        "occupation profession career living business boss colleague colleagues staff office",
        "what which who whose where",
    ),
    "hometown": _family(
        "hometown town grew grow growing raised childhood child from upbringing",
        "home live lives lived living reside resides city place",
        "what which where",
    ),
    "instrument": _family(
        "instrument instruments play plays playing played practise practises practice practices perform performs "
        "performing",
        "music musical musician band song concert evening evenings",
        "what which",
    ),
    "birth_year": _family("born birth birthday birthdate year", "age old", "what which when"),
    "invention": _family(
        "invent invents invented inventing inventor invention",
        "create creates created creator design designs designed designer device build built make makes made come comes "
        "came",
        "what which",
    ),
    "nickname": _family("nickname nicknames nicknamed call calls called alias", "name known friends", "what which"),
    "hq_city": _family(
        "headquarters headquartered hq based base office where", "located location city site", "what which where"
    ),
    "product": _family(
        "make makes made manufacture manufactures manufactured manufacturer produce produces produced product products "
        "goods sell sells",
        "known business",
        "what which",
    ),
    "region": _family(
        "region regions lie lies located location where area district",
        "county province state country part",
        "what which where",
    ),
}
# Only attributes a subject of that kind can have compete: "where" names hq_city for an org, nothing for a person.
_SIBLINGS: dict[str, tuple[str, ...]] = {
    **{a: ("employer", "hometown", "instrument", "birth_year", "invention", "nickname") for a in FAMILIES},
    "hq_city": ("hq_city", "product"),
    "product": ("hq_city", "product"),
    "region": ("region",),
}
_WH = frozenset("what which who whom whose where when why how".split())
_YES_NO_OPENERS = frozenset("is are was were do does did has have had can could will would should".split())
_META = frozenset("text passage sentence paragraph document context above mentioned statement article says say".split())
_STOP = _WH | _YES_NO_OPENERS | frozenset(
    "be been being of in on at to for by with and or but s t his her he she they them their its it this that these "
    "those about as up".split()
)
_FILLER = frozenset(
    "name names exactly exact currently current main primary full specific kind type sort actually real person someone "
    "people known come comes came get gets go goes there please musical friends device".split()
)


@dataclass(frozen=True)
class QueryLabel:
    cls: str
    reason: str


def label_question(query: str, attr: str, subject_forms: Sequence[str], value_terms: Sequence[str]) -> QueryLabel:
    """Pure. `subject_forms`: every way of naming the subject. `value_terms`: anything that would give the answer away."""
    tokens = normalize_tokens(query)
    if not tokens:
        return QueryLabel(EXCLUDED, "empty")
    if any(contains_term(tokens, normalize_tokens(term)) for term in value_terms):
        return QueryLabel(EXCLUDED, "contains_value")
    subject_tokens: set[str] = set()
    names_subject = False
    for form in subject_forms:
        form_tokens = normalize_tokens(form)
        if contains_term(tokens, form_tokens):
            names_subject = True
            subject_tokens.update(form_tokens)
    body = set(tokens) - subject_tokens
    if body & _META:
        return QueryLabel(EXCLUDED, "about_the_text")
    family = FAMILIES[attr]
    rest = body - _STOP
    asks_own = bool(body & family.strong)
    asks_sibling = any(body & FAMILIES[other].strong for other in _SIBLINGS[attr] if other != attr)
    if asks_own:
        if asks_sibling:
            return QueryLabel(EXCLUDED, "several_topics")
        if not names_subject:
            return QueryLabel(EXCLUDED, "same_topic_no_subject")
        if tokens[0] in _YES_NO_OPENERS:
            return QueryLabel(EXCLUDED, "yes_no")
        if (body & _WH) - family.wh:
            return QueryLabel(EXCLUDED, "wrong_interrogative")
        if rest - family.strong - _FILLER:  # "the boss at Dara's company" has the family's word and asks something else.
            return QueryLabel(EXCLUDED, "extra_terms")
        return QueryLabel(POSITIVE, "asks_value")
    if body & family.soft:
        return QueryLabel(EXCLUDED, "related_topic")
    if asks_sibling:
        return QueryLabel(THIRD, "sibling_attribute")
    if not rest - _FILLER:
        return QueryLabel(EXCLUDED, "generic")
    return QueryLabel(THIRD, "off_topic")


def fact_terms(world: Any, fact: Any) -> tuple[list[str], list[str]]:
    """(subject forms, value terms) of a world's fact."""
    entity = world.entities[fact.subject]
    value_forms = list(world.answer_forms(fact))
    value_terms = list(value_forms)
    if fact.value_id is None:  # A literal value: "brass hinges" or a device. Any of its words gives it away.
        value_terms.extend(normalize_tokens(fact.value))
    subject_forms = [alias for alias in entity.aliases if alias not in value_forms]
    if entity.kind == "person":
        subject_forms.append(entity.name.split()[0])
    return subject_forms, value_terms


def label_for_episode(world: Any, query: str, episode_id: str) -> QueryLabel:
    fact = world.fact_of_episode(episode_id)
    subject_forms, value_terms = fact_terms(world, fact)
    return label_question(query, fact.attr, subject_forms, value_terms)
