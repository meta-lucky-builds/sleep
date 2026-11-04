"""Controlled dependence on memory: a Monte Carlo sequence KL between two contexts (THEORY section 3).

    CHD(x; e, d) = KL( p(y|x,e) || p(y|x,d) ) ~= (1/K) sum_k [ log p(y_k|x,e) - log p(y_k|x,d) ],  y_k ~ p(.|x,e)

Samples come from the e side at temperature 1, or the estimator is not this KL. Both log-probs cover the
same tokens, with the stop token scored on both sides when it was sampled. Generation is in answer mode (A1.9): it
stops at EOS or a token containing a newline, so the signal covers the answer and not what a raw model runs on into.

Truncation: y is "the continuation up to and including the stop token, or the first m tokens if none has come by
then". A sample without one is therefore a legitimate outcome, the event that those m tokens occurred, and
its probability is the product of its token conditionals. The estimator is unbiased for the KL between the
laws of this truncated y, a prefix KL. By data processing that is a lower bound on the KL of untruncated
sequences, and it equals E[min(L, m)] * KL per token when tokens are iid with stopping length L.

The estimate can be slightly negative; it is left so, because clamping would bias it upward.

The second half of this file holds E0b's candidates for telling whether an episode bears on a query (A2.3). All are
oracle-free: they see the query, the contexts and the model. `score_candidates` computes any subset and reports what each
would cost alone, so that sharing work between candidates in one run does not flatter or penalise any of them.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .eval.answers import first_answer, normalize_answer, normalize_tokens
from .handle import ModelHandle, Sample
from .ledger import ComputeLedger, LedgerExhausted


@dataclass
class CHDResult:
    """All per query. `values` in nats, `per_token` the mean of log-ratio / length, `stderr` of `values`.

    `logp_d` holds log p(y_k | x, d) per query and sample.
    """

    values: Tensor
    per_token: Tensor
    stderr: Tensor
    samples: list[list[Sample]]
    logp_d: list[list[float]] = field(default_factory=list)


class SignalCache:
    """Samples and estimates keyed on (query, ctx, banks_version), so that budget isn't paid twice.

    Lives for one night. The bank version in the key changes with any weight change, including a rollback.
    """

    def __init__(self) -> None:
        self._samples: dict[tuple, list[Sample]] = {}
        self._values: dict[tuple, tuple[float, float, float, list[Sample], list[float]]] = {}

    def get_samples(self, key: tuple, n: int) -> Optional[list[Sample]]:
        samples = self._samples.get(key)
        return samples[:n] if samples is not None and len(samples) >= n else None

    def put_samples(self, key: tuple, samples: list[Sample]) -> None:
        self._samples[key] = samples

    def get_value(self, key: tuple) -> Optional[tuple[float, float, float, list[Sample], list[float]]]:
        return self._values.get(key)

    def put_value(self, key: tuple, value: tuple[float, float, float, list[Sample], list[float]]) -> None:
        self._values[key] = value


def _sample_key(handle: ModelHandle, query: str, context: Optional[str], max_new_tokens: int) -> tuple:
    return (query, context, handle.banks_key(), max_new_tokens)


def get_samples(
    handle: ModelHandle,
    query: str,
    context: Optional[str],
    n: int,
    max_new_tokens: int,
    generator: torch.Generator,
    cache: Optional[SignalCache] = None,
) -> list[Sample]:
    """n temperature-1 samples for one query in one context, reusing cached ones."""
    key = _sample_key(handle, query, context, max_new_tokens)
    if cache is not None:
        hit = cache.get_samples(key, n)
        if hit is not None:
            return hit
    samples = handle.sample([handle.format(query, context)], n, max_new_tokens, 1.0, generator, answer_mode=True)[0]
    if cache is not None:
        cache.put_samples(key, samples)
    return samples


def chd(
    handle: ModelHandle,
    queries: Sequence[str],
    ctx_e: Sequence[Optional[str]],
    ctx_d: Sequence[Optional[str]],
    k: int,
    max_new_tokens: int,
    generator: torch.Generator,
    cache: Optional[SignalCache] = None,
) -> CHDResult:
    """CHD per query, under the active banks. A context of None is the closed-book condition."""
    if not len(queries) == len(ctx_e) == len(ctx_d):
        raise ValueError("queries and contexts differ in length")
    if k < 1:
        raise ValueError("k must be at least 1")
    size = len(queries)
    bank_key = handle.banks_key()
    entries: list[Any] = [None] * size
    todo: list[int] = []
    for i in range(size):
        hit = cache.get_value((queries[i], ctx_e[i], ctx_d[i], bank_key, k, max_new_tokens)) if cache else None
        if hit is not None:
            entries[i] = hit
        else:
            todo.append(i)

    if todo:
        samples_by_row: dict[int, list[Sample]] = {}
        need: list[int] = []
        for i in todo:
            hit = cache.get_samples(_sample_key(handle, queries[i], ctx_e[i], max_new_tokens), k) if cache else None
            if hit is not None:
                samples_by_row[i] = hit
            else:
                need.append(i)
        if need:
            prompts = [handle.format(queries[i], ctx_e[i]) for i in need]
            drawn = handle.sample(prompts, k, max_new_tokens, 1.0, generator, answer_mode=True)
            for i, samples in zip(need, drawn):
                samples_by_row[i] = samples
                if cache is not None:
                    cache.put_samples(_sample_key(handle, queries[i], ctx_e[i], max_new_tokens), samples)

        # The e-side log-probs were recorded while sampling; score the same tokens on the d side.
        flat_prompts: list[list[int]] = []
        flat_conts: list[list[int]] = []
        for i in todo:
            prompt = handle.encode_prompt(handle.format(queries[i], ctx_d[i]))
            for sample in samples_by_row[i]:
                flat_prompts.append(prompt)
                flat_conts.append(sample.token_ids)
        logp_d = handle.sequence_logprob(flat_prompts, flat_conts).tolist()

        cursor = 0
        for i in todo:
            samples = samples_by_row[i]
            ratios = torch.tensor(
                [s.logprob - logp_d[cursor + j] for j, s in enumerate(samples)], dtype=torch.float64
            )
            lengths = torch.tensor([len(s.token_ids) for s in samples], dtype=torch.float64)
            row_logp = logp_d[cursor : cursor + len(samples)]
            cursor += len(samples)
            stderr = float(ratios.std() / k**0.5) if k > 1 else float("nan")
            entries[i] = (float(ratios.mean()), float((ratios / lengths).mean()), stderr, samples, row_logp)
            if cache is not None:
                cache.put_value((queries[i], ctx_e[i], ctx_d[i], bank_key, k, max_new_tokens), entries[i])

    return CHDResult(
        values=torch.tensor([e[0] for e in entries], dtype=torch.float64),
        per_token=torch.tensor([e[1] for e in entries], dtype=torch.float64),
        stderr=torch.tensor([e[2] for e in entries], dtype=torch.float64),
        samples=[e[3] for e in entries],
        logp_d=[e[4] for e in entries],
    )


def raw_kl_nocontext(
    handle: ModelHandle,
    queries: Sequence[str],
    ctx_e: Sequence[Optional[str]],
    k: int,
    max_new_tokens: int,
    generator: torch.Generator,
    cache: Optional[SignalCache] = None,
) -> CHDResult:
    """Control for H1: KL( p(y|x,e) || p(y|x) ), contaminated by the effect of having context at all."""
    return chd(handle, queries, ctx_e, [None] * len(queries), k, max_new_tokens, generator, cache)


def chd_safe(
    handle: ModelHandle,
    items: Sequence[tuple[str, Optional[str], Optional[str]]],
    k: int,
    max_new_tokens: int,
    generator: torch.Generator,
    cache: Optional[SignalCache] = None,
    chunk: int = 4,
    per_token: bool = True,
) -> tuple[list[Optional[float]], bool]:
    """CHD for (query, ctx_e, ctx_d) items in chunks. On exhaustion keeps what finished: (values, exhausted).

    Per answer token by default: every decision rule compares sequences of different lengths.
    """
    values: list[Optional[float]] = [None] * len(items)
    for start in range(0, len(items), chunk):
        part = items[start : start + chunk]
        try:
            result = chd(
                handle,
                [q for q, _, _ in part],
                [e for _, e, _ in part],
                [d for _, _, d in part],
                k,
                max_new_tokens,
                generator,
                cache,
            )
        except LedgerExhausted:
            return values, True
        values[start : start + len(part)] = (result.per_token if per_token else result.values).tolist()
    return values, False


# ---- E0b's candidates (A2.3) ---------------------------------------------------------------------------------

CANDIDATES = ("chd", "chd_mix", "grounding", "agreement", "entropy_drop")
INCUMBENT = "chd"

# Words that say nothing about having read an episode. Case-folded; compared with whole word tokens only.
STOPWORDS = frozenset(
    "a about all also am an and any are as at be because been being but by can could did do does each few for from "
    "had has have he her here hers him his how i if in into is it its just me more most my no nor not now of on only "
    "or other our out over own same she should so some such t than that the their them then there these they this "
    "those to too under until up very was we were what when where which while who whom whose why will with would you "
    "your s".split()
)


@dataclass(frozen=True)
class CandidateSpec:
    """`k` and `max_new_tokens` are the deployed ones. CHD_mix keeps `mix_k` of the `k` samples: see configs/README.md."""

    k: int = 4
    max_new_tokens: int = 16
    mix_k: int = 2
    mix_m: int = 4
    max_skip: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.mix_k <= self.k or self.mix_m < 1 or self.max_skip < 0:
            raise ValueError("need 1 <= mix_k <= k, mix_m >= 1 and max_skip >= 0")


@dataclass
class CandidateScores:
    """`values`: higher means the episode bears more on the query. `costs`: forward tokens each would pay alone."""

    values: dict[str, float]
    costs: dict[str, int]
    answers_e: list[str] = field(default_factory=list)


class DistractorMemo:
    """The distractor side of one query, reused by its own-episode and unrelated-episode rows.

    Those rows see the same query and distractors, so the work is identical. The candidate's cost still counts it,
    because a deployed candidate pays for it on each query.
    """

    def __init__(self) -> None:
        self.samples: dict[str, tuple[list[Sample], int]] = {}
        self.entropy: dict[str, tuple[float, int]] = {}


def _tokens_so_far(ledger: ComputeLedger) -> int:
    return ledger.exempt_forward + ledger.unmetered_forward + sum(ledger.forward.values())


def log_mean_exp(values: Tensor, dim: int = 0) -> Tensor:
    """log((1/M) sum exp(values)) along `dim`, stable where exp(values) would underflow."""
    return torch.logsumexp(values, dim=dim) - math.log(values.size(dim))


def _mix_result(samples: list[Sample], logp_by_distractor: Sequence[Sequence[float]]) -> CHDResult:
    """One query's estimate. `logp_by_distractor`: (M, K), each row log p(y_k | x, d_m), over the same samples."""
    logp_e = torch.tensor([s.logprob for s in samples], dtype=torch.float64)
    mixture = log_mean_exp(torch.tensor(logp_by_distractor, dtype=torch.float64), dim=0)
    ratios = logp_e - mixture
    lengths = torch.tensor([len(s.token_ids) for s in samples], dtype=torch.float64)
    size = len(samples)
    stderr = float(ratios.std() / size**0.5) if size > 1 else float("nan")
    return CHDResult(
        values=ratios.mean().unsqueeze(0),
        per_token=(ratios / lengths).mean().unsqueeze(0),
        stderr=torch.tensor([stderr], dtype=torch.float64),
        samples=[samples],
        logp_d=[list(mixture.tolist())],
    )


def chd_mix(
    handle: ModelHandle,
    query: str,
    ctx_e: Optional[str],
    distractors: Sequence[Optional[str]],
    k: int,
    max_new_tokens: int,
    generator: torch.Generator,
) -> CHDResult:
    """KL( p(y|x,e) || (1/M) sum_m p(y|x,d_m) ), Monte Carlo over y_k ~ p(.|x,e).

    The mixture is over whole answers, not token by token, and its density is computed exactly, so the estimate is
    unbiased, as CHD is. A distractor that gives y tiny mass cannot inflate the ratio, as it can with one distractor: the
    mixture is at least 1/M of the best distractor's probability, so the ratio exceeds its value against that distractor
    by at most log(M).
    """
    if not distractors:
        raise ValueError("the mixture needs at least one distractor")
    samples = handle.sample([handle.format(query, ctx_e)], k, max_new_tokens, 1.0, generator, answer_mode=True)[0]
    conts = [s.token_ids for s in samples]
    rows = []
    for distractor in distractors:
        prompt = handle.encode_prompt(handle.format(query, distractor))
        rows.append(handle.sequence_logprob([prompt] * len(conts), conts).tolist())
    return _mix_result(samples, rows)


def content_words(text: str) -> set[str]:
    """Case-folded whole words, without stopwords. Digits stay: a year is a value."""
    return {word for word in normalize_tokens(text) if word not in STOPWORDS}


def grounding_rate(answers: Iterable[str], target: set[str]) -> float:
    """Share of answers containing any word of `target`. No target words: 0, for either side of a comparison."""
    answers = list(answers)
    if not answers or not target:
        return 0.0
    return sum(1 for answer in answers if content_words(answer) & target) / len(answers)


def modal_agreement(answers: Iterable[str]) -> float:
    """Share of answers equal to the most common one, after cutting to the first answer and normalising."""
    counts = Counter(normalize_answer(first_answer(answer)) for answer in answers)
    total = sum(counts.values())
    return counts.most_common(1)[0][1] / total if total else 0.0


def first_answer_token_entropy(handle: ModelHandle, prompt_ids: Sequence[int], max_skip: int = 2) -> float:
    """H(first answer token), in nats. Exact, not sampled.

    The first answer token is the first token of the continuation whose text has a letter or a digit. A raw first
    token may be a lone space or punctuation mark, which says nothing about the answer: a tokenizer that spells " 1975"
    as a space and then digits would give the raw first token of a year no entropy in any context. So: follow the most
    likely token while it has neither, at most `max_skip` times, stopping if it ends the answer. Then take the
    distribution at that point, conditioned on the next token having a letter or a digit. The other tokens' mass is
    dropped and the rest renormalised, on both sides of a comparison alike.

    Each step is a full pass over the prompt so far, charged to the ledger.
    """
    prefix: list[int] = []
    for step in range(max_skip + 1):
        logits = handle.next_token_logits([list(prompt_ids) + prefix])[0]
        content = handle.content_token_mask(logits.size(-1))
        top = int(logits.argmax())
        if bool(content[top]) or step == max_skip or top in handle.stop_token_ids(logits.size(-1)):
            break
        prefix.append(top)
    if not bool(content.any()):
        return 0.0
    log_p = F.log_softmax(logits[content].double(), dim=-1)
    return float(-(log_p.exp() * log_p).sum())


def _metered(ledger: ComputeLedger, body: Callable[[], Any]) -> tuple[Any, int]:
    before = _tokens_so_far(ledger)
    out = body()
    return out, _tokens_so_far(ledger) - before


def score_candidates(
    handle: ModelHandle,
    query: str,
    e_text: str,
    ctx_e: str,
    distractors: Sequence[str],
    names: Sequence[str],
    spec: CandidateSpec,
    generator_for: Callable[[str], torch.Generator],
    memo: Optional[DistractorMemo] = None,
) -> CandidateScores:
    """The named candidates on one (query, episode) row. `generator_for(purpose)` must return a fresh seeded generator.

    Samples from e are drawn in the same order whichever candidates are named, so a candidate's value does not depend
    on which others ran. `e_text` is the episode's text, the source of grounding's words. `distractors[0]` is the
    single distractor of (a), (c), (d) and (e); (b) uses the first `mix_m`.
    """
    wanted = set(names)
    unknown = wanted - set(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    if not distractors or ("chd_mix" in wanted and len(distractors) < spec.mix_m):
        raise ValueError("not enough distractors")
    memo = memo if memo is not None else DistractorMemo()
    ledger = handle.ledger
    first = distractors[0]
    values: dict[str, float] = {}
    costs: dict[str, int] = {}

    def draw(n: int) -> list[Sample]:
        return handle.sample(prompt_e, n, spec.max_new_tokens, 1.0, stream, answer_mode=True)[0]

    prompt_e = [handle.encode_prompt(handle.format(query, ctx_e))]
    e_head: list[Sample] = []
    e_tail: list[Sample] = []
    cost_head = cost_tail = 0
    needs_all = wanted & {"chd", "grounding", "agreement"}
    if needs_all or "chd_mix" in wanted:
        stream = generator_for(f"e|{query}|{ctx_e}")
        e_head, cost_head = _metered(ledger, lambda: draw(spec.mix_k))
    if needs_all:
        e_tail, cost_tail = _metered(ledger, lambda: draw(spec.k - spec.mix_k))
    e_all = e_head + e_tail
    answers_e = [handle.decode(s.token_ids) for s in e_all]

    first_logp: Optional[list[float]] = None
    if "chd" in wanted:
        cache = SignalCache()  # So that the incumbent is literally `chd`, on these samples.
        cache.put_samples(_sample_key(handle, query, ctx_e, spec.max_new_tokens), e_all)
        result, cost = _metered(
            ledger,
            lambda: chd(handle, [query], [ctx_e], [first], spec.k, spec.max_new_tokens, stream, cache),
        )
        values["chd"] = float(result.per_token[0])
        costs["chd"] = cost_head + cost_tail + cost
        first_logp = result.logp_d[0][: spec.mix_k]

    if "chd_mix" in wanted:
        conts = [s.token_ids for s in e_head]
        rows: list[list[float]] = []
        cost = 0
        for index, distractor in enumerate(distractors[: spec.mix_m]):
            prompt = handle.encode_prompt(handle.format(query, distractor))
            if index == 0 and first_logp is not None:
                rows.append(first_logp)  # Already paid for by `chd`; what scoring these would charge, it is charged.
                cost += sum(len(prompt) + len(cont) for cont in conts)
                continue
            row, spent = _metered(ledger, lambda: handle.sequence_logprob([prompt] * len(conts), conts).tolist())
            rows.append(row)
            cost += spent
        values["chd_mix"] = float(_mix_result(e_head, rows).per_token[0])
        costs["chd_mix"] = cost_head + cost

    if wanted & {"grounding", "agreement"}:
        if first not in memo.samples:
            d_stream = generator_for(f"d|{query}|{first}")
            prompt_d = [handle.encode_prompt(handle.format(query, first))]
            memo.samples[first] = _metered(
                ledger, lambda: handle.sample(prompt_d, spec.k, spec.max_new_tokens, 1.0, d_stream, answer_mode=True)[0]
            )
        d_samples, cost_d = memo.samples[first]
        answers_d = [handle.decode(s.token_ids) for s in d_samples]
        if "grounding" in wanted:
            target = content_words(e_text) - content_words(query)
            values["grounding"] = grounding_rate(answers_e, target) - grounding_rate(answers_d, target)
            costs["grounding"] = cost_head + cost_tail + cost_d
        if "agreement" in wanted:
            values["agreement"] = modal_agreement(answers_e) - modal_agreement(answers_d)
            costs["agreement"] = cost_head + cost_tail + cost_d

    if "entropy_drop" in wanted:
        if first not in memo.entropy:
            prompt_d_ids = handle.encode_prompt(handle.format(query, first))
            memo.entropy[first] = _metered(
                ledger, lambda: first_answer_token_entropy(handle, prompt_d_ids, spec.max_skip)
            )
        entropy_d, cost_entropy_d = memo.entropy[first]
        entropy_e, cost_entropy_e = _metered(
            ledger, lambda: first_answer_token_entropy(handle, prompt_e[0], spec.max_skip)
        )
        values["entropy_drop"] = entropy_d - entropy_e
        costs["entropy_drop"] = cost_entropy_d + cost_entropy_e

    return CandidateScores(values, costs, answers_e)
