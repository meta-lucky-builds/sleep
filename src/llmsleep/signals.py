"""Controlled dependence on memory: a Monte Carlo sequence KL between two contexts (THEORY section 3).

    CHD(x; e, d) = KL( p(y|x,e) || p(y|x,d) ) ~= (1/K) sum_k [ log p(y_k|x,e) - log p(y_k|x,d) ],  y_k ~ p(.|x,e)

Samples come from the e side at temperature 1, or the estimator is not this KL. Both log-probs cover the
same tokens, with EOS scored on both sides when it was sampled.

Truncation: y is "the continuation up to and including EOS, or the first m tokens if EOS has not come by
then". A sample without EOS is therefore a legitimate outcome, the event that those m tokens occurred, and
its probability is the product of its token conditionals. The estimator is unbiased for the KL between the
laws of this truncated y, a prefix KL. By data processing that is a lower bound on the KL of untruncated
sequences, and it equals E[min(L, m)] * KL per token when tokens are iid with stopping length L.

The estimate can be slightly negative; it is left so, because clamping would bias it upward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from .handle import ModelHandle, Sample
from .ledger import LedgerExhausted


@dataclass
class CHDResult:
    """All per query. `values` in nats, `per_token` the mean of log-ratio / length, `stderr` of `values`."""

    values: Tensor
    per_token: Tensor
    stderr: Tensor
    samples: list[list[Sample]]


class SignalCache:
    """Samples and estimates keyed on (query, ctx, banks_version), so that budget isn't paid twice.

    Lives for one night. The bank version in the key changes with any weight change, including a rollback.
    """

    def __init__(self) -> None:
        self._samples: dict[tuple, list[Sample]] = {}
        self._values: dict[tuple, tuple[float, float, float, list[Sample]]] = {}

    def get_samples(self, key: tuple, n: int) -> Optional[list[Sample]]:
        samples = self._samples.get(key)
        return samples[:n] if samples is not None and len(samples) >= n else None

    def put_samples(self, key: tuple, samples: list[Sample]) -> None:
        self._samples[key] = samples

    def get_value(self, key: tuple) -> Optional[tuple[float, float, float, list[Sample]]]:
        return self._values.get(key)

    def put_value(self, key: tuple, value: tuple[float, float, float, list[Sample]]) -> None:
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
    samples = handle.sample([handle.format(query, context)], n, max_new_tokens, 1.0, generator)[0]
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
            drawn = handle.sample(prompts, k, max_new_tokens, 1.0, generator)
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
            cursor += len(samples)
            stderr = float(ratios.std() / k**0.5) if k > 1 else float("nan")
            entries[i] = (float(ratios.mean()), float((ratios / lengths).mean()), stderr, samples)
            if cache is not None:
                cache.put_value((queries[i], ctx_e[i], ctx_d[i], bank_key, k, max_new_tokens), entries[i])

    return CHDResult(
        values=torch.tensor([e[0] for e in entries], dtype=torch.float64),
        per_token=torch.tensor([e[1] for e in entries], dtype=torch.float64),
        stderr=torch.tensor([e[2] for e in entries], dtype=torch.float64),
        samples=[e[3] for e in entries],
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
) -> tuple[list[Optional[float]], bool]:
    """CHD for (query, ctx_e, ctx_d) items in chunks. On exhaustion keeps what finished: (values, exhausted)."""
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
        values[start : start + len(part)] = result.values.tolist()
    return values, False
