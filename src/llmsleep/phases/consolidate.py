"""Stage 2: precomputed teacher targets and the training loop.

All targets are computed before training, under explicit banks and no_grad. Training then runs with one fixed
bank set (committed + fresh) and compares with stored tensors. So no bank switching happens during forward or
backward, and a gradient-checkpoint recomputation during backward sees the same banks as its forward did.

Reduced target distribution. Storing a full vocabulary distribution per answer token is too expensive
(about 150k floats). Let T be the teacher's top-k tokens at a position. Coarsen the vocabulary to
T + {*}, where * is everything else, for both teacher p and student q:
    p'(i) = p(i), q'(i) = q(i) for i in T;  p'(*) = 1 - sum_T p,  q'(*) = 1 - sum_T q
    KL(p' || q') = sum_T p(i) log(p(i)/q(i)) + p'(*) log(p'(*)/q'(*))
This is an exact KL between genuine distributions, so it is non-negative and zero iff the student matches the
head and the tail's total mass. By the log-sum inequality KL(p'||q') <= KL(p||q), with equality iff q is
proportional to p inside the tail. So the penalty never overstates the divergence, ignores only the shape of
the tail, where teacher probabilities are near noise, and costs k+1 floats plus k indices per token. A
sample-based reduced estimate was rejected: it adds variance on the term that needs to be steady, and it
cannot represent mass on the student's side outside the sampled tokens. A student piling mass into the tail
is still penalised here through q'(*).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import AbstractSet, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import TeacherConfig, TrainConfig
from ..delta import BankSpec
from ..handle import ModelHandle, build_batch, gather_answer_logits, gather_answer_targets
from ..ledger import LedgerExhausted
from ..memory import Episode
from ..signals import SignalCache, get_samples
from .queries import normalize_query


def residual_log_mass(log_probs: Tensor, top_idx: Tensor) -> Tensor:
    """(N,) log of the total mass outside `top_idx`. Summed in log space: 1 - sum(top) loses all precision."""
    return torch.logsumexp(log_probs.scatter(-1, top_idx, float("-inf")), dim=-1)


def reduce_distribution(log_probs: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
    """(N, V) -> top indices (N, k'), top log-probs (N, k'), residual log mass (N,). k' <= V-1 leaves a tail."""
    vocab = log_probs.size(-1)
    if vocab < 2:
        raise ValueError("vocabulary too small to reduce")
    top_logp, top_idx = log_probs.topk(min(k, vocab - 1), dim=-1)
    # Clamped so that a vanishing tail gives p = 0 times a finite number, not 0 * inf.
    return top_idx, top_logp.clamp(min=-1000.0), residual_log_mass(log_probs, top_idx).clamp(min=-1000.0)


def reduced_kl(student_log_probs: Tensor, top_idx: Tensor, top_logp: Tensor, res_logp: Tensor) -> Tensor:
    """(N,) KL(p'||q') from the module docstring, differentiable in the student."""
    student_top = student_log_probs.gather(-1, top_idx)
    student_res = residual_log_mass(student_log_probs, top_idx)
    head = (top_logp.exp() * (top_logp - student_top)).sum(dim=-1)
    tail = res_logp.exp() * (res_logp - student_res)
    return head + tail


@dataclass
class TokenTargets:
    """A stored target for one sampled sequence. L = len(cont_ids), one row per answer token.

    The student sees `prompt_ids` then `cont_ids`, and is compared with the teacher at each answer position.
    """

    prompt_ids: list[int]
    cont_ids: list[int]
    top_idx: Tensor  # (L, k), CPU
    top_logp: Tensor  # (L, k), CPU
    res_logp: Tensor  # (L,), CPU
    kind: str
    episode_ids: tuple[str, ...] = ()  # Episodes this teaches; empty for anchors, which only hold things still.


@dataclass
class HardExample:
    """Real data: cross-entropy on `cont_ids` given `prompt_ids`."""

    prompt_ids: list[int]
    cont_ids: list[int]
    episode_ids: tuple[str, ...] = ()


@dataclass
class TargetSpec:
    """How to build targets for a query. The student is always closed-book."""

    query: str
    teacher_context: Optional[str]
    teacher_banks: BankSpec
    kind: str
    episode_ids: tuple[str, ...] = ()


def _targets_from_samples(
    handle: ModelHandle,
    teacher_prompt: list[int],
    student_prompt: list[int],
    continuations: list[list[int]],
    top_k: int,
    kind: str,
    episode_ids: tuple[str, ...] = (),
) -> list[TokenTargets]:
    out: list[TokenTargets] = []
    prompts = [teacher_prompt] * len(continuations)
    for rows, logits in handle.iter_answer_logits(prompts, continuations):
        log_probs = F.log_softmax(logits, dim=-1)
        top_idx, top_logp, res_logp = reduce_distribution(log_probs, top_k)
        cursor = 0
        for row in rows:
            length = len(continuations[row])
            out.append(
                TokenTargets(
                    student_prompt,
                    continuations[row],
                    top_idx[cursor : cursor + length].cpu(),
                    top_logp[cursor : cursor + length].cpu(),
                    res_logp[cursor : cursor + length].cpu(),
                    kind,
                    episode_ids,
                )
            )
            cursor += length
    return out


def build_targets(
    handle: ModelHandle,
    specs: Sequence[TargetSpec],
    cfg: TeacherConfig,
    generator: torch.Generator,
    cache: Optional[SignalCache] = None,
) -> tuple[list[TokenTargets], bool]:
    """Sample y ~ teacher(x, ctx), store teacher(. | x, ctx, y_<t) per answer token. Returns (targets, exhausted).

    Averaging the per-token KL over y sampled from the teacher estimates the teacher-to-student sequence KL,
    by the chain rule. Fresh cannot receive gradients: everything here is no_grad and the outputs are detached.
    """
    targets: list[TokenTargets] = []
    for spec in specs:
        try:
            with handle.banks(spec.teacher_banks):
                samples = get_samples(
                    handle, spec.query, spec.teacher_context, cfg.samples, cfg.max_new_tokens, generator, cache
                )
                targets.extend(
                    _targets_from_samples(
                        handle,
                        handle.encode_prompt(handle.format(spec.query, spec.teacher_context)),
                        handle.encode_prompt(handle.format(spec.query, None)),
                        [s.token_ids for s in samples],
                        cfg.top_k,
                        spec.kind,
                        spec.episode_ids,
                    )
                )
        except LedgerExhausted:
            return targets, True
    return targets, False


def build_self_sampled_targets(
    handle: ModelHandle, cfg: TeacherConfig, generator: torch.Generator
) -> tuple[list[TokenTargets], bool]:
    """B4: text sampled from germline, with germline's own distributions as targets, on all its tokens."""
    start = handle.start_ids()
    try:
        with handle.banks("germline"):
            samples = handle.sample([start], cfg.self_samples, cfg.self_max_new_tokens, 1.0, generator)[0]
            targets = _targets_from_samples(handle, start, start, [s.token_ids for s in samples], cfg.top_k, "self")
    except LedgerExhausted:
        return [], True
    return targets, False


def hard_examples(
    handle: ModelHandle, episodes: Sequence[Episode], exclude_queries: AbstractSet[str] = frozenset()
) -> list[HardExample]:
    """Episode text and `meta["statements"]` as plain text; `meta["qa"]` as closed-book prompt plus answer.

    A QA form whose normalised query is in `exclude_queries` is skipped, so hard labels cannot teach a query
    the audit holds out.
    """
    start = handle.start_ids()
    eos = [handle.eos_id] if handle.eos_id is not None else []
    out: list[HardExample] = []
    for episode in episodes:
        for text in [episode.text] + list(episode.meta.get("statements", [])):
            ids = handle.encode_continuation(text)
            if ids:
                out.append(HardExample(start, ids + eos, (episode.id,)))
        for query, answer in episode.meta.get("qa", []):
            if normalize_query(query) not in exclude_queries:
                prompt = handle.encode_prompt(handle.format(query, None))
                out.append(HardExample(prompt, handle.encode_answer(answer), (episode.id,)))
    return out


class _Cycler:
    """Uniform draws without replacement within an epoch, from a local generator."""

    def __init__(self, items: Sequence, generator: torch.Generator) -> None:
        self.items = list(items)
        self.generator = generator
        self.order: list[int] = []

    def draw(self, n: int) -> list:
        if not self.items:
            return []
        out = []
        for _ in range(n):
            if not self.order:
                self.order = torch.randperm(len(self.items), generator=self.generator).tolist()
            out.append(self.items[self.order.pop()])
        return out

    def mean_tokens(self) -> float:
        if not self.items:
            return 0.0
        return sum(len(x.prompt_ids) + len(x.cont_ids) for x in self.items) / len(self.items)


@dataclass
class TrainResult:
    """What the loop did. `exhausted` marks a partial result: the steps taken so far stand."""

    planned_steps: int = 0
    steps: int = 0
    exhausted: bool = False
    nonfinite_steps: int = 0
    loss_hard: float = 0.0  # Means over the last completed step.
    loss_kl: float = 0.0
    loss_anchor: float = 0.0
    trained_episode_ids: set[str] = field(default_factory=set)  # In at least one completed optimizer step.

    def summary(self) -> dict:
        return {**self.__dict__, "trained_episode_ids": sorted(self.trained_episode_ids)}


def compute_loss(
    handle: ModelHandle,
    hard: Sequence[HardExample],
    integrate: Sequence[TokenTargets],
    anchor: Sequence[TokenTargets],
    cfg: TrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    """L = w_h CE + w_k KL + w_a KL, each a mean over its answer tokens, all in one right-padded batch."""
    rows = list(hard) + list(integrate) + list(anchor)
    batch = build_batch([r.prompt_ids for r in rows], [r.cont_ids for r in rows], handle.pad_id, "right")
    logits = handle.forward_train(batch)
    # Row-major selection, so the (N, V) answer rows follow `rows`, each row's tokens in order.
    answer_logits = gather_answer_logits(logits, batch.answer_mask.to(logits.device)).float()
    log_probs = F.log_softmax(answer_logits, dim=-1)

    n_hard = sum(len(r.cont_ids) for r in hard)
    n_integrate = sum(len(r.cont_ids) for r in integrate)
    n_anchor = sum(len(r.cont_ids) for r in anchor)
    if n_hard + n_integrate + n_anchor != log_probs.size(0):
        raise AssertionError("answer tokens do not line up with the batch")

    loss = log_probs.new_zeros(())
    parts = {"hard": 0.0, "kl": 0.0, "anchor": 0.0}
    if n_hard:
        targets = gather_answer_targets(batch.input_ids, batch.answer_mask)[:n_hard].to(log_probs.device)
        ce = -log_probs[:n_hard].gather(-1, targets.unsqueeze(-1)).mean()
        loss = loss + cfg.w_hard * ce
        parts["hard"] = float(ce.detach())
    if n_integrate:
        kl = _stored_kl(log_probs[n_hard : n_hard + n_integrate], integrate)
        loss = loss + cfg.w_kl * kl
        parts["kl"] = float(kl.detach())
    if n_anchor:
        kl = _stored_kl(log_probs[n_hard + n_integrate :], anchor)
        loss = loss + cfg.w_anchor * kl
        parts["anchor"] = float(kl.detach())
    return loss, parts


def _stored_kl(log_probs: Tensor, targets: Sequence[TokenTargets]) -> Tensor:
    device = log_probs.device
    top_idx = torch.cat([t.top_idx for t in targets]).to(device)
    top_logp = torch.cat([t.top_logp for t in targets]).to(device)
    res_logp = torch.cat([t.res_logp for t in targets]).to(device)
    return reduced_kl(log_probs, top_idx, top_logp, res_logp).mean()


def train_loop(
    handle: ModelHandle,
    hard: Sequence[HardExample],
    integrate: Sequence[TokenTargets],
    anchor: Sequence[TokenTargets],
    cfg: TrainConfig,
    generator: torch.Generator,
    max_steps: Optional[int] = None,
) -> TrainResult:
    """Train fresh only, until the steps or the current stage's tokens run out. Cosine lr over planned steps.

    Exhaustion is caught here: partly accumulated gradients are dropped and completed steps stand.
    """
    delta = handle.delta
    if delta is None:
        raise ValueError("training needs a DeltaModel")
    delta.assert_only_fresh_trainable()
    result = TrainResult()
    hard_pool, kl_pool, anchor_pool = _Cycler(hard, generator), _Cycler(integrate, generator), _Cycler(anchor, generator)
    micro_tokens = (
        cfg.hard_batch * hard_pool.mean_tokens()
        + cfg.kl_batch * kl_pool.mean_tokens()
        + cfg.anchor_batch * anchor_pool.mean_tokens()
    )
    if micro_tokens == 0:
        return result
    limit = max_steps if max_steps is not None else cfg.max_steps
    remaining = handle.ledger.remaining()
    if remaining is None and limit is None:
        raise ValueError("an unmetered ledger needs max_steps")
    planned = limit if limit is not None else 0
    if remaining is not None:
        affordable = max(1, int(remaining / (2 * micro_tokens * cfg.accum_steps)))  # 2: forward and backward.
        planned = min(planned, affordable) if limit is not None else affordable
    result.planned_steps = planned

    parameters = delta.fresh_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(handle.device.type) if handle.fp16 else None
    with delta.banks("night"), handle.training_mode(cfg.gradient_checkpointing):
        try:
            for step in range(planned):
                lr = cfg.lr * 0.5 * (1.0 + math.cos(math.pi * step / max(planned, 1)))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                finite = True
                totals = {"hard": 0.0, "kl": 0.0, "anchor": 0.0}
                taught: set[str] = set()
                for _ in range(cfg.accum_steps):
                    hard_rows = hard_pool.draw(cfg.hard_batch)
                    kl_rows = kl_pool.draw(cfg.kl_batch)
                    loss, parts = compute_loss(handle, hard_rows, kl_rows, anchor_pool.draw(cfg.anchor_batch), cfg)
                    if not torch.isfinite(loss):
                        finite = False
                        break
                    scaled = loss / cfg.accum_steps
                    (scaler.scale(scaled) if scaler is not None else scaled).backward()
                    for key, value in parts.items():
                        totals[key] += value / cfg.accum_steps
                    for row in list(hard_rows) + list(kl_rows):
                        taught.update(row.episode_ids)
                if not finite:
                    result.nonfinite_steps += 1
                    continue
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
                if scaler is not None:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < scale_before:  # The scaler found inf gradients and skipped the step.
                        result.nonfinite_steps += 1
                        continue
                else:
                    optimizer.step()
                delta.bump_version()
                result.steps += 1
                result.trained_episode_ids |= taught
                result.loss_hard, result.loss_kl, result.loss_anchor = totals["hard"], totals["kl"], totals["anchor"]
        except LedgerExhausted:
            result.exhausted = True
    optimizer.zero_grad(set_to_none=True)
    return result
