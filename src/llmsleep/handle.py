"""Model access: prompt format, token alignment, scoring and sampling. All of it charged to the ledger."""
from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .delta import BankSpec, DeltaModel
from .ledger import ComputeLedger

TokenLike = Union[str, Sequence[int]]


@dataclass(frozen=True)
class PromptFormat:
    """One wrapper for open, distractor and closed conditions, so only the context's content differs."""

    template: str = "Context: {context}\nQuestion: {query}\nAnswer:"
    empty_context: str = "(none)"
    answer_prefix: str = " "

    def render(self, query: str, context: Optional[str] = None) -> str:
        return self.template.format(context=context if context else self.empty_context, query=query)


@dataclass
class TokenBatch:
    """All (B, T). `answer_mask` is True exactly at continuation tokens, in input positions."""

    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    answer_mask: Tensor

    def to(self, device: torch.device) -> "TokenBatch":
        return TokenBatch(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.position_ids.to(device),
            self.answer_mask.to(device),
        )


def answer_token_mask(prompt_lens: Sequence[int], cont_lens: Sequence[int], side: str) -> Tensor:
    """(B, T) bool, True at continuation tokens. Right padding: [prompt cont pad]. Left: [pad prompt cont]."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {side!r}")
    if any(p < 1 for p in prompt_lens):
        # The first token of a row has no preceding logits, so a continuation there cannot be scored.
        raise ValueError("every prompt needs at least one token")
    total = [p + c for p, c in zip(prompt_lens, cont_lens)]
    width = max(total)
    mask = torch.zeros(len(total), width, dtype=torch.bool)
    for row, (p, length) in enumerate(zip(prompt_lens, total)):
        start = 0 if side == "right" else width - length
        mask[row, start + p : start + length] = True
    return mask


def build_batch(
    prompts: Sequence[Sequence[int]], continuations: Sequence[Sequence[int]], pad_id: int, side: str
) -> TokenBatch:
    """Pad to a batch. Right for training and scoring, left for generation."""
    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations differ in length")
    answer_mask = answer_token_mask([len(p) for p in prompts], [len(c) for c in continuations], side)
    batch, width = answer_mask.shape
    input_ids = torch.full((batch, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch, width, dtype=torch.long)
    for row, (prompt, cont) in enumerate(zip(prompts, continuations)):
        sequence = list(prompt) + list(cont)
        start = 0 if side == "right" else width - len(sequence)
        input_ids[row, start : start + len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, start : start + len(sequence)] = 1
    return TokenBatch(input_ids, attention_mask, positions_from_mask(attention_mask), answer_mask)


def positions_from_mask(attention_mask: Tensor) -> Tensor:
    """Position of each token among real tokens. Must be explicit: left padding shifts absolute indices."""
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


def gather_answer_logits(logits: Tensor, answer_mask: Tensor) -> Tensor:
    """(N, V) logits predicting each continuation token, row-major, N = answer_mask.sum().

    Logits at t predict the token at t+1, so position t is selected when the token at t+1 is an answer
    token. This is what makes the mask correct under either padding side.
    """
    return logits[:, :-1][answer_mask[:, 1:]]


def gather_answer_targets(input_ids: Tensor, answer_mask: Tensor) -> Tensor:
    """(N,) the continuation tokens, in the same order as `gather_answer_logits`."""
    return input_ids[:, 1:][answer_mask[:, 1:]]


def continuation_logprobs(logits: Tensor, input_ids: Tensor, answer_mask: Tensor) -> Tensor:
    """(B,) sum of log p(continuation tokens | everything before them)."""
    log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)  # (B, T-1, V)
    token_lp = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    # masked_fill, not multiply: an unselected -inf times zero would be NaN.
    return token_lp.masked_fill(~answer_mask[:, 1:], 0.0).sum(dim=-1)


@dataclass
class Sample:
    """One sampled continuation. `logprob` is under the untempered model, EOS included if sampled.

    `finished` False means truncated at max_new_tokens: `token_ids` is then a prefix, not a whole answer.
    """

    token_ids: list[int]
    logprob: float
    finished: bool


class ModelHandle:
    """`model` is usually a DeltaModel; a plain module (a stub) also works, with no bank selection."""

    def __init__(
        self,
        model: Union[DeltaModel, nn.Module],
        tokenizer: Any,
        device: Union[str, torch.device] = "cpu",
        ledger: Optional[ComputeLedger] = None,
        prompt_format: PromptFormat = PromptFormat(),
        batch_size: int = 8,
        fp16: bool = False,
        eos_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.delta = model if isinstance(model, DeltaModel) else None
        self.net: nn.Module = model.model if isinstance(model, DeltaModel) else model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.ledger = ledger if ledger is not None else ComputeLedger()
        self.prompt_format = prompt_format
        self.batch_size = batch_size
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.eos_id: Optional[int] = tokenizer.eos_token_id
        self.eos_ids = set(eos_ids) if eos_ids is not None else ({self.eos_id} if self.eos_id is not None else set())
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_id
        if pad_id is None:
            raise ValueError("tokenizer has neither a pad nor an EOS token")
        self.pad_id: int = int(pad_id)
        self.net.to(self.device)
        self.net.eval()
        parameters = inspect.signature(self.net.forward).parameters
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        self._accepts_attention = "attention_mask" in parameters or has_var_kw
        self._accepts_positions = "position_ids" in parameters or has_var_kw
        self._accepts_return_dict = "return_dict" in parameters or has_var_kw
        # Some transformers versions route cache arguments through **kwargs, so the signature alone cannot
        # say. A config with `use_cache` means a real HF model. A stub with neither re-feeds the sequence.
        hf_config_has_cache = getattr(getattr(self.net, "config", None), "use_cache", None) is not None
        self._has_cache = "past_key_values" in parameters or (has_var_kw and hf_config_has_cache)

    # ---- text ---------------------------------------------------------------------------------------

    def format(self, query: str, context: Optional[str] = None) -> str:
        return self.prompt_format.render(query, context)

    def encode_prompt(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def encode_continuation(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode_answer(self, answer: str, add_eos: bool = True) -> list[int]:
        """Tokens of a gold answer as it should follow a prompt."""
        ids = self.encode_continuation(self.prompt_format.answer_prefix + answer.strip())
        return ids + [self.eos_id] if add_eos and self.eos_id is not None else ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def start_ids(self) -> list[int]:
        """A one-token prompt for unconditional text."""
        bos = getattr(self.tokenizer, "bos_token_id", None)
        return [int(bos if bos is not None else self.pad_id if self.eos_id is None else self.eos_id)]

    def _prompt_ids(self, prompt: TokenLike) -> list[int]:
        return self.encode_prompt(prompt) if isinstance(prompt, str) else list(prompt)

    def _cont_ids(self, continuation: TokenLike) -> list[int]:
        return self.encode_continuation(continuation) if isinstance(continuation, str) else list(continuation)

    # ---- banks --------------------------------------------------------------------------------------

    def banks(self, spec: BankSpec):
        """Bank selection context; a no-op without a DeltaModel."""
        return self.delta.banks(spec) if self.delta is not None else nullcontext()

    def banks_key(self) -> Any:
        return self.delta.banks_key() if self.delta is not None else None

    # ---- forward ------------------------------------------------------------------------------------

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16) if self.fp16 else nullcontext()

    def _forward(
        self, input_ids: Tensor, attention_mask: Tensor, position_ids: Tensor, past: Any = None, use_cache: bool = False
    ) -> Any:
        """One forward. `past` is opaque: a DynamicCache or a legacy tuple goes back exactly as the model
        returned it, and is only ever compared with None (an empty cache may be falsy). A DynamicCache
        updates in place, so each cache is used once, by the next step, and never kept."""
        kwargs: dict[str, Any] = {}
        if self._accepts_attention:
            kwargs["attention_mask"] = attention_mask
        if self._accepts_positions:
            kwargs["position_ids"] = position_ids
        if self._accepts_return_dict:
            kwargs["return_dict"] = True  # A config with return_dict=False would give tuples without .logits.
        if self._has_cache:
            kwargs["use_cache"] = use_cache
            if past is not None:
                kwargs["past_key_values"] = past
        with self.autocast():
            return self.net(input_ids=input_ids, **kwargs)

    @contextmanager
    def training_mode(self, gradient_checkpointing: bool = False) -> Iterator[None]:
        """train() is what enables HF checkpointing; dropout stays off, since it would draw on global RNG."""
        checkpointing = gradient_checkpointing and hasattr(self.net, "gradient_checkpointing_enable")
        if checkpointing:
            self.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.net.train()
        for module in self.net.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        try:
            yield
        finally:
            self.net.eval()
            if checkpointing:
                self.net.gradient_checkpointing_disable()

    def forward_train(self, batch: TokenBatch) -> Tensor:
        """(B, T, V) logits with gradients. Charges forward and backward up front, padding-free."""
        tokens = int(batch.attention_mask.sum())
        self.ledger.charge(forward=tokens, backward=tokens)
        b = batch.to(self.device)
        return self._forward(b.input_ids, b.attention_mask, b.position_ids).logits

    # ---- scoring ------------------------------------------------------------------------------------

    def answer_token_mask(self, prompts: Sequence[TokenLike], continuations: Sequence[TokenLike], side: str) -> Tensor:
        """Mask of continuation positions, for the batch `build_batch` would make from these inputs."""
        return answer_token_mask(
            [len(self._prompt_ids(p)) for p in prompts], [len(self._cont_ids(c)) for c in continuations], side
        )

    @torch.no_grad()
    def sequence_logprob(self, prompts: Sequence[TokenLike], continuations: Sequence[TokenLike]) -> Tensor:
        """(N,) float32 on CPU: sum of log p(continuation | prompt) under the active banks."""
        prompt_ids = [self._prompt_ids(p) for p in prompts]
        cont_ids = [self._cont_ids(c) for c in continuations]
        results = []
        for start in range(0, len(prompt_ids), self.batch_size):
            batch = build_batch(
                prompt_ids[start : start + self.batch_size],
                cont_ids[start : start + self.batch_size],
                self.pad_id,
                "right",
            ).to(self.device)
            self.ledger.charge(forward=int(batch.attention_mask.sum()))
            logits = self._forward(batch.input_ids, batch.attention_mask, batch.position_ids).logits
            results.append(continuation_logprobs(logits, batch.input_ids, batch.answer_mask).cpu())
        return torch.cat(results) if results else torch.zeros(0)

    def iter_answer_logits(
        self, prompt_ids: Sequence[Sequence[int]], cont_ids: Sequence[Sequence[int]]
    ) -> Iterator[tuple[list[int], Tensor]]:
        """Yield (rows, (N, V) float32 logits) per micro-batch, split among `rows` by continuation length."""
        for start in range(0, len(prompt_ids), self.batch_size):
            stop = min(start + self.batch_size, len(prompt_ids))
            batch = build_batch(prompt_ids[start:stop], cont_ids[start:stop], self.pad_id, "right").to(self.device)
            self.ledger.charge(forward=int(batch.attention_mask.sum()))
            with torch.no_grad():
                logits = self._forward(batch.input_ids, batch.attention_mask, batch.position_ids).logits
            yield list(range(start, stop)), gather_answer_logits(logits, batch.answer_mask).float()

    # ---- sampling -----------------------------------------------------------------------------------

    def sample(
        self,
        prompts: Sequence[TokenLike],
        n: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> list[list[Sample]]:
        """n samples per prompt, under the active banks. Replaces HF generate, which uses global RNG.

        `generator` must be a CPU generator (see config.make_generator). temperature=0 is greedy.
        """
        if generator is None:
            raise ValueError("sampling needs an explicit generator; the global RNG is never used")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        prompt_ids = [self._prompt_ids(p) for p in prompts]
        rows = [i for i in range(len(prompt_ids)) for _ in range(n)]
        out: list[list[Sample]] = [[] for _ in prompt_ids]
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start : start + self.batch_size]
            samples = self._sample_batch([prompt_ids[i] for i in chunk], max_new_tokens, temperature, generator)
            for row, sample in zip(chunk, samples):
                out[row].append(sample)
        return out

    @torch.no_grad()
    def _sample_batch(
        self, prompts: list[list[int]], max_new_tokens: int, temperature: float, generator: torch.Generator
    ) -> list[Sample]:
        batch = build_batch(prompts, [[] for _ in prompts], self.pad_id, "left").to(self.device)
        input_ids, attention = batch.input_ids, batch.attention_mask
        size = input_ids.size(0)
        tokens: list[list[int]] = [[] for _ in range(size)]
        logprobs = [0.0] * size
        finished = torch.zeros(size, dtype=torch.bool)  # CPU, like everything the generator touches.

        self.ledger.charge(forward=int(attention.sum()))
        output = self._forward(input_ids, attention, batch.position_ids, use_cache=True)
        for step in range(max_new_tokens):
            # Left padding puts every row's last real token in the final column, so -1 is right for all.
            last = output.logits[:, -1, :].float()  # (B, V)
            log_probs = F.log_softmax(last, dim=-1).cpu()
            if temperature == 0:
                next_ids = log_probs.argmax(dim=-1)
            else:
                probs = log_probs.exp() if temperature == 1.0 else F.softmax(last / temperature, dim=-1).cpu()
                # Finished rows draw pad deterministically; their logits are ignored and never reach multinomial.
                probs[finished] = 0.0
                probs[finished, self.pad_id] = 1.0
                next_ids = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            next_ids[finished] = self.pad_id
            for row in range(size):
                if finished[row]:
                    continue
                token = int(next_ids[row])
                tokens[row].append(token)
                logprobs[row] += float(log_probs[row, token])
                if token in self.eos_ids:
                    finished[row] = True
            if step == max_new_tokens - 1 or bool(finished.all()):
                break
            # Rows finished at or before this step are masked and not charged; their logits are unused.
            input_ids = torch.cat([input_ids, next_ids.to(self.device).unsqueeze(1)], dim=1)
            keep = (~finished).long().to(self.device).unsqueeze(1)
            attention = torch.cat([attention, keep], dim=1)
            positions = positions_from_mask(attention)
            past = getattr(output, "past_key_values", None) if self._has_cache else None
            if past is not None:
                self.ledger.charge(forward=int((~finished).sum()))
                # One new token per row, but the mask spans the whole sequence including the cache.
                output = self._forward(input_ids[:, -1:], attention, positions[:, -1:], past=past, use_cache=True)
            else:
                self.ledger.charge(forward=int(attention.sum()))
                output = self._forward(input_ids, attention, positions)
        return [Sample(tokens[i], logprobs[i], bool(finished[i])) for i in range(size)]
