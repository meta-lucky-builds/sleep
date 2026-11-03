"""Tiny in-process models, tokenizer and world for tests. No network, no downloads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from llmsleep.config import (
    BudgetConfig,
    CompactConfig,
    ModelConfig,
    NightConfig,
    QueryConfig,
    SelectConfig,
    SignalConfig,
    TeacherConfig,
    TrainConfig,
)
from llmsleep.memory import Episode
from llmsleep.phases.night import State

PAD = 0
EOS = 1
PRINTABLE = "".join(chr(c) for c in range(32, 127)) + "\n"


class TinyTokenizer:
    """Char-level. pad=0, eos=1, then the alphabet. Unknown characters map to the alphabet's first."""

    def __init__(self, alphabet: str = PRINTABLE) -> None:
        self.alphabet = alphabet
        self.pad_token_id = PAD
        self.eos_token_id = EOS
        self.bos_token_id = None
        self.vocab_size = len(alphabet) + 2
        self._to_id = {ch: i + 2 for i, ch in enumerate(alphabet)}

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        return {"input_ids": [self._to_id.get(ch, 2) for ch in text]}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for token in ids:
            if token >= 2:
                chars.append(self.alphabet[token - 2])
            elif not skip_special_tokens:
                chars.append("<pad>" if token == PAD else "<eos>")
        return "".join(chars)


@dataclass
class _Output:
    logits: torch.Tensor


class BigramStub(nn.Module):
    """Logits depend only on the current token: logits[b, t] = table[input_ids[b, t]]. No cache.

    With a known table, log-probs can be worked out by hand, and padding cannot change them.
    """

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = nn.Parameter(table.clone(), requires_grad=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        return _Output(self.table[input_ids])


class ContextCategoricalStub(nn.Module):
    """Every position's logits are fixed by the row's first real token: tokens are iid given that token.

    So a continuation's law is known exactly, and the sequence KL has a closed form.
    """

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = nn.Parameter(table.clone(), requires_grad=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch, width = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        first = attention_mask.argmax(dim=1)  # Index of the first real token, under left or right padding.
        heads = input_ids[torch.arange(batch), first]  # (B,)
        return _Output(self.table[heads].unsqueeze(1).expand(batch, width, -1))


def _with_seed(seed: int, build):
    """Build under a fixed seed without disturbing the global RNG the code under test must not depend on."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build()


def tiny_gpt2(vocab_size: int, seed: int = 0) -> nn.Module:
    """2 layers, HF Conv1D linears. Dropout is zero."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=EOS,
        eos_token_id=EOS,
        pad_token_id=PAD,
        attn_implementation="eager",
    )
    return _with_seed(seed, lambda: GPT2LMHeadModel(config).eval())


def tiny_qwen2(vocab_size: int, seed: int = 0) -> nn.Module:
    """2 layers, nn.Linear attention and MLP."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        attention_dropout=0.0,
        use_sliding_window=False,
        tie_word_embeddings=False,
        bos_token_id=EOS,
        eos_token_id=EOS,
        pad_token_id=PAD,
        attn_implementation="eager",
    )
    return _with_seed(seed, lambda: Qwen2ForCausalLM(config).eval())


class TinyWorld:
    """Satisfies WorldLike. Two episodes per night; episodes share entity Zor across a night, so pairs exist."""

    def __init__(self, nights: int = 3) -> None:
        self.nights = nights
        self._distractors = [
            Episode(f"filler-{i}", f"The harbour tide tables list {i} ferries each week in spring.", -1)
            for i in range(6)
        ]

    def episodes(self, night: int) -> list[Episode]:
        out = []
        for i in range(2):
            name = f"Mira{night}{i}"
            out.append(
                Episode(
                    id=f"n{night}-e{i}",
                    text=f"{name} of Zor{night} keeps {night + i + 2} blue kites.",
                    night=night,
                    entities=[name, f"Zor{night}"],
                    meta={
                        "statements": [f"{night + i + 2} blue kites belong to {name}."],
                        "qa": [[f"How many kites does {name} keep?", str(night + i + 2)]],
                    },
                )
            )
        return out

    def template_queries(self, episode: Episode) -> list[str]:
        name = episode.entities[0]
        return [
            f"How many kites has {name}?",
            f"What does {name} keep?",
            f"Where is {name} from?",
            f"Who is {name}?",
        ]

    def pair_template_queries(self, a: Episode, b: Episode) -> list[str]:
        return [f"Together, how many kites have {a.entities[0]} and {b.entities[0]}?"]

    def distractor_pool(self) -> list[Episode]:
        return self._distractors


def small_config(arm, night_tokens: int = 80_000) -> NightConfig:
    """Budgets and sample counts small enough for a tiny model on CPU."""
    return NightConfig(
        arm=arm,
        model=ModelConfig(rank=4, alpha=4.0, batch_size=8),
        budget=BudgetConfig(night_tokens=night_tokens),
        signals=SignalConfig(k=2, max_new_tokens=4, chunk=4),
        queries=QueryConfig(per_episode=4, model_samples=1, model_max_new_tokens=8, max_pairs=2),
        select=SelectConfig(n_null=8, min_null=4),
        teacher=TeacherConfig(samples=1, max_new_tokens=4, top_k=8, self_samples=4, self_max_new_tokens=4),
        train=TrainConfig(lr=1e-3, hard_batch=2, kl_batch=2, anchor_batch=2, max_steps=4),
        compact=CompactConfig(threshold=0.01, max_rank=6, repair_steps=2),
    )


def fresh_gpt2_state(cfg: NightConfig, world, tokenizer: TinyTokenizer, world_seed: int = 7) -> State:
    """A new tiny model and State, always from the same initial weights."""
    return State.create(tiny_gpt2(tokenizer.vocab_size), tokenizer, world, world_seed, cfg)
