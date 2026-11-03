"""Compute ledger: a per-night token allowance split across stages (invariant 4).

A token is one non-padding input token through a forward pass, or through a backward pass.
Charges are checked *before* the work is done, so spend can never exceed the allowance.

Each active stage gets a cap, so one stage cannot starve the rest. When a stage closes, its unused
tokens roll forward to the stages still open, so an arm is never left with budget it cannot use.
Evaluation runs inside `exempt()` and is tracked separately.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping, Optional


class LedgerExhausted(RuntimeError):
    """Raised before any work that would overrun the current stage's cap."""

    def __init__(self, stage: str, requested: int, remaining: int) -> None:
        super().__init__(f"stage {stage!r}: requested {requested} tokens, {remaining} remain")
        self.stage = stage
        self.requested = requested
        self.remaining = remaining


class ComputeLedger:
    """Unmetered until `begin_night`; afterwards every charge needs a stage or `exempt()`."""

    def __init__(self) -> None:
        self.allowance: Optional[int] = None
        self.caps: dict[str, int] = {}
        self.forward: dict[str, int] = {}
        self.backward: dict[str, int] = {}
        self.exhausted: list[str] = []
        self.exempt_forward = 0
        self.exempt_backward = 0
        self.unmetered_forward = 0
        self.unmetered_backward = 0
        self._fractions: dict[str, float] = {}
        self._closed: set[str] = set()
        self._stage: Optional[str] = None
        self._limit: Optional[int] = None
        self._exempt_depth = 0

    def begin_night(self, allowance: int, fractions: Mapping[str, float]) -> None:
        """Start a night. `fractions` holds the active stages only; it is renormalised to sum to 1."""
        if allowance < 0:
            raise ValueError("allowance must be non-negative")
        total = float(sum(fractions.values()))
        if any(f < 0 for f in fractions.values()) or (fractions and total <= 0):
            raise ValueError("stage fractions must be non-negative with a positive sum")
        self.allowance = int(allowance)
        self._fractions = dict(fractions)
        self.caps = {s: int(allowance * f / total) for s, f in fractions.items()}
        if self.caps:
            # Flooring loses a few tokens; give them to the largest stage so that caps sum exactly.
            largest = max(fractions, key=lambda s: fractions[s])
            self.caps[largest] += self.allowance - sum(self.caps.values())
        self.forward = {s: 0 for s in fractions}
        self.backward = {s: 0 for s in fractions}
        self.exhausted = []
        self._closed = set()
        self._stage = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Charge to `name` inside the block. Re-entrant; restores the previous stage on exit."""
        if self.allowance is not None and name not in self.caps:
            raise KeyError(f"stage {name!r} is not active tonight")
        previous = self._stage
        self._stage = name
        try:
            yield
        finally:
            self._stage = previous

    @contextmanager
    def exempt(self) -> Iterator[None]:
        """Evaluation: charges are counted separately and never raise."""
        self._exempt_depth += 1
        try:
            yield
        finally:
            self._exempt_depth -= 1

    @contextmanager
    def limit(self, tokens: int) -> Iterator[None]:
        """At most `tokens` more in the current stage inside the block, so one step cannot eat the stage's rest."""
        previous = self._limit
        stage = self._stage
        if stage is None:
            raise RuntimeError("limit needs a current stage")
        self._limit = self.spent(stage) + max(0, int(tokens))
        try:
            yield
        finally:
            self._limit = previous

    def transfer(self, source: str, target: str, tokens: int) -> None:
        """Move unspent allowance between open stages. The night's total does not change."""
        if tokens < 0 or tokens > self.caps[source] - self.spent(source):
            raise ValueError(f"cannot move {tokens} tokens out of stage {source!r}")
        if source in self._closed or target in self._closed:
            raise ValueError("cannot move tokens to or from a closed stage")
        self.caps[source] -= tokens
        self.caps[target] += tokens

    def restore_exempt(self, forward: int, backward: int) -> None:
        """Set the exempt counters: from a checkpoint, or back to before per-process setup work."""
        self.exempt_forward = int(forward)
        self.exempt_backward = int(backward)

    def close_stage(self, name: str) -> None:
        """No more spend in `name`. Its unused tokens go to open stages, in proportion to their fractions."""
        if name in self._closed:
            return
        unused = self.caps[name] - self.spent(name)
        self.caps[name] = self.spent(name)
        self._closed.add(name)
        still_open = [s for s in self.caps if s not in self._closed]
        if not still_open or unused == 0:
            return
        total = sum(self._fractions[s] for s in still_open)
        if total <= 0:
            self.caps[still_open[-1]] += unused
            return
        given = 0
        for s in still_open:
            share = int(unused * self._fractions[s] / total)
            self.caps[s] += share
            given += share
        self.caps[still_open[-1]] += unused - given

    def charge(self, forward: int = 0, backward: int = 0) -> None:
        """Record tokens, or raise LedgerExhausted without recording anything."""
        if forward < 0 or backward < 0:
            raise ValueError("token counts must be non-negative")
        if self._exempt_depth:
            self.exempt_forward += forward
            self.exempt_backward += backward
            return
        if self.allowance is None:
            self.unmetered_forward += forward
            self.unmetered_backward += backward
            return
        stage = self._stage
        if stage is None:
            raise RuntimeError("metered charge outside a stage; use ledger.stage(...) or ledger.exempt()")
        requested = forward + backward
        left = self.caps[stage] - self.spent(stage)
        if self._limit is not None:
            left = min(left, self._limit - self.spent(stage))
        if requested > left:
            if stage not in self.exhausted:
                self.exhausted.append(stage)
            raise LedgerExhausted(stage, requested, left)
        self.forward[stage] += forward
        self.backward[stage] += backward

    def spent(self, stage: Optional[str] = None) -> int:
        """Tokens spent in one stage, or in the whole night."""
        if stage is not None:
            return self.forward[stage] + self.backward[stage]
        return sum(self.forward.values()) + sum(self.backward.values())

    def remaining(self, stage: Optional[str] = None) -> Optional[int]:
        """Tokens left in `stage` (default: the current one). None when unmetered."""
        if self.allowance is None:
            return None
        name = stage if stage is not None else self._stage
        if name is None:
            return self.allowance - self.spent()
        return self.caps[name] - self.spent(name)

    def summary(self) -> dict:
        """JSON-able record of the night so far."""
        return {
            "allowance": self.allowance,
            "spent": self.spent(),
            "unspent": (self.allowance - self.spent()) if self.allowance is not None else None,
            "caps": dict(self.caps),
            "forward": dict(self.forward),
            "backward": dict(self.backward),
            "exhausted": list(self.exhausted),
            "exempt_forward": self.exempt_forward,
            "exempt_backward": self.exempt_backward,
        }
