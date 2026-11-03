"""Sleep for language models: controlled, many-cycle consolidation. See docs/DESIGN.md."""
from .arms import ARMS, ArmConfig, get_arm
from .config import NightConfig, derive_seed, make_generator
from .delta import DeltaModel
from .handle import ModelHandle, PromptFormat
from .ledger import ComputeLedger, LedgerExhausted
from .memory import Episode, EpisodicStore
from .phases.night import NightReport, SleepPressure, State, run_night
from .phases.queries import WorldLike
from .signals import SignalCache, chd

__all__ = [
    "ARMS",
    "ArmConfig",
    "ComputeLedger",
    "DeltaModel",
    "Episode",
    "EpisodicStore",
    "LedgerExhausted",
    "ModelHandle",
    "NightConfig",
    "NightReport",
    "PromptFormat",
    "SignalCache",
    "SleepPressure",
    "State",
    "WorldLike",
    "chd",
    "derive_seed",
    "get_arm",
    "make_generator",
    "run_night",
]
