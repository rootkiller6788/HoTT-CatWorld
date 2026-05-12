"""Tactic definitions for HoTT/higher category proof search.

Actions include:
- compose: compose two morphisms/paths
- rewrite: rewrite using a path/equality
- transport: transport along a path
- apply_univalence: turn an equivalence into a path
- apply_yoneda: use Yoneda lemma
- fill_horn: fill a horn to get a composite
- construct_equivalence: build an equivalence structure
- use_segal: apply Segal composition
- use_rezk: apply Rezk completeness
- intro: introduce a hypothesis
- apply: apply a lemma to a goal
- reflexivity: close a goal by reflexivity
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch


class TacticKind(Enum):
    COMPOSE = "compose"
    REWRITE = "rewrite"
    TRANSPORT = "transport"
    APPLY_UNIVALENCE = "apply_univalence"
    APPLY_YONEDA = "apply_yoneda"
    FILL_HORN = "fill_horn"
    CONSTRUCT_EQUIVALENCE = "construct_equivalence"
    USE_SEGAL = "use_segal"
    USE_REZK = "use_rezk"
    INTRO = "intro"
    APPLY = "apply"
    REFLEXIVITY = "reflexivity"
    EXACT = "exact"
    REVERSE = "reverse"


@dataclass
class Tactic:
    """A single proof tactic action."""
    kind: TacticKind
    args: List[Any] = field(default_factory=list)
    description: str = ""
    source: Optional[str] = None
    target: Optional[str] = None

    def to_vector(self, action_dim: int = 16) -> torch.Tensor:
        """Encode tactic as a fixed-dimension vector."""
        kind_idx = list(TacticKind).index(self.kind)
        vec = torch.zeros(action_dim)
        vec[0] = float(kind_idx)
        vec[1] = len(self.args) / 10.0 if self.args else 0.0
        return vec

    @staticmethod
    def from_vector(vec: torch.Tensor) -> "Tactic":
        kind_idx = round(vec[0].item())
        kind_idx = max(0, min(kind_idx, len(TacticKind) - 1))
        return Tactic(kind=list(TacticKind)[kind_idx])


class TacticLibrary:
    """Library of available tactics with applicability predicates."""

    def __init__(self):
        self._tactics: List[TacticKind] = list(TacticKind)

    def all_actions(self) -> List[Tactic]:
        return [Tactic(kind=k) for k in self._tactics]

    def applicable(self, tactic: Tactic, state: Any) -> bool:
        """Check if a tactic is applicable to the given proof state."""
        goals = getattr(state, "goals", [])
        if not goals:
            return False

        always_applicable = {
            TacticKind.INTRO,
            TacticKind.REFLEXIVITY,
            TacticKind.APPLY,
            TacticKind.EXACT,
        }
        if tactic.kind in always_applicable:
            return True

        needs_equivalence = {
            TacticKind.APPLY_UNIVALENCE,
            TacticKind.CONSTRUCT_EQUIVALENCE,
        }
        if tactic.kind in needs_equivalence:
            ctx = getattr(state, "context", {})
            defs = getattr(ctx, "definitions", {}) if ctx else {}
            return any("equiv" in k.lower() or "iso" in k.lower() for k in defs)

        needs_path = {
            TacticKind.COMPOSE,
            TacticKind.REWRITE,
            TacticKind.TRANSPORT,
        }
        if tactic.kind in needs_path:
            return len(goals) >= 1

        return True

    def action_vector(self, tactic: Tactic, state: Any) -> torch.Tensor:
        """Get action vector with mask for applicable actions."""
        app = 1.0 if self.applicable(tactic, state) else 0.0
        vec = tactic.to_vector()
        vec[-1] = app
        return vec
