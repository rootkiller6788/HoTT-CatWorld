"""Feature extraction from proof state for world model.

Computes structured features:
- Subgoal count and complexity
- Diagram closure degree
- Coherence obligations solved / total
- Transport complexity metric
- Lemma match score
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class StateFeatures:
    """Extracted features from a proof state used by world model and value function."""
    num_goals: int = 0
    num_hypos: int = 0
    num_defs: int = 0
    diagram_closure: float = 0.0
    coherence_solved: int = 0
    coherence_total: int = 1
    transport_depth: int = 0
    step_count: int = 0
    lemma_match_score: float = 0.0
    raw: dict = field(default_factory=dict)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.num_goals / 50.0,
            self.num_hypos / 200.0,
            self.num_defs / 100.0,
            self.diagram_closure,
            self.coherence_solved / max(self.coherence_total, 1),
            self.transport_depth / 20.0,
            self.step_count / 200.0,
            self.lemma_match_score,
        ], dtype=torch.float32)

    def coherence_ratio(self) -> float:
        return self.coherence_solved / max(self.coherence_total, 1)

    def is_terminal(self) -> bool:
        return self.num_goals == 0


def extract_features(
    goals: List[Any],
    context: Any,
    diagram_closure: float = 0.0,
    coherence_solved: int = 0,
    coherence_total: int = 1,
    transport_depth: int = 0,
    step_count: int = 0,
    lemma_match_score: float = 0.0,
) -> StateFeatures:
    """Extract features from proof state components."""
    num_hypos = 0
    num_defs = 0
    if hasattr(context, "hypotheses"):
        num_hypos = len(context.hypotheses)
    if hasattr(context, "definitions"):
        num_defs = len(context.definitions)
    elif isinstance(context, dict):
        num_hypos = len(context.get("hypotheses", {}))
        num_defs = len(context.get("definitions", {}))

    return StateFeatures(
        num_goals=len(goals),
        num_hypos=num_hypos,
        num_defs=num_defs,
        diagram_closure=diagram_closure,
        coherence_solved=coherence_solved,
        coherence_total=coherence_total,
        transport_depth=transport_depth,
        step_count=step_count,
        lemma_match_score=lemma_match_score,
    )
