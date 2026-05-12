"""Proof state encoder: converts proof state to tensor representation.

Encodes:
- goals: list of outstanding proof obligations
- context: current hypotheses and definitions
- higher cells: objects, morphisms, 2-cells present in context
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class ProofGoal:
    """A single proof goal."""
    name: str
    type_expr: str
    context: List[str] = field(default_factory=list)
    depth: int = 0


@dataclass
class ProofContext:
    """The ambient context: hypotheses, definitions, lemmas."""
    hypotheses: Dict[str, str] = field(default_factory=dict)
    definitions: Dict[str, str] = field(default_factory=dict)
    lemmas: Dict[str, str] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)


@dataclass
class ProofState:
    """Complete proof state for HoTT/Category Theory."""
    goals: List[ProofGoal] = field(default_factory=list)
    context: ProofContext = field(default_factory=dict)  # type: ignore[arg-type]
    history: List[str] = field(default_factory=list)
    step: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.context, dict) or not isinstance(self.context, ProofContext):
            self.context = ProofContext()

    @property
    def num_goals(self) -> int:
        return len(self.goals)

    @property
    def is_done(self) -> bool:
        return len(self.goals) == 0


class StateEncoder:
    """Encodes a ProofState into a fixed-dimension tensor.

    Features:
    - Goal embeddings (type structure, depth)
    - Context embeddings (hypotheses, definitions)
    - History encoding (last N actions)
    - Global features (num_goals, num_hypos, etc.)
    """

    def __init__(self, state_dim: int = 64, max_goals: int = 10):
        self.state_dim = state_dim
        self.max_goals = max_goals
        self._goal_dim = state_dim // 4
        self._ctx_dim = state_dim // 4
        self._hist_dim = state_dim // 4
        self._global_dim = state_dim - 3 * (state_dim // 4)

    def encode(self, state: ProofState) -> torch.Tensor:
        goal_emb = self._encode_goals(state.goals)
        ctx_emb = self._encode_context(state.context)
        hist_emb = self._encode_history(state.history)
        global_emb = self._encode_global(state)

        combined = torch.cat([goal_emb, ctx_emb, hist_emb, global_emb])
        return combined

    def _encode_goals(self, goals: List[ProofGoal]) -> torch.Tensor:
        if not goals:
            return torch.zeros(self._goal_dim)
        encodings = []
        for g in goals[:self.max_goals]:
            type_hash = hash(g.type_expr) % 1000 / 1000.0
            depth_norm = min(g.depth, 50) / 50.0
            ctx_size = min(len(g.context), 100) / 100.0
            gdim = self._goal_dim // 3
            vec = torch.zeros(gdim)
            vec[0] = type_hash
            vec[1 % gdim] = depth_norm
            vec[2 % gdim] = ctx_size
            encodings.append(vec)
        encodings_t = torch.stack(encodings)
        pooled = encodings_t.mean(dim=0)
        if len(pooled) < self._goal_dim:
            pooled = torch.cat([pooled, torch.zeros(self._goal_dim - len(pooled))])
        return pooled[:self._goal_dim]

    def _encode_context(self, ctx: ProofContext) -> torch.Tensor:
        n_hypos = min(len(ctx.hypotheses), 200) / 200.0
        n_defs = min(len(ctx.definitions), 100) / 100.0
        n_lemmas = min(len(ctx.lemmas), 100) / 100.0
        n_constraints = min(len(ctx.constraints), 50) / 50.0
        vec = torch.tensor([n_hypos, n_defs, n_lemmas, n_constraints],
                           dtype=torch.float32)
        if len(vec) < self._ctx_dim:
            vec = torch.cat([vec, torch.zeros(self._ctx_dim - len(vec))])
        return vec[:self._ctx_dim]

    def _encode_history(self, history: List[str]) -> torch.Tensor:
        if not history:
            return torch.zeros(self._hist_dim)
        hist = history[-10:]
        vec = torch.zeros(self._hist_dim)
        for i, h in enumerate(hist):
            action_hash = hash(h) % 100 / 100.0
            idx = i % self._hist_dim
            vec[idx] = action_hash
        return vec

    def _encode_global(self, state: ProofState) -> torch.Tensor:
        ngoals = min(state.num_goals, 50) / 50.0
        step_norm = min(state.step, 200) / 200.0
        vec = torch.tensor([ngoals, step_norm], dtype=torch.float32)
        if len(vec) < self._global_dim:
            vec = torch.cat([vec, torch.zeros(self._global_dim - len(vec))])
        return vec[:self._global_dim]
