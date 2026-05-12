"""Coherence-aware reward function for HoTT proof search.

Reward components:
- R_done: +10.0 for completing the proof
- R_goal: +1.0 per subgoal eliminated
- R_coherence: +0.5 per coherence obligation discharged
- R_diagram: +2.0 for closing a diagram
- R_transport: +0.3 per transport simplification
- R_step: -0.05 per step (encourages efficiency)
- R_illegal: -1.0 for invalid tactic application
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class CoherenceReward:
    """Reward function configuration and computation."""
    r_done: float = 10.0
    r_goal: float = 1.0
    r_coherence: float = 0.5
    r_diagram: float = 2.0
    r_transport: float = 0.3
    r_step_penalty: float = 0.05
    r_illegal: float = -1.0

    def compute(
        self,
        prev_state: Any,
        action: Any,
        next_state: Any,
        is_done: bool = False,
        is_illegal: bool = False,
        coherence_discharged: int = 0,
        diagrams_closed: int = 0,
    ) -> float:
        if is_illegal:
            return self.r_illegal

        reward = 0.0

        if is_done:
            reward += self.r_done

        prev_goals = getattr(prev_state, "num_goals", len(getattr(prev_state, "goals", [])))
        next_goals = getattr(next_state, "num_goals", len(getattr(next_state, "goals", [])))
        eliminated = prev_goals - next_goals
        if eliminated > 0:
            reward += self.r_goal * eliminated

        reward += self.r_coherence * coherence_discharged
        reward += self.r_diagram * diagrams_closed
        reward -= self.r_step_penalty

        return reward

    def compute_batch(
        self,
        prev_states: List[Any],
        actions: List[Any],
        next_states: List[Any],
        dones: List[bool],
        illegals: List[bool],
        coherence_discharged: List[int],
        diagrams_closed: List[int],
    ) -> torch.Tensor:
        rewards = []
        for prev_s, action, next_s, done, illegal, coh, diag in zip(
            prev_states, actions, next_states, dones, illegals,
            coherence_discharged, diagrams_closed,
        ):
            r = self.compute(prev_s, action, next_s, done, illegal, coh, diag)
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32)
