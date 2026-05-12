"""Structure World Model: predicts next proof state given current state and action.

P(s_{t+1} | s_t, a_t) - the core transition model for HoTT proof search.

Architecture: MLP with residual connections.
Input: [state_encoding | action_encoding] -> Output: next state encoding.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """Transition model P(s' | s, a) for HoTT proof states.

    Predicts the next proof state encoding given current state and action.
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)

        self.predictor = nn.Linear(hidden_dim, state_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next state encoding.

        Args:
            state: (batch, state_dim) current proof state encoding
            action: (batch, action_dim) tactic action encoding

        Returns:
            (batch, state_dim) predicted next state encoding
        """
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        delta = self.predictor(h)
        return state + delta

    def predict_single(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next state for a single (state, action) pair."""
        return self.forward(state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)

    @torch.no_grad()
    def rollout(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> list:
        """Roll out a sequence of actions from an initial state.

        Args:
            state: (state_dim,) initial state
            actions: (seq_len, action_dim) sequence of actions

        Returns:
            List of (state_dim,) predicted states including initial
        """
        states = [state]
        current = state.unsqueeze(0)
        for action in actions:
            current = self.forward(current, action.unsqueeze(0))
            states.append(current.squeeze(0))
        return states
