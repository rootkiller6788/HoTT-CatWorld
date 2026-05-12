"""Coherence-aware value function for proof states.

V(s) evaluates how "close to done" a proof state is:
- Subgoal reduction
- Diagram closure degree
- Coherence obligations solved
- Transport complexity decrease
- Reusable lemma match score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueFunc(nn.Module):
    """Value function V(s) estimating proof completion likelihood from state.

    Takes a state encoding and outputs a scalar value in [0, 1].
    Higher values indicate states closer to a completed proof.
    """

    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = state_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate state value.

        Args:
            state: (batch, state_dim) proof state encoding

        Returns:
            (batch, 1) value in [0, 1]
        """
        h = self.encoder(state)
        return torch.sigmoid(self.head(h))

    def evaluate_single(self, state: torch.Tensor) -> float:
        """Evaluate value for a single state."""
        return self.forward(state.unsqueeze(0)).item()
