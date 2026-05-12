"""Abstract proof environment interface.

Defines the common protocol for all proof assistant backends:
- Cubical Agda, Rzk, and the simulated environment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hottworld.state.encoder import ProofState


@dataclass
class EnvStep:
    """Result of a single environment step."""
    state: ProofState
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ProofEnvironment(ABC):
    """Abstract proof environment following the Gym-like interface.

    Each step applies a tactic (action) and returns the new proof state,
    reward, and termination flag.
    """

    @abstractmethod
    def reset(self, problem: Dict[str, Any]) -> ProofState:
        """Initialize a new proof problem.

        Args:
            problem: dict with 'goals', 'context', 'type_expr', etc.

        Returns:
            Initial proof state.
        """
        ...

    @abstractmethod
    def step(self, action: str, args: List[str] = None) -> EnvStep:
        """Apply a tactic and step the environment."""
        ...

    @abstractmethod
    def observe(self) -> ProofState:
        """Return current proof state without modifying it."""
        ...

    @abstractmethod
    def close(self):
        """Clean up resources."""
        ...
