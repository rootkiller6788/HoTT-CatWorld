"""Rzk proof assistant stub for synthetic ∞-categories.

Future integration point for Rzk/sHoTT.
Currently provides the interface contract and placeholder methods.

For real integration, this would:
1. Start an Rzk process
2. Interact via Rzk's command interface
3. Parse Segal types, Rezk types, Yoneda embeddings
4. Map simplical type structures to our representation
"""

from typing import Any, Dict, List, Optional

from hottworld.env.base import ProofEnvironment, EnvStep
from hottworld.state.encoder import ProofState


class RzkStub(ProofEnvironment):
    """Stub for Rzk integration. Raises NotImplementedError.

    Full integration roadmap:
    1. Install Rzk (rzk-lang.github.io/rzk)
    2. Use Rzk's interactive mode
    3. Parse .rzk files for Segal/Rezk type structures
    4. Map Rzk's directed interval and tope logic to our representation
    """

    def __init__(self, rzk_path: Optional[str] = None):
        self.rzk_path = rzk_path or "rzk"
        self._state: Optional[ProofState] = None
        self._connected = False

    def reset(self, problem: Dict[str, Any]) -> ProofState:
        raise NotImplementedError(
            "Rzk integration is not yet implemented. "
            "Use SimulatedHoTTEnv for the MVP."
        )

    def step(self, action: str, args: List[str] = None) -> EnvStep:
        raise NotImplementedError(
            "Rzk integration is not yet implemented."
        )

    def observe(self) -> ProofState:
        raise NotImplementedError(
            "Rzk integration is not yet implemented."
        )

    def close(self):
        pass
