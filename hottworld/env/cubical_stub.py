"""Cubical Agda proof assistant stub.

Future integration point for Cubical Agda.
Currently provides the interface contract and placeholder methods.

For real integration, this would:
1. Start an Agda process with cubical mode
2. Send proof obligations via I/O or LSP
3. Parse Agda output for proof state
4. Extract paths, hcomp, transp, Glue, univalence structures
"""

from typing import Any, Dict, List, Optional

from hottworld.env.base import ProofEnvironment, EnvStep
from hottworld.state.encoder import ProofState


class CubicalAgdaStub(ProofEnvironment):
    """Stub for Cubical Agda integration. Raises NotImplementedError.

    Full integration roadmap:
    1. Install Agda with cubical library
    2. Use Agda's Emacs/LSP protocol for interaction
    3. Parse .agda files for module structure
    4. Map Agda terms to our higher-categorical representation
    """

    def __init__(self, agda_path: Optional[str] = None):
        self.agda_path = agda_path or "agda"
        self._state: Optional[ProofState] = None
        self._connected = False
        self._raise_not_implemented()

    def _raise_not_implemented(self):
        raise NotImplementedError(
            "Cubical Agda integration is not yet implemented. "
            "Use SimulatedHoTTEnv for the MVP."
        )

    def reset(self, problem: Dict[str, Any]) -> ProofState:
        self._raise_not_implemented()

    def step(self, action: str, args: List[str] = None) -> EnvStep:
        self._raise_not_implemented()

    def observe(self) -> ProofState:
        self._raise_not_implemented()

    def close(self):
        pass
