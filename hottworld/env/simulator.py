"""Simulated HoTT proof environment.

Does NOT connect to a real proof assistant. Instead uses the RuleVerifier
to simulate proof state transitions. Suitable for:
- Training world models
- Evaluating search algorithms
- Generating synthetic training data

The simulated environment supports:
- Path composition problems
- Transport simplification
- Naturality square closure
- Simple equivalence proofs
- Coherence obligation tracking
"""

from typing import Any, Dict, List, Optional

from hottworld.env.base import ProofEnvironment, EnvStep
from hottworld.state.encoder import ProofState, ProofGoal, ProofContext
from hottworld.search.verifier import RuleVerifier
from hottworld.model.reward import CoherenceReward


class SimulatedHoTTEnv(ProofEnvironment):
    """Simulated proof environment for HoTT/category theory problems."""

    def __init__(self, seed: Optional[int] = None, max_depth: int = 100):
        self.verifier = RuleVerifier(max_depth=max_depth)
        self.reward_fn = CoherenceReward()
        self._state: Optional[ProofState] = None
        self._initial_state: Optional[ProofState] = None
        if seed is not None:
            import random
            random.seed(seed)

    def reset(self, problem: Dict[str, Any]) -> ProofState:
        goals = []
        for gdata in problem.get("goals", []):
            goals.append(ProofGoal(
                name=gdata.get("name", f"goal_{len(goals)}"),
                type_expr=gdata.get("type_expr", "? : ?"),
                context=gdata.get("context", []),
                depth=gdata.get("depth", 0),
            ))

        ctx_data = problem.get("context", {})
        context = ProofContext(
            hypotheses=ctx_data.get("hypotheses", {}),
            definitions=ctx_data.get("definitions", {}),
            lemmas=ctx_data.get("lemmas", {}),
            constraints=ctx_data.get("constraints", []),
        )

        self._state = ProofState(
            goals=goals,
            context=context,
            history=problem.get("history", []),
            step=0,
        )
        self._initial_state = ProofState(
            goals=[ProofGoal(name=g.name, type_expr=g.type_expr, context=g.context, depth=g.depth) for g in goals],
            context=ProofContext(
                hypotheses=dict(context.hypotheses),
                definitions=dict(context.definitions),
                lemmas=dict(context.lemmas),
                constraints=list(context.constraints),
            ),
        )
        return self._state

    def step(self, action: str, args: List[str] = None) -> EnvStep:
        if args is None:
            args = []

        prev_state = self._state
        valid, next_state, message = self.verifier.verify(
            self._state, action, args,
        )

        if not valid:
            self._state = prev_state
            return EnvStep(
                state=prev_state,
                reward=self.reward_fn.r_illegal,
                done=False,
                info={"message": message, "valid": False},
            )

        self._state = next_state
        done = next_state.is_done
        reward = self.reward_fn.compute(prev_state, action, next_state, is_done=done)

        return EnvStep(
            state=next_state,
            reward=reward,
            done=done,
            info={"message": message, "valid": True, "step": next_state.step},
        )

    def observe(self) -> ProofState:
        return self._state

    def close(self):
        self._state = None
        self._initial_state = None

    def available_actions(self) -> List[str]:
        """Return list of applicable tactic names for current state."""
        if self._state is None or self._state.is_done:
            return []

        all_actions = [
            "intro", "reflexivity", "apply", "exact", "compose",
            "rewrite", "transport", "construct_equivalence",
            "use_segal", "use_rezk", "fill_horn", "reverse",
        ]
        applicable = []
        for action in all_actions:
            valid, _, _ = self.verifier.verify(self._state, action, [])
            if valid:
                applicable.append(action)
            else:
                self._state.step -= 1
        return applicable
