"""Synthetic proof state data generator.

Generates training data for the world model by creating diverse
proof states and simulating transitions in the simulated environment.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hottworld.state.encoder import ProofState, ProofGoal, ProofContext
from hottworld.env.simulator import SimulatedHoTTEnv


@dataclass
class Trajectory:
    """A proof trajectory: sequence of (state, action, next_state, reward, done)."""
    problem: Dict[str, Any]
    steps: List[dict] = field(default_factory=list)

    def to_records(self) -> List[dict]:
        return [
            {
                "state": s["state"],
                "action": s["action"],
                "next_state": s["next_state"],
                "reward": s["reward"],
                "done": s["done"],
            }
            for s in self.steps
        ]


TEMPLATE_PROBLEMS = [
    {
        "name": "path_composition",
        "goals": [
            {
                "name": "comp_goal",
                "type_expr": "P(a) = P(c)",
                "context": ["p : a = b", "q : b = c"],
            }
        ],
        "context": {
            "hypotheses": {"p": "a = b", "q": "b = c"},
            "definitions": {},
            "lemmas": {"comp": "p: a=b, q: b=c -> a=c"},
        },
        "solution": ["compose"],
    },
    {
        "name": "naturality_square",
        "goals": [
            {
                "name": "nat_goal",
                "type_expr": "G(f) ∘ α_a = α_b ∘ F(f)",
                "context": ["α : F => G", "f : a -> b"],
            }
        ],
        "context": {
            "hypotheses": {
                "α": "F => G",
                "f": "a -> b",
                "α_a": "F(a) -> G(a)",
                "α_b": "F(b) -> G(b)",
                "Ff": "F(a) -> F(b)",
                "Gf": "G(a) -> G(b)",
            },
            "definitions": {},
            "lemmas": {"naturality": "naturality of α"},
        },
        "solution": ["rewrite", "apply", "reflexivity"],
    },
    {
        "name": "simple_equality",
        "goals": [
            {
                "name": "refl_goal",
                "type_expr": "x = x",
                "context": ["x : A"],
            }
        ],
        "context": {
            "hypotheses": {"x": "A"},
            "definitions": {"A": "Type"},
            "lemmas": {},
        },
        "solution": ["reflexivity"],
    },
    {
        "name": "transport_simplify",
        "goals": [
            {
                "name": "trans_goal",
                "type_expr": "transport(P, p, x) = y",
                "context": ["p : a = b", "x : P(a)", "y : P(b)"],
            }
        ],
        "context": {
            "hypotheses": {"p": "a = b", "x": "P(a)", "y": "P(b)"},
            "definitions": {"P": "A -> Type"},
            "lemmas": {"transport_eq": "transport(P, p, x) = y -> ..."},
        },
        "solution": ["transport", "rewrite"],
    },
    {
        "name": "equivalence_proof",
        "goals": [
            {
                "name": "equiv_goal",
                "type_expr": "isEquiv(f)",
                "context": ["f : A -> B"],
            }
        ],
        "context": {
            "hypotheses": {"f": "A -> B", "g": "B -> A"},
            "definitions": {"isEquiv": "Σ(g:B->A), ..."},
            "lemmas": {"equiv_criterion": "given g: ..."},
        },
        "solution": ["construct_equivalence", "exact", "exact", "reflexivity"],
    },
]


class ProofStateGenerator:
    """Generates synthetic proof states and trajectories for training."""

    def __init__(self, env: Optional[SimulatedHoTTEnv] = None, seed: int = 42):
        self.env = env or SimulatedHoTTEnv(seed=seed)
        self.templates = list(TEMPLATE_PROBLEMS)
        self._rng = random.Random(seed)

    def generate_problem(self, template_idx: Optional[int] = None) -> Dict[str, Any]:
        """Generate a single proof problem from templates."""
        if template_idx is not None:
            template = self.templates[template_idx % len(self.templates)]
        else:
            template = self._rng.choice(self.templates)
        return dict(template)

    def generate_trajectory(
        self, problem: Optional[Dict[str, Any]] = None, randomize: bool = False,
    ) -> Trajectory:
        """Generate a trajectory by solving a problem with random actions."""
        if problem is None:
            problem = self.generate_problem()

        self.env.reset(problem)
        traj = Trajectory(problem=problem)

        for _ in range(self.env.verifier.max_depth):
            if self.env.observe().is_done:
                break

            actions = self.env.available_actions()
            if not actions:
                break

            action = self._rng.choice(actions)

            prev_state = self.env.observe()
            step = self.env.step(action, [])
            traj.steps.append({
                "state": prev_state,
                "action": action,
                "next_state": step.state,
                "reward": step.reward,
                "done": step.done,
            })

            if step.done:
                break

        return traj

    def generate_dataset(
        self, num_samples: int = 100,
    ) -> List[Trajectory]:
        """Generate a dataset of proof trajectories."""
        trajectories = []
        for i in range(num_samples):
            template_idx = i % len(self.templates)
            problem = self.generate_problem(template_idx)
            traj = self.generate_trajectory(problem)
            trajectories.append(traj)
        return trajectories

    def generate_expert_trajectory(
        self, problem: Dict[str, Any],
    ) -> Optional[Trajectory]:
        """Generate a trajectory following the expert solution path."""
        self.env.reset(problem)
        traj = Trajectory(problem=problem)

        solution = problem.get("solution", [])
        for action in solution:
            if self.env.observe().is_done:
                break

            prev_state = self.env.observe()
            step = self.env.step(action, [])
            traj.steps.append({
                "state": prev_state,
                "action": action,
                "next_state": step.state,
                "reward": step.reward,
                "done": step.done,
            })

            if step.done:
                break

        return traj if traj.steps else None
