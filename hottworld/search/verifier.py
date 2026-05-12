"""Rule-based verifier for simulated HoTT proof environment.

Validates tactic applications against proof state and produces
the resulting proof state without calling an external proof assistant.
"""

from copy import deepcopy
from typing import Any, List, Optional, Tuple

from hottworld.state.encoder import ProofState, ProofGoal


class RuleVerifier:
    """Verifier that simulates proof state transitions using simple rules.

    For the MVP, uses a simplified type-theoretic model:
    - Goals are type expressions (strings)
    - Tactics transform goals based on pattern matching
    - No actual type checking is performed
    """

    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth

    def verify(
        self, state: ProofState, tactic_name: str, tactic_args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        """Apply a tactic and return (valid, next_state, message)."""
        state = deepcopy(state)
        state.step += 1

        if state.step > self.max_depth:
            return False, state, "Max depth exceeded"

        if not state.goals:
            return False, state, "No goals remain - proof is already complete"

        handler = getattr(self, f"_apply_{tactic_name}", None)
        if handler is None:
            return False, state, f"Unknown tactic: {tactic_name}"

        return handler(state, tactic_args)

    def _apply_intro(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals.pop(0)
        hyp_name = args[0] if args else "x"
        goal_type_parts = goal.type_expr.split(" -> ", 1)
        if len(goal_type_parts) == 2:
            new_goal = ProofGoal(
                name=goal.name, type_expr=goal_type_parts[1], depth=goal.depth + 1,
                context=goal.context + [hyp_name],
            )
            state.goals.insert(0, new_goal)
        state.context.hypotheses[hyp_name] = goal_type_parts[0] if len(goal_type_parts) == 2 else goal.type_expr
        state.history.append(f"intro {hyp_name}")
        return True, state, f"Introduced {hyp_name}"

    def _apply_reflexivity(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals[0]
        parts = goal.type_expr.split(" = ")
        if len(parts) == 2 and parts[0].strip() == parts[1].strip():
            state.goals.pop(0)
            state.history.append("reflexivity")
            return True, state, "Reflexivity succeeded"
        return False, state, f"Not trivially equal: {goal.type_expr}"

    def _apply_apply(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        if not args:
            return False, state, "apply requires lemma name"
        lemma_name = args[0]
        if lemma_name in state.context.lemmas or lemma_name in state.context.definitions:
            goal = state.goals.pop(0)
            state.history.append(f"apply {lemma_name}")
            return True, state, f"Applied {lemma_name} to {goal.name}"
        return False, state, f"Lemma {lemma_name} not found"

    def _apply_compose(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals[0]
        parts = goal.type_expr.split(" -> ")
        if len(parts) >= 3:
            state.goals.pop(0)
            state.goals.insert(0, ProofGoal(
                name=f"{goal.name}_1", type_expr=f"{parts[0]} -> ?",
                depth=goal.depth + 1, context=goal.context,
            ))
            state.goals.insert(1, ProofGoal(
                name=f"{goal.name}_2", type_expr=f"? -> {parts[-1]}",
                depth=goal.depth + 1, context=goal.context,
            ))
            state.history.append("compose")
            return True, state, "Composed into subgoals"
        return False, state, "Cannot decompose composition"

    def _apply_rewrite(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals.pop(0)
        state.history.append(f"rewrite {args[0] if args else '?'}")
        new_type = goal.type_expr
        if args:
            new_type = f"R({goal.type_expr})"
        new_goal = ProofGoal(
            name=goal.name, type_expr=new_type,
            depth=goal.depth + 1, context=goal.context,
        )
        state.goals.insert(0, new_goal)
        return True, state, "Rewrote goal"

    def _apply_transport(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals.pop(0)
        state.history.append(f"transport {args[0] if args else '?'}")
        new_goal = ProofGoal(
            name=goal.name, type_expr=f"T({goal.type_expr})",
            depth=goal.depth + 2, context=goal.context,
        )
        state.goals.insert(0, new_goal)
        return True, state, "Applied transport"

    def _apply_exact(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        if not args:
            return False, state, "exact requires term name"
        term = args[0]
        if term in state.context.hypotheses or term in state.context.definitions:
            state.goals.pop(0)
            state.history.append(f"exact {term}")
            return True, state, f"Exact {term}"
        return False, state, f"Term {term} not in context"

    def _apply_construct_equivalence(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals[0]
        if "equiv" in goal.type_expr.lower() or "iso" in goal.type_expr.lower():
            state.goals.pop(0)
            base_name = goal.name
            state.goals.insert(0, ProofGoal(
                name=f"{base_name}_fwd", type_expr="forward map",
                depth=goal.depth + 1, context=goal.context,
            ))
            state.goals.insert(1, ProofGoal(
                name=f"{base_name}_inv", type_expr="inverse map",
                depth=goal.depth + 1, context=goal.context,
            ))
            state.goals.insert(2, ProofGoal(
                name=f"{base_name}_coh", type_expr="coherence",
                depth=goal.depth + 1, context=goal.context,
            ))
            state.history.append("construct_equivalence")
            return True, state, "Decomposed equivalence into 3 subgoals"
        return False, state, "Goal is not an equivalence type"

    def _apply_use_segal(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals.pop(0)
        state.history.append("use_segal")
        new_goal = ProofGoal(
            name=goal.name, type_expr=f"Segal({goal.name})",
            depth=goal.depth + 1, context=goal.context,
        )
        state.goals.insert(0, new_goal)
        return True, state, "Applied Segal composition"

    def _apply_use_rezk(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals.pop(0)
        state.history.append("use_rezk")
        new_goal = ProofGoal(
            name=goal.name, type_expr=f"Rezk({goal.name})",
            depth=goal.depth + 1, context=goal.context,
        )
        state.goals.insert(0, new_goal)
        return True, state, "Applied Rezk completeness"

    def _apply_fill_horn(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals.pop(0)
        state.history.append("fill_horn")
        new_goal = ProofGoal(
            name=f"{goal.name}_horn", type_expr=f"horn({goal.type_expr})",
            depth=goal.depth + 1, context=goal.context,
        )
        state.goals.insert(0, new_goal)
        return True, state, "Horn filler constructed"

    def _apply_reverse(
        self, state: ProofState, args: List[str],
    ) -> Tuple[bool, Optional[ProofState], str]:
        if not state.goals:
            return False, state, "No goals"
        goal = state.goals.pop(0)
        state.history.append("reverse")
        parts = goal.type_expr.split(" = ", 1)
        if len(parts) == 2:
            new_type = f"{parts[1].strip()} = {parts[0].strip()}"
        else:
            new_type = f"inv({goal.type_expr})"
        new_goal = ProofGoal(
            name=goal.name, type_expr=new_type,
            depth=goal.depth + 1, context=goal.context,
        )
        state.goals.insert(0, new_goal)
        return True, state, "Reversed equality"
