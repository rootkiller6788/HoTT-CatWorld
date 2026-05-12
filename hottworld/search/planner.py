"""Verifier-Guided MCTS Planner for HoTT proof search.

Uses Monte Carlo Tree Search with a learned world model to guide
proof search through higher-categorical state spaces.

Algorithm:
1. Selection: traverse tree using UCB1 (PUCT variant)
2. Expansion: add child nodes for applicable tactics
3. Simulation: use world model to simulate forward
4. Backpropagation: update value estimates up the tree
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from hottworld.state.encoder import ProofState, StateEncoder
from hottworld.model.world_model import WorldModel
from hottworld.model.value_func import ValueFunc
from hottworld.model.reward import CoherenceReward
from hottworld.search.tactics import Tactic, TacticKind, TacticLibrary
from hottworld.search.verifier import RuleVerifier


@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""
    state: Any
    parent: Optional["MCTSNode"] = None
    action: Optional[Tactic] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.0

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self) -> "MCTSNode":
        if not self.children:
            raise ValueError("No children to select from")
        return max(self.children, key=lambda c: c.visits)

    def most_visited_child(self) -> "MCTSNode":
        if not self.children:
            raise ValueError("No children to select from")
        return max(self.children, key=lambda c: c.visits)


class MCTSPlanner:
    """MCTS planner guided by world model and value function."""

    def __init__(
        self,
        world_model: Optional[WorldModel] = None,
        value_func: Optional[ValueFunc] = None,
        state_encoder: Optional[StateEncoder] = None,
        reward_fn: Optional[CoherenceReward] = None,
        tactic_library: Optional[TacticLibrary] = None,
        verifier: Optional[RuleVerifier] = None,
        exploration_constant: float = 1.414,
        num_simulations: int = 50,
        max_depth: int = 20,
        seed: Optional[int] = None,
    ):
        self.world_model = world_model or WorldModel()
        self.value_func = value_func or ValueFunc()
        self.state_encoder = state_encoder or StateEncoder()
        self.reward_fn = reward_fn or CoherenceReward()
        self.tactic_library = tactic_library or TacticLibrary()
        self.verifier = verifier or RuleVerifier()
        self.exploration_constant = exploration_constant
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def search(
        self, root_state: ProofState, verbose: bool = False,
    ) -> Tuple[List[Tactic], float, List[ProofState]]:
        """Run MCTS and return the best tactic sequence.

        Returns:
            (tactics, confidence, state_trace) - best action sequence found
        """
        root = MCTSNode(state=root_state)

        for sim in range(self.num_simulations):
            node = self._select(root)
            if node is None:
                continue

            expanded = self._expand(node)
            value = self._simulate(expanded or node)
            self._backpropagate(expanded or node, value)

        if not root.children:
            return [], 0.0, [root_state]

        best = root.best_child()
        tactics = self._extract_path(best)
        trace = self._trace_path(best)
        confidence = best.value

        return tactics, confidence, trace

    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Select a leaf node using UCB1."""
        depth = 0
        current = node
        while not current.is_leaf and depth < self.max_depth:
            if not current.children:
                break
            current = max(current.children, key=lambda c: c.ucb_score(self.exploration_constant))
            depth += 1
        return current

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand node by adding children for all applicable tactics."""
        if node.visits == 0:
            return node

        state = node.state
        if getattr(state, "is_done", False):
            return node

        for tactic in self.tactic_library.all_actions():
            applicable = self.tactic_library.applicable(tactic, state)
            if not applicable:
                continue

            valid, next_state, _ = self.verifier.verify(
                state, tactic.kind.value, tactic.args,
            )
            if valid:
                child = MCTSNode(
                    state=next_state, parent=node, action=tactic,
                )
                node.children.append(child)

        if not node.children:
            return node

        return random.choice(node.children)

    def _simulate(self, node: MCTSNode) -> float:
        """Simulate from node to estimate value using world model + value func."""
        state = node.state
        if getattr(state, "is_done", False):
            return 10.0

        state_enc = self.state_encoder.encode(state)
        value = self.value_func.evaluate_single(state_enc)
        return value

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _extract_path(self, node: MCTSNode) -> List[Tactic]:
        """Extract action sequence from root to node."""
        path = []
        current = node
        while current.parent is not None:
            if current.action is not None:
                path.append(current.action)
            current = current.parent
        path.reverse()
        return path

    def _trace_path(self, node: MCTSNode) -> List[ProofState]:
        """Extract state trace from root to node."""
        trace = []
        current = node
        while current is not None:
            trace.append(current.state)
            current = current.parent
        trace.reverse()
        return trace

    def beam_search(
        self, root_state: ProofState, beam_width: int = 8,
    ) -> Tuple[List[Tactic], float]:
        """Alternative: beam search guided by value function."""
        beams: List[Tuple[List[Tactic], ProofState, float]] = [([], root_state, 0.0)]

        for _ in range(self.max_depth):
            candidates = []
            for tactics, state, score in beams:
                if getattr(state, "is_done", False):
                    candidates.append((tactics, state, score))
                    continue

                for tactic in self.tactic_library.all_actions():
                    applicable = self.tactic_library.applicable(tactic, state)
                    if not applicable:
                        continue
                    valid, next_state, _ = self.verifier.verify(
                        state, tactic.kind.value, tactic.args,
                    )
                    if not valid:
                        continue

                    state_enc = self.state_encoder.encode(next_state)
                    val = self.value_func.evaluate_single(state_enc)
                    new_score = score + val
                    candidates.append((tactics + [tactic], next_state, new_score))

            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = candidates[:beam_width]

            if not beams:
                break
            if getattr(beams[0][1], "is_done", False):
                break

        if beams:
            return beams[0][0], beams[0][2]
        return [], 0.0
