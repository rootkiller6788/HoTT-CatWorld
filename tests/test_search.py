"""Tests for proof search: planner, tactics, verifier."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hottworld.state.encoder import ProofState, ProofGoal, ProofContext
from hottworld.search.tactics import Tactic, TacticKind, TacticLibrary
from hottworld.search.verifier import RuleVerifier
from hottworld.search.planner import MCTSPlanner, MCTSNode


class TestTactic:
    def test_create(self):
        t = Tactic(kind=TacticKind.INTRO)
        assert t.kind == TacticKind.INTRO

    def test_to_vector(self):
        t = Tactic(kind=TacticKind.COMPOSE, args=["f", "g"])
        v = t.to_vector(action_dim=16)
        assert v.shape == (16,)

    def test_from_vector(self):
        orig = Tactic(kind=TacticKind.INTRO)
        v = orig.to_vector()
        recovered = Tactic.from_vector(v)
        assert recovered.kind == orig.kind


class TestTacticLibrary:
    def test_all_actions(self):
        lib = TacticLibrary()
        actions = lib.all_actions()
        assert len(actions) >= 5

    def test_applicable(self):
        lib = TacticLibrary()
        state = ProofState(goals=[ProofGoal("g1", "A = B")])
        t = Tactic(kind=TacticKind.INTRO)
        assert lib.applicable(t, state)

    def test_not_applicable_on_empty(self):
        lib = TacticLibrary()
        state = ProofState()
        t = Tactic(kind=TacticKind.INTRO)
        assert not lib.applicable(t, state)

    def test_action_vector(self):
        lib = TacticLibrary()
        state = ProofState(goals=[ProofGoal("g1", "A = B")])
        t = Tactic(kind=TacticKind.INTRO)
        v = lib.action_vector(t, state)
        assert v.shape == (16,)
        assert v[-1].item() == 1.0


class TestRuleVerifier:
    def setup_method(self):
        self.verifier = RuleVerifier()
        self.base_goal = ProofGoal("g1", "x = x", context=["x : A"])
        self.base_state = ProofState(
            goals=[self.base_goal],
            context=ProofContext(
                hypotheses={"x": "A"},
                definitions={"A": "Type"},
            ),
        )

    def test_reflexivity_succeeds(self):
        valid, state, msg = self.verifier.verify(self.base_state, "reflexivity", [])
        assert valid
        assert state.is_done

    def test_reflexivity_fails_on_inequality(self):
        state = ProofState(goals=[ProofGoal("g1", "x = y")])
        valid, state, msg = self.verifier.verify(state, "reflexivity", [])
        assert not valid

    def test_intro(self):
        state = ProofState(goals=[ProofGoal("g1", "A -> B", context=[])])
        valid, next_state, msg = self.verifier.verify(state, "intro", ["x"])
        assert valid
        assert "x" in next_state.context.hypotheses

    def test_apply_with_existing_lemma(self):
        state = ProofState(
            goals=[ProofGoal("g1", "B")],
            context=ProofContext(lemmas={"lem": "A -> B"}),
        )
        valid, next_state, msg = self.verifier.verify(state, "apply", ["lem"])
        assert valid

    def test_apply_missing_lemma(self):
        state = ProofState(goals=[ProofGoal("g1", "B")])
        valid, next_state, msg = self.verifier.verify(state, "apply", ["nonexistent"])
        assert not valid

    def test_exact(self):
        state = ProofState(
            goals=[ProofGoal("g1", "A")],
            context=ProofContext(hypotheses={"x": "A"}),
        )
        valid, next_state, msg = self.verifier.verify(state, "exact", ["x"])
        assert valid
        assert next_state.is_done

    def test_compose(self):
        state = ProofState(goals=[ProofGoal("g1", "A -> B -> C")])
        valid, next_state, msg = self.verifier.verify(state, "compose", [])
        assert valid
        assert len(next_state.goals) >= 1

    def test_rewrite(self):
        valid, state, msg = self.verifier.verify(self.base_state, "rewrite", ["p"])
        assert valid

    def test_transport(self):
        valid, state, msg = self.verifier.verify(self.base_state, "transport", ["p"])
        assert valid

    def test_construct_equivalence(self):
        state = ProofState(goals=[ProofGoal("g1", "isEquiv(f)")])
        valid, next_state, msg = self.verifier.verify(
            state, "construct_equivalence", [],
        )
        assert valid
        assert len(next_state.goals) == 3

    def test_segal(self):
        valid, state, msg = self.verifier.verify(self.base_state, "use_segal", [])
        assert valid

    def test_rezk(self):
        valid, state, msg = self.verifier.verify(self.base_state, "use_rezk", [])
        assert valid

    def test_fill_horn(self):
        valid, state, msg = self.verifier.verify(self.base_state, "fill_horn", [])
        assert valid

    def test_reverse(self):
        valid, state, msg = self.verifier.verify(self.base_state, "reverse", [])
        assert valid

    def test_empty_goals(self):
        state = ProofState()
        valid, state, msg = self.verifier.verify(state, "intro", [])
        assert not valid

    def test_max_depth(self):
        verifier = RuleVerifier(max_depth=3)
        state = ProofState(goals=[ProofGoal("g1", "A -> A")])
        for _ in range(5):
            valid, state, msg = verifier.verify(state, "intro", ["x"])
        assert not valid
        assert "Max depth" in msg

    def test_unknown_tactic(self):
        valid, state, msg = self.verifier.verify(self.base_state, "unknown_tactic", [])
        assert not valid


class TestMCTSNode:
    def test_create(self):
        state = ProofState()
        node = MCTSNode(state=state)
        assert node.visits == 0
        assert node.is_leaf

    def test_value_with_visits(self):
        state = ProofState()
        node = MCTSNode(state=state, visits=10, total_value=5.0)
        assert node.value == 0.5

    def test_value_zero_visits(self):
        state = ProofState()
        node = MCTSNode(state=state)
        assert node.value == 0.0

    def test_ucb_inf_for_zero_visits(self):
        state = ProofState()
        parent = MCTSNode(state=state, visits=10, total_value=5.0)
        child = MCTSNode(state=state, parent=parent, visits=0)
        assert child.ucb_score() == float("inf")


class TestMCTSPlanner:
    def test_create(self):
        planner = MCTSPlanner(
            num_simulations=10,
            max_depth=5,
            seed=42,
        )
        assert planner.num_simulations == 10

    def test_search_reflexivity(self):
        planner = MCTSPlanner(num_simulations=20, max_depth=5, seed=42)
        state = ProofState(
            goals=[ProofGoal("g1", "x = x")],
            context=ProofContext(hypotheses={"x": "A"}, definitions={"A": "Type"}),
        )
        tactics, confidence, trace = planner.search(state)
        assert isinstance(tactics, list)
        assert isinstance(confidence, float)

    def test_beam_search(self):
        planner = MCTSPlanner(num_simulations=10, max_depth=5, seed=42)
        state = ProofState(
            goals=[ProofGoal("g1", "x = x")],
            context=ProofContext(hypotheses={"x": "A"}),
        )
        tactics, confidence = planner.beam_search(state, beam_width=4)
        assert isinstance(tactics, list)
