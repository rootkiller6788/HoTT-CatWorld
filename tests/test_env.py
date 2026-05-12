"""Tests for environment: simulator and stubs."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from hottworld.env.simulator import SimulatedHoTTEnv
from hottworld.env.cubical_stub import CubicalAgdaStub
from hottworld.env.rzk_stub import RzkStub


class TestSimulatedHoTTEnv:
    def setup_method(self):
        self.env = SimulatedHoTTEnv(seed=42)
        self.problem = {
            "name": "refl_test",
            "goals": [
                {"name": "g1", "type_expr": "x = x", "context": ["x : A"]}
            ],
            "context": {
                "hypotheses": {"x": "A"},
                "definitions": {"A": "Type"},
            },
        }

    def test_reset(self):
        state = self.env.reset(self.problem)
        assert state is not None
        assert state.num_goals == 1

    def test_step_reflexivity(self):
        self.env.reset(self.problem)
        step = self.env.step("reflexivity", [])
        assert step.info["valid"]
        assert step.state.is_done
        assert step.reward > 0

    def test_step_illegal(self):
        self.env.reset(self.problem)
        step = self.env.step("nonexistent", [])
        assert not step.info["valid"]
        assert step.reward < 0

    def test_observe(self):
        self.env.reset(self.problem)
        state = self.env.observe()
        assert state is not None
        assert state.num_goals == 1

    def test_available_actions(self):
        self.env.reset(self.problem)
        actions = self.env.available_actions()
        assert len(actions) > 0
        assert "reflexivity" in actions

    def test_close(self):
        self.env.reset(self.problem)
        self.env.close()
        assert self.env.observe() is None

    def test_multistep_proof(self):
        problem = {
            "name": "equiv_test",
            "goals": [
                {"name": "g1", "type_expr": "isEquiv(f)", "context": ["f : A -> B"]}
            ],
            "context": {
                "hypotheses": {"f": "A -> B", "g": "B -> A"},
                "definitions": {"isEquiv": "Type"},
                "lemmas": {"equiv_criterion": "..."},
            },
        }
        state = self.env.reset(problem)
        assert state.num_goals == 1

        actions = self.env.available_actions()
        assert "construct_equivalence" in actions or "apply" in actions


class TestCubicalAgdaStub:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            CubicalAgdaStub()

    def test_close_does_not_raise(self):
        stub = CubicalAgdaStub.__new__(CubicalAgdaStub)
        stub.close()
