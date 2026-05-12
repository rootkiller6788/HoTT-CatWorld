"""Tests for state encoding, graph, and feature extraction."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from hottworld.state.encoder import (
    ProofState, ProofGoal, ProofContext, StateEncoder,
)
from hottworld.state.graph import HigherGraph, build_higher_graph
from hottworld.state.features import StateFeatures, extract_features


class TestProofState:
    def test_create_empty_state(self):
        state = ProofState()
        assert state.num_goals == 0
        assert state.is_done

    def test_create_state_with_goals(self):
        goal = ProofGoal(name="g1", type_expr="A = B")
        state = ProofState(goals=[goal])
        assert state.num_goals == 1
        assert not state.is_done

    def test_state_serialize(self):
        goal = ProofGoal(name="g1", type_expr="A = B")
        state = ProofState(goals=[goal], step=5)
        assert state.step == 5


class TestStateEncoder:
    def test_encoder_create(self):
        enc = StateEncoder(state_dim=64)
        assert enc.state_dim == 64

    def test_encode_empty_state(self):
        enc = StateEncoder(state_dim=64)
        state = ProofState()
        vec = enc.encode(state)
        assert vec.shape == (64,)
        assert vec.dtype == torch.float32

    def test_encode_state_with_goals(self):
        enc = StateEncoder(state_dim=64)
        goals = [
            ProofGoal("g1", "A = B", depth=1),
            ProofGoal("g2", "B = C", depth=2),
        ]
        state = ProofState(goals=goals)
        vec = enc.encode(state)
        assert vec.shape == (64,)

    def test_encode_same_state_gives_same_output(self):
        enc = StateEncoder(state_dim=64)
        state = ProofState(goals=[ProofGoal("g1", "A = B")])
        v1 = enc.encode(state)
        v2 = enc.encode(state)
        assert torch.allclose(v1, v2)


class TestHigherGraph:
    def test_create_graph(self):
        g = HigherGraph()
        g.add_node("A", node_type=0)
        g.add_node("B", node_type=0)
        g.add_edge("A", "B", edge_type=0)
        assert g.num_nodes() == 2
        assert g.num_edges() == 1

    def test_build_tensors(self):
        g = HigherGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")
        node_feats, edge_index = g.build_tensors(feat_dim=8)
        assert node_feats.shape == (2, 8)
        assert edge_index.shape == (2, 1)

    def test_empty_graph_tensors(self):
        g = HigherGraph()
        node_feats, edge_index = g.build_tensors()
        assert node_feats.shape == (0, 8)
        assert edge_index.shape == (2, 0)


class TestBuildHigherGraph:
    def test_build_from_goals(self):
        goal = ProofGoal("g1", "A = B", context=["x:A"])
        g = build_higher_graph([goal], {})
        assert g.num_nodes() >= 1
        assert g.num_edges() >= 0


class TestStateFeatures:
    def test_create_features(self):
        f = StateFeatures(num_goals=3, num_hypos=5)
        assert f.num_goals == 3
        assert not f.is_terminal()

    def test_terminal(self):
        f = StateFeatures(num_goals=0)
        assert f.is_terminal()

    def test_to_tensor(self):
        f = StateFeatures(num_goals=2, num_hypos=10, coherence_total=5, coherence_solved=3)
        t = f.to_tensor()
        assert t.shape == (8,)

    def test_coherence_ratio(self):
        f = StateFeatures(coherence_solved=3, coherence_total=5)
        assert f.coherence_ratio() == 0.6


class TestExtractFeatures:
    def test_extract_simple(self):
        goals = [ProofGoal("g1", "A = B")]
        ctx = ProofContext(hypotheses={"x": "A"}, definitions={"f": "A->B"})
        features = extract_features(goals, ctx)
        assert features.num_goals == 1
        assert features.num_hypos == 1
        assert features.num_defs == 1

    def test_extract_empty(self):
        features = extract_features([], {})
        assert features.num_goals == 0
