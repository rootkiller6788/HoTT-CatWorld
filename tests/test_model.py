"""Tests for world model, value function, and reward."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from hottworld.model.world_model import WorldModel
from hottworld.model.value_func import ValueFunc
from hottworld.model.reward import CoherenceReward


class TestWorldModel:
    def test_create(self):
        wm = WorldModel(state_dim=64, action_dim=16, hidden_dim=128)
        assert isinstance(wm, WorldModel)

    def test_forward_shape(self):
        wm = WorldModel(state_dim=64, action_dim=16, hidden_dim=64, num_layers=2)
        s = torch.randn(4, 64)
        a = torch.randn(4, 16)
        out = wm(s, a)
        assert out.shape == (4, 64)

    def test_predict_single(self):
        wm = WorldModel(state_dim=64, action_dim=16, hidden_dim=64)
        s = torch.randn(64)
        a = torch.randn(16)
        out = wm.predict_single(s, a)
        assert out.shape == (64,)

    def test_rollout(self):
        wm = WorldModel(state_dim=64, action_dim=16, hidden_dim=64)
        s = torch.randn(64)
        actions = torch.randn(5, 16)
        states = wm.rollout(s, actions)
        assert len(states) == 6
        assert all(st.shape == (64,) for st in states)

    def test_gradient_flow(self):
        wm = WorldModel(state_dim=64, action_dim=16, hidden_dim=64)
        opt = torch.optim.Adam(wm.parameters(), lr=0.01)
        s = torch.randn(8, 64)
        a = torch.randn(8, 16)
        tgt = torch.randn(8, 64)
        pred = wm(s, a)
        loss = ((pred - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        assert loss.item() > 0


class TestValueFunc:
    def test_create(self):
        vf = ValueFunc(state_dim=64, hidden_dim=32)
        assert isinstance(vf, ValueFunc)

    def test_forward_shape(self):
        vf = ValueFunc(state_dim=64, hidden_dim=32)
        s = torch.randn(4, 64)
        out = vf(s)
        assert out.shape == (4, 1)

    def test_output_range(self):
        vf = ValueFunc(state_dim=64, hidden_dim=32)
        s = torch.randn(10, 64)
        out = vf(s)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_evaluate_single(self):
        vf = ValueFunc(state_dim=64, hidden_dim=32)
        s = torch.randn(64)
        val = vf.evaluate_single(s)
        assert 0.0 <= val <= 1.0


class TestCoherenceReward:
    def test_create(self):
        r = CoherenceReward()
        assert r.r_done == 10.0

    def test_compute_done(self):
        r = CoherenceReward()
        reward = r.compute(None, None, None, is_done=True)
        assert reward > 5.0

    def test_compute_illegal(self):
        r = CoherenceReward()
        reward = r.compute(None, None, None, is_illegal=True)
        assert reward < 0

    def test_compute_step_penalty(self):
        r = CoherenceReward()
        from hottworld.state.encoder import ProofState
        s = ProofState()
        reward = r.compute(s, "test", s, is_done=False)
        assert reward < 0
