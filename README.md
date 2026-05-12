# HoTT-CatWorld

**Structure-Aware World Models for Verifiable Reasoning in Homotopy Type Theory and Higher Category Theory**

A Python-based research framework that applies world model learning to higher-categorical proof search. Instead of predicting raw tactic sequences, the model learns to encode proof states as higher-dimensional graphs (objects, morphisms, 2-cells, paths, coherences) and predict structured state transitions.

## Overview

| Component | Description |
|---|---|
| `core/` | Higher-categorical data structures: cells, paths, diagrams, morphisms, coherences |
| `state/` | Proof state encoder, higher graph representation, feature extraction |
| `model/` | World model (MLP), value function, coherence-aware reward |
| `search/` | MCTS planner, tactic library (13 tactics), rule-based verifier |
| `env/` | Simulated HoTT environment + Agda/Rzk stubs |
| `data/` | Synthetic proof state generator, data loader |

## Quick Start

```bash
# Create venv and install
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"

# Run tests
.venv\Scripts\python -m pytest tests/ -v

# CLI commands
hottworld info
hottworld simulate --problem path_composition
hottworld search --problem simple_equality --simulations 50
hottworld generate --samples 100
hottworld train --samples 200

# Examples
.venv\Scripts\python examples/path_composition.py
.venv\Scripts\python examples/naturality_square.py
.venv\Scripts\python examples/simple_equivalence.py
```

## Architecture

```
Formal Environment (simulated / Agda stub / Rzk stub)
        |
        v
Higher-Categorical State Encoder
  objects, morphisms, 2-cells, paths, diagrams, coherences
        |
        v
Structure World Model
  P(s_{t+1} | s_t, a_t) -- MLP with LayerNorm
        |
        v
Verifier-Guided Planner
  MCTS (UCB1) + beam search + value function
        |
        v
Rule-Based Verifier (simulated)
  compose, rewrite, transport, univalence, Segal, Rezk, ...
```

## Tactic Actions

| Tactic | Description |
|---|---|
| `intro` | Introduce hypothesis |
| `reflexivity` | Close reflexive equality |
| `apply` | Apply lemma to goal |
| `exact` | Exact term match |
| `compose` | Decompose composition |
| `rewrite` | Rewrite with equality |
| `transport` | Transport along path |
| `reverse` | Reverse equality |
| `construct_equivalence` | Build equivalence structure |
| `fill_horn` | Fill simplicial horn |
| `use_segal` | Apply Segal composition |
| `use_rezk` | Apply Rezk completeness |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy, PyYAML, NetworkX

## Future Integration

Real proof assistant backends (currently stubs):

- **Cubical Agda** — `env/cubical_stub.py`
- **Rzk / sHoTT** — `env/rzk_stub.py`

Training data sources (planned):

- [1Lab](https://1lab.dev/) — extensive HoTT/category theory formalization
- [agda-unimath](https://github.com/UniMath/agda-unimath) — univalent mathematics
- [Rzk proof assistant](https://rzk-lang.github.io/rzk/) — synthetic ∞-categories

## License

MIT
