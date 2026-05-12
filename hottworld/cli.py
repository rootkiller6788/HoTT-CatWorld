"""HoTT-CatWorld CLI.

Commands:
    hottworld simulate --problem <name>     Run simulated proof
    hottworld search --problem <name>        Run proof search
    hottworld train --config <path>          Train world model
    hottworld generate --samples <n>         Generate training data
    hottworld eval --checkpoint <path>       Evaluate model
    hottworld info                           Print package info
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def cmd_simulate(args):
    """Run a simulated proof problem."""
    from hottworld.env.simulator import SimulatedHoTTEnv
    from hottworld.data.generator import ProofStateGenerator, TEMPLATE_PROBLEMS

    template_names = [t["name"] for t in TEMPLATE_PROBLEMS]
    env = SimulatedHoTTEnv(seed=args.seed)
    generator = ProofStateGenerator(env=env, seed=args.seed)

    if args.problem:
        problem = None
        for t in TEMPLATE_PROBLEMS:
            if t["name"] == args.problem:
                problem = t
                break
        if problem is None:
            print(f"Unknown problem: {args.problem}. Available: {template_names}")
            return
        problem = dict(problem)
    else:
        problem = generator.generate_problem()

    print(f"Problem: {problem['name']}")
    state = env.reset(problem)
    print(f"  Goals: {[g.type_expr for g in state.goals]}")
    print(f"  Context: {len(state.context.hypotheses)} hypotheses, {len(state.context.definitions)} defs")
    print()

    done = False
    step_count = 0
    while not done and step_count < args.max_steps:
        actions = env.available_actions()
        if not actions:
            print("  No applicable actions. Stuck.")
            break

        action = actions[0]
        step = env.step(action, [])
        status = "SUCCESS" if step.info.get("valid") else "FAILED"
        print(f"  Step {step_count}: {action:25s} -> {status:8s} | reward={step.reward:+.2f} | {step.info.get('message', '')} | goals={step.state.num_goals}")

        done = step.done
        step_count += 1

    print()
    if step.state.is_done:
        print(f"  Proof completed in {step_count} steps!")
        print(f"  History: {' -> '.join(step.state.history)}")
    else:
        print(f"  Proof incomplete after {step_count} steps. Remaining goals: {step.state.num_goals}")


def cmd_search(args):
    """Run MCTS proof search."""
    from hottworld.env.simulator import SimulatedHoTTEnv
    from hottworld.search.planner import MCTSPlanner
    from hottworld.data.generator import ProofStateGenerator, TEMPLATE_PROBLEMS

    env = SimulatedHoTTEnv(seed=args.seed)
    generator = ProofStateGenerator(env=env, seed=args.seed)

    problem = None
    for t in TEMPLATE_PROBLEMS:
        if t["name"] == args.problem:
            problem = dict(t)
            break

    if problem is None:
        print(f"Unknown problem: {args.problem}")
        return

    state = env.reset(problem)
    print(f"Search problem: {problem['name']}")
    print(f"  Goals: {[g.type_expr for g in state.goals]}")

    env2 = SimulatedHoTTEnv(seed=args.seed)
    env2.reset(problem)

    planner = MCTSPlanner(
        verifier=env2.verifier,
        num_simulations=args.simulations,
        max_depth=args.max_depth,
        seed=args.seed,
    )

    tactics, confidence, trace = planner.search(state)
    print(f"  Search complete: {len(tactics)} tactics, confidence={confidence:.4f}")
    print(f"  Action sequence: {' -> '.join(t.kind.value for t in tactics)}")

    env3 = SimulatedHoTTEnv(seed=args.seed)
    state = env3.reset(problem)
    for tactic in tactics:
        step = env3.step(tactic.kind.value, tactic.args)
        print(f"    {tactic.kind.value:25s} -> goals={step.state.num_goals}, reward={step.reward:+.2f}")

    if env3.observe().is_done:
        print(f"  Proof completed!")
    else:
        print(f"  Proof not complete. Remaining goals: {env3.observe().num_goals}")


def cmd_generate(args):
    """Generate synthetic training data."""
    from hottworld.data.generator import ProofStateGenerator
    from hottworld.data.loader import DataLoader

    config = _load_config(args.config)
    num_samples = args.samples or config.get("data", {}).get("synthetic_samples", 100)
    output_file = args.output or "trajectories.jsonl"

    generator = ProofStateGenerator(seed=args.seed)
    trajectories = generator.generate_dataset(num_samples=num_samples)

    loader = DataLoader()
    loader.save_trajectories(trajectories, output_file)

    solved = sum(1 for t in trajectories if any(s.get("done") for s in t.steps))
    total_steps = sum(len(t.steps) for t in trajectories)

    print(f"Generated {len(trajectories)} trajectories ({solved} solved)")
    print(f"Total steps: {total_steps}, Avg steps/traj: {total_steps/max(len(trajectories),1):.1f}")
    print(f"Saved to: {output_file}")


def cmd_train(args):
    """Train the world model and value function."""
    import torch
    import torch.nn as nn
    from hottworld.model.world_model import WorldModel
    from hottworld.model.value_func import ValueFunc
    from hottworld.state.encoder import StateEncoder
    from hottworld.data.generator import ProofStateGenerator
    from hottworld.env.simulator import SimulatedHoTTEnv

    config = _load_config(args.config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    state_dim = config.get("data", {}).get("state_dim", 64)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.001)
    num_epochs = train_cfg.get("num_epochs", 100)
    num_samples = args.samples or 200

    wm = WorldModel(
        state_dim=state_dim,
        hidden_dim=model_cfg.get("world_model", {}).get("hidden_dim", 128),
        num_layers=model_cfg.get("world_model", {}).get("num_layers", 3),
    )
    vf = ValueFunc(
        state_dim=state_dim,
        hidden_dim=model_cfg.get("value_func", {}).get("hidden_dim", 64),
    )

    encoder = StateEncoder(state_dim=state_dim)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=lr)
    vf_opt = torch.optim.Adam(vf.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    generator = ProofStateGenerator(seed=args.seed)
    trajectories = generator.generate_dataset(num_samples=num_samples)

    train_data = []
    for traj in trajectories:
        for step in traj.steps:
            s = encoder.encode(step["state"])
            if step["done"]:
                tgt = s
            else:
                tgt = encoder.encode(step["next_state"])
            action_name = step["action"]
            a = torch.zeros(16)
            a[hash(action_name) % 16] = 1.0
            train_data.append((s, a, tgt))

    print(f"Training on {len(train_data)} samples, {num_epochs} epochs")
    for epoch in range(num_epochs):
        total_wm_loss = 0.0
        total_vf_loss = 0.0
        n = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            s = torch.stack([b[0] for b in batch])
            a = torch.stack([b[1] for b in batch])
            tgt = torch.stack([b[2] for b in batch])

            pred = wm(s, a)
            wm_loss = mse_loss(pred, tgt)

            wm_opt.zero_grad()
            wm_loss.backward()
            wm_opt.step()

            v = vf(s)
            done_flag = (tgt == s).all(dim=1).float().unsqueeze(1)
            vf_loss = mse_loss(v, done_flag)

            vf_opt.zero_grad()
            vf_loss.backward()
            vf_opt.step()

            total_wm_loss += wm_loss.item() * len(batch)
            total_vf_loss += vf_loss.item() * len(batch)
            n += len(batch)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{num_epochs} | WM loss: {total_wm_loss/n:.6f} | VF loss: {total_vf_loss/n:.6f}")

    checkpoint_dir = args.output or "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    torch.save({
        "world_model": wm.state_dict(),
        "value_func": vf.state_dict(),
        "state_dim": state_dim,
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


def cmd_eval(args):
    """Evaluate a trained model."""
    print("Eval command - not yet implemented")
    print(f"Would load checkpoint from: {args.checkpoint}")


def cmd_info(args):
    """Print package information."""
    from hottworld import __version__
    print(f"HoTT-CatWorld v{__version__}")
    print(f"  Structure-Aware World Models for HoTT & Higher Category Theory")
    print()
    print(f"  Components:")
    print(f"    core/     Higher-categorical data structures (cells, paths, diagrams)")
    print(f"    state/    Proof state encoding (encoder, graph, features)")
    print(f"    model/    World model & value function")
    print(f"    search/   MCTS planner, tactics, verifier")
    print(f"    env/      Simulated HoTT environment (+ Agda/Rzk stubs)")
    print(f"    data/     Synthetic data generation & loading")


def main():
    parser = argparse.ArgumentParser(
        description="HoTT-CatWorld: Structure-Aware World Models for HoTT",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_sim = subparsers.add_parser("simulate", help="Run simulated proof")
    p_sim.add_argument("--problem", type=str, default=None, help="Problem name")
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--max-steps", type=int, default=20)

    p_search = subparsers.add_parser("search", help="Run MCTS proof search")
    p_search.add_argument("--problem", type=str, default="path_composition")
    p_search.add_argument("--seed", type=int, default=42)
    p_search.add_argument("--simulations", type=int, default=50)
    p_search.add_argument("--max-depth", type=int, default=20)

    p_gen = subparsers.add_parser("generate", help="Generate training data")
    p_gen.add_argument("--samples", type=int, default=100)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--output", type=str, default="trajectories.jsonl")
    p_gen.add_argument("--config", type=str, default=None)

    p_train = subparsers.add_parser("train", help="Train world model")
    p_train.add_argument("--config", type=str, default=None)
    p_train.add_argument("--samples", type=int, default=200)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--output", type=str, default="checkpoints")

    p_eval = subparsers.add_parser("eval", help="Evaluate model")
    p_eval.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")

    subparsers.add_parser("info", help="Print package info")

    args = parser.parse_args()

    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
