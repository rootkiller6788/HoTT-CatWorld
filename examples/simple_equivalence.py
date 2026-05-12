"""Example: Simple equivalence proof search using MCTS planner.

Search for a proof that x = x using the simulated environment
and MCTS-guided search with world model.
"""

from hottworld.env.simulator import SimulatedHoTTEnv
from hottworld.state.encoder import ProofState, ProofGoal, ProofContext
from hottworld.search.planner import MCTSPlanner


def main():
    print("=" * 60)
    print("Equivalence Proof Search Example")
    print("=" * 60)

    problem = {
        "name": "refl_test",
        "goals": [{"name": "g1", "type_expr": "x = x"}],
        "context": {
            "hypotheses": {"x": "A"},
            "definitions": {"A": "Type"},
        },
    }

    env = SimulatedHoTTEnv(seed=42)
    state = env.reset(problem)
    print(f"\n1. Problem: prove x = x")
    print(f"   Goals: {[g.type_expr for g in state.goals]}")

    planner = MCTSPlanner(
        verifier=env.verifier,
        num_simulations=30,
        max_depth=5,
        seed=42,
    )

    print(f"\n2. Running MCTS search (30 simulations)...")
    tactics, confidence, trace = planner.search(state)
    print(f"   Found {len(tactics)} tactic(s), confidence = {confidence:.4f}")

    print(f"\n3. Action sequence:")
    for i, t in enumerate(tactics):
        print(f"   {i}: {t.kind.value}")

    print(f"\n4. Verifying found proof:")
    env2 = SimulatedHoTTEnv()
    state = env2.reset(problem)
    for tactic in tactics:
        step = env2.step(tactic.kind.value, tactic.args)
        print(f"   {tactic.kind.value:20s} -> valid={step.info['valid']}, "
              f"reward={step.reward:+.2f}, goals={step.state.num_goals}")

    if env2.observe().is_done:
        print(f"\n   PROOF COMPLETE!")
    else:
        print(f"\n   Proof incomplete. Remaining: {env2.observe().num_goals} goals")

    print("\n" + "=" * 60)
    print("Done! This demonstrates proof search via MCTS")
    print("with a simulated HoTT environment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
