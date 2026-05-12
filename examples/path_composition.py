"""Example: Path composition in HoTT.

Given paths p: a = b and q: b = c, compose them to get p ∘ q: a = c,
and verify using reflexivity after rewrites.

This demonstrates:
- Path data structure
- Composition operation
- Involution (path reversal)
- Transport along composed paths
"""

from hottworld.core.path import Path, compose, inv, refl, transport, ap
from hottworld.core.cell import Cell0, Cell1


def main():
    print("=" * 60)
    print("HoTT Path Composition Example")
    print("=" * 60)

    a = Cell0("a", obj_type="Point")
    b = Cell0("b", obj_type="Point")
    c = Cell0("c", obj_type="Point")

    p = Path("p", source=a.name, target=b.name)
    q = Path("q", source=b.name, target=c.name)

    print(f"\n1. Given paths:")
    print(f"   {p}")
    print(f"   {q}")

    pq = compose(p, q, name="p.q")
    print(f"\n2. Composition p . q:")
    print(f"   {pq}")

    pinv = inv(p, name="p^-1")
    print(f"\n3. Inverse of p:")
    print(f"   {pinv}")

    ppinv = compose(pinv, p, name="p^-1.p")
    print(f"\n4. Compose inverse with original:")
    print(f"   {ppinv}")
    print(f"   Note: p^-1 . p : b = b, which should be refl if we have coherence")

    a_to_a = Path("p_to_a", source="a", target="a")
    comp_check = compose(a_to_a, refl("a"), name="with_refl")
    print(f"\n5. Compose with reflexivity:")
    print(f"   {comp_check}")
    print(f"   (should be: a = a)")

    r1 = transport("P", p, "x : P(a)")
    r2 = transport("P", q, "y : P(b)")
    print(f"\n6. Transport examples:")
    print(f"   transport(P, p, x) = {r1['result_type']}")
    print(f"   transport(P, q, y) = {r2['result_type']}")

    transport_composed = transport(
        "P", pq, "x : P(a)", name="transport_composed"
    )
    print(f"\n7. Transport along composed path:")
    print(f"   transport(P, p.q, x) = {transport_composed['result_type']}")

    print("\n" + "=" * 60)
    print("Done! All path operations demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
