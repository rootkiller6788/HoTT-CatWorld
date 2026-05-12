"""Example: Naturality square closure.

Given a natural transformation alpha: F => G and a morphism f: a -> b,
construct the naturality square:

    F(a) ---alpha_a--> G(a)
     |                   |
  F(f)|                   |G(f)
     v                   v
    F(b) ---alpha_b--> G(b)

and check which parts need to be proved commutative.
"""

from hottworld.core.diagram import Diagram, naturality_square, check_commute
from hottworld.core.cell import Cell0, Cell1, Cell2


def main():
    print("=" * 60)
    print("Naturality Square Example")
    print("=" * 60)

    Fa = Cell0("F(a)", obj_type="Type")
    Ga = Cell0("G(a)", obj_type="Type")
    Fb = Cell0("F(b)", obj_type="Type")
    Gb = Cell0("G(b)", obj_type="Type")

    alpha_a = Cell1("alpha_a", source=Fa, target=Ga)
    alpha_b = Cell1("alpha_b", source=Fb, target=Gb)
    Ff = Cell1("F(f)", source=Fa, target=Fb)
    Gf = Cell1("G(f)", source=Ga, target=Gb)

    print(f"\n1. Objects (0-cells):")
    print(f"   {Fa.name}, {Ga.name}, {Fb.name}, {Gb.name}")

    print(f"\n2. Morphisms (1-cells):")
    print(f"   {alpha_a.name}: {alpha_a.source.name} -> {alpha_a.target.name}")
    print(f"   {alpha_b.name}: {alpha_b.source.name} -> {alpha_b.target.name}")
    print(f"   {Ff.name}: {Ff.source.name} -> {Ff.target.name}")
    print(f"   {Gf.name}: {Gf.source.name} -> {Gf.target.name}")

    d = naturality_square("F", "G", "alpha", "a", "b")
    print(f"\n3. Diagram: {d}")
    print(f"   Vertices: {d.vertices}")
    print(f"   Edges: {d.edges}")

    commutes = check_commute(d)
    print(f"\n4. Commutativity check:")
    print(f"   The square commutes: {commutes}")

    nat_square = Cell2(
        "naturality_square",
        source=Cell1("top_right", source=Fa, target=Gb),
        target=Cell1("bottom_left", source=Fa, target=Gb),
        cell_type="homotopy",
    )
    print(f"\n5. Naturality witness (2-cell):")
    print(f"   {nat_square.name}: {nat_square.source.name} => {nat_square.target.name}")

    print("\n" + "=" * 60)
    print("To fully prove the naturality square closes,")
    print("you need to construct a 2-cell witness that")
    print("G(f) . alpha_a = alpha_b . F(f)")
    print("=" * 60)


if __name__ == "__main__":
    main()
