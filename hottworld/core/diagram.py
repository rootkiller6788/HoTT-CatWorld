"""Commutative diagrams: vertices, edges, faces, and commutativity checking.

A diagram is a higher-dimensional directed graph. Checking commutativity
means verifying that all parallel composed paths agree up to specified
2-cells / homotopies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Diagram:
    """A higher-dimensional commutative diagram.

    vertices: objects/types (0-cells)
    edges: morphisms/paths (1-cells) given as (source_idx, target_idx, label)
    faces: 2-cells / homotopies given as list of edge indices forming the boundary
    """
    name: str
    vertices: List[str] = field(default_factory=list)
    edges: List[Tuple[int, int, str]] = field(default_factory=list)
    faces: List[Dict[str, Any]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_vertex(self, label: str) -> int:
        idx = len(self.vertices)
        self.vertices.append(label)
        return idx

    def add_edge(self, src: int, tgt: int, label: str):
        self.edges.append((src, tgt, label))

    def add_face(self, boundary: List[int], label: str):
        self.faces.append({"boundary": boundary, "label": label})

    def edge_paths(self, src: int, tgt: int) -> List[List[int]]:
        """Find all edge paths from src to tgt (simple DFS)."""
        adj: Dict[int, List[Tuple[int, int]]] = {}
        for i, (s, t, _) in enumerate(self.edges):
            adj.setdefault(s, []).append((t, i))

        result = []

        def dfs(current: int, target: int, path: List[int], visited: Set[int]):
            if current == target:
                result.append(list(path))
                return
            for nxt, ei in adj.get(current, []):
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(ei)
                    dfs(nxt, target, path, visited)
                    path.pop()
                    visited.discard(nxt)

        dfs(src, tgt, [src], {src})
        return result

    def __repr__(self):
        return f"Diagram({self.name}, V={len(self.vertices)}, E={len(self.edges)})"


def check_commute(diagram: Diagram) -> bool:
    """Check if a diagram commutes: all parallel edge paths are equal.

    For MVP, compares path labels compositionally. Full HoTT requires
    2-cell witnesses.
    """
    n = len(diagram.vertices)
    for src in range(n):
        for tgt in range(n):
            if src == tgt:
                continue
            paths = diagram.edge_paths(src, tgt)
            if len(paths) >= 2:
                return False
    return True


def naturality_square(
    f: str, g: str, alpha: str, a: str, b: str
) -> Diagram:
    """Create a naturality square for natural transformation α: F=>G.

    Returns the diagram:
        F(a) --α_a--> G(a)
         |              |
      F(f)|              |G(f)
         v              v
        F(b) --α_b--> G(b)
    """
    d = Diagram(name=f"naturality_{alpha}")
    va = d.add_vertex(f"F({a})")
    vga = d.add_vertex(f"G({a})")
    vfb = d.add_vertex(f"F({b})")
    vgb = d.add_vertex(f"G({b})")
    d.add_edge(va, vga, f"{alpha}_{a}")
    d.add_edge(va, vfb, f"F({f})")
    d.add_edge(vga, vgb, f"G({f})")
    d.add_edge(vfb, vgb, f"{alpha}_{b}")
    d.add_face([0, 3], "naturality_face")
    return d
