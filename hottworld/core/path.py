"""Path algebra: composition, transport, and ap (function application).

In HoTT, paths are identity types: p : a =_A b
- compose(p, q): transitivity of equality
- transport(P, p, x): transport x : P(a) -> P(b) along p
- ap(f, p): congruence, apply f to both sides of equality
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Path:
    """A path between two terms in HoTT, representing a proof of equality."""
    name: str
    source: Any
    target: Any
    path_type: str = "Id"
    children: List["Path"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"{self.name}: {self.source} = {self.target}"


def refl(a: Any, name: str = "refl") -> Path:
    """Reflexivity: a = a."""
    return Path(name=name, source=a, target=a, path_type="refl")


def compose(p: Path, q: Path, name: str = "comp") -> Path:
    """Composition: given p: a=b and q: b=c, produce p ∘ q: a=c."""
    if p.target != q.source:
        raise ValueError(
            f"Cannot compose: target of {p.name} ({p.target!r}) != "
            f"source of {q.name} ({q.source!r})"
        )
    return Path(
        name=name,
        source=p.source,
        target=q.target,
        path_type="comp",
        children=[p, q],
    )


def inv(p: Path, name: str = "inv") -> Path:
    """Inverse: given p: a=b, produce p^-1: b=a."""
    return Path(
        name=name,
        source=p.target,
        target=p.source,
        path_type="inv",
        children=[p],
    )


def transport(
    type_family: str,
    p: Path,
    x: Any,
    name: str = "transport",
) -> Any:
    """Transport: if P: A->Type and p: a=b, then x: P(a) -> P(b).

    Returns a dictionary representing the transported term.
    """
    return {
        "operation": "transport",
        "type_family": type_family,
        "path": p,
        "term": x,
        "result_type": f"{type_family}({p.target})",
    }


def ap(f: str, p: Path, name: str = "ap") -> Path:
    """Apply function: given f: A->B and p: a=b, produce ap_f(p): f(a)=f(b)."""
    return Path(
        name=name,
        source=f"{f}({p.source})",
        target=f"{f}({p.target})",
        path_type="ap",
        children=[p],
        metadata={"function": f},
    )


def path_concat(paths: List[Path], name: str = "concat") -> Path:
    """Compose a list of paths in sequence."""
    if not paths:
        raise ValueError("Cannot concat empty path list")
    result = paths[0]
    for q in paths[1:]:
        result = compose(result, q)
    result.name = name
    return result
