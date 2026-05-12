"""Morphism types and equivalences in HoTT.

Defines morphisms, equivalences (ishae), and h-level classifiers:
isContr, isProp, isSet, isGrpd.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Morphism:
    """A morphism between two objects in a (higher) category."""
    name: str
    domain: Any
    codomain: Any
    morph_type: str = "hom"
    metadata: dict = field(default_factory=dict)


@dataclass
class Equivalence:
    """A homotopy equivalence (ishae): f: A->B with quasi-inverse g,
    homotopies η: g∘f ~ id, ε: f∘g ~ id, and coherence τ: fη = εf.
    """
    name: str
    domain: Any
    codomain: Any
    forward: str
    inverse: str
    homotopy_fwd: Optional[str] = None
    homotopy_inv: Optional[str] = None
    coherence: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def is_valid(self) -> bool:
        return all([
            self.forward,
            self.inverse,
            self.homotopy_fwd,
            self.homotopy_inv,
        ])


def is_contr(A: Any) -> Dict[str, Any]:
    """isContr(A) = Σ(c:A), Π(x:A), c = x. A type is contractible."""
    return {"predicate": "isContr", "type": A}


def is_prop(A: Any) -> Dict[str, Any]:
    """isProp(A) = Π(x,y:A), x = y. A type is a mere proposition."""
    return {"predicate": "isProp", "type": A}


def is_set(A: Any) -> Dict[str, Any]:
    """isSet(A) = Π(x,y:A), isProp(x = y). A type is a set (h-set)."""
    return {"predicate": "isSet", "type": A}


def is_grpd(A: Any) -> Dict[str, Any]:
    """isGrpd(A) = Π(x,y:A), isSet(x = y). A type is a 1-groupoid."""
    return {"predicate": "isGrpd", "type": A}


def hlevel(n: int, A: Any) -> Dict[str, Any]:
    """General h-level: isOfHLevel(n, A). n=-2: contractible, n=-1: prop."""
    return {"predicate": "isOfHLevel", "n": n, "type": A}
