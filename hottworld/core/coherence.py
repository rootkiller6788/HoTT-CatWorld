"""Coherence conditions for higher categories.

Segal condition: composition is uniquely determined up to homotopy.
Rezk condition: equivalences are detected by the universe.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CoherenceObligation:
    """A coherence obligation that must be satisfied in a higher category.

    name: identifier for the coherence
    description: what needs to be proved
    type_expr: the formal type expression of the obligation
    dependencies: other coherence obligations this one depends on
    resolved: whether this obligation has been discharged
    """
    name: str
    description: str
    type_expr: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    resolved: bool = False
    metadata: dict = field(default_factory=dict)


def segal_condition(
    object_name: str,
    composable_pairs: List[Any],
) -> CoherenceObligation:
    """Segal condition: for composable pairs, the type of composites is contractible.

    In a Segal type, horn filling determines composition uniquely.
    """
    return CoherenceObligation(
        name=f"segal_{object_name}",
        description=f"Segal condition for {object_name}: horn fillers determine unique composites",
        type_expr=f"isContr(Σ(g:Hom), Hom2(Δ², {object_name}, g))",
        metadata={"object": object_name, "pairs": composable_pairs},
    )


def rezk_condition(object_name: str) -> CoherenceObligation:
    """Rezk completeness: equivalences coincide with paths in the universe.

    In a Rezk type, the map idtoiso: A=B -> A≅B is an equivalence.
    """
    return CoherenceObligation(
        name=f"rezk_{object_name}",
        description=f"Rezk completeness for {object_name}: idtoiso is an equivalence",
        type_expr=f"isEquiv(idtoiso({object_name}))",
        metadata={"object": object_name},
    )


def coherence_list(
    object_name: str,
    n: int,
) -> List[CoherenceObligation]:
    """Generate a list of coherence obligations for an n-truncated object."""
    obligations = []
    for i in range(n):
        obligations.append(CoherenceObligation(
            name=f"coh_{object_name}_level_{i}",
            description=f"Coherence level {i} for {object_name}",
            type_expr=f"isOfHLevel({i}, {object_name})",
            dependencies=[o.name for o in obligations],
        ))
    return obligations
