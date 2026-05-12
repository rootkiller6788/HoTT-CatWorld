"""Higher-cell algebraic data structures for HoTT/Category Theory.

Cells represent objects (0-cells), morphisms/paths (1-cells),
homotopies/natural transformations (2-cells), and higher coherences (n-cells).
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Cell0:
    """A 0-cell: an object, a type, or a point in a space."""
    name: str
    obj_type: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Cell1:
    """A 1-cell: a morphism, a path, or a homotopy between points."""
    name: str
    source: Cell0
    target: Cell0
    morph_type: Optional[str] = None
    is_inverse: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class Cell2:
    """A 2-cell: a homotopy between 1-cells or a natural transformation."""
    name: str
    source: Cell1
    target: Cell1
    cell_type: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class NCell:
    """An n-cell for arbitrary coherences (n >= 3)."""
    name: str
    dimension: int
    source: Any
    target: Any
    cell_type: Optional[str] = None
    metadata: dict = field(default_factory=dict)
