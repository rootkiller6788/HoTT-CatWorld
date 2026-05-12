"""Tests for core higher-categorical data structures."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hottworld.core.cell import Cell0, Cell1, Cell2, NCell
from hottworld.core.path import Path, compose, transport, ap, inv, refl, path_concat
from hottworld.core.diagram import Diagram, check_commute, naturality_square
from hottworld.core.morphism import Morphism, Equivalence, is_contr, is_prop, is_set, is_grpd
from hottworld.core.coherence import (
    CoherenceObligation,
    segal_condition, rezk_condition, coherence_list,
)


class TestCell:
    def test_cell0_create(self):
        c = Cell0(name="A", obj_type="Type")
        assert c.name == "A"
        assert c.obj_type == "Type"

    def test_cell1_create(self):
        a = Cell0("A")
        b = Cell0("B")
        f = Cell1(name="f", source=a, target=b)
        assert f.name == "f"
        assert f.source.name == "A"
        assert f.target.name == "B"

    def test_cell2_create(self):
        a = Cell0("A")
        b = Cell0("B")
        f = Cell1("f", a, b)
        g = Cell1("g", a, b)
        alpha = Cell2(name="alpha", source=f, target=g)
        assert alpha.name == "alpha"
        assert alpha.source.name == "f"
        assert alpha.target.name == "g"

    def test_ncell_create(self):
        c = NCell(name="coh", dimension=3, source="f", target="g")
        assert c.dimension == 3


class TestPath:
    def test_refl(self):
        p = refl("a")
        assert p.source == "a"
        assert p.target == "a"
        assert p.path_type == "refl"

    def test_compose(self):
        p = Path("p", "a", "b")
        q = Path("q", "b", "c")
        r = compose(p, q)
        assert r.source == "a"
        assert r.target == "c"
        assert r.path_type == "comp"

    def test_compose_mismatch_raises(self):
        p = Path("p", "a", "b")
        q = Path("q", "c", "d")
        try:
            compose(p, q)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_inv(self):
        p = Path("p", "a", "b")
        pinv = inv(p)
        assert pinv.source == "b"
        assert pinv.target == "a"

    def test_transport(self):
        p = Path("p", "a", "b")
        result = transport("P", p, "x")
        assert result["operation"] == "transport"
        assert result["result_type"] == "P(b)"

    def test_ap(self):
        p = Path("p", "x", "y")
        apf = ap("f", p)
        assert apf.source == "f(x)"
        assert apf.target == "f(y)"
        assert apf.metadata["function"] == "f"

    def test_path_concat(self):
        p = Path("p", "a", "b")
        q = Path("q", "b", "c")
        r = Path("r", "c", "d")
        result = path_concat([p, q, r])
        assert result.source == "a"
        assert result.target == "d"


class TestDiagram:
    def test_create_diagram(self):
        d = Diagram(name="test")
        va = d.add_vertex("A")
        vb = d.add_vertex("B")
        d.add_edge(va, vb, "f")
        assert len(d.vertices) == 2
        assert len(d.edges) == 1

    def test_naturality_square(self):
        d = naturality_square("F", "G", "alpha", "a", "b")
        assert len(d.vertices) == 4
        assert len(d.edges) == 4

    def test_check_commute_simple(self):
        d = Diagram(name="commuting")
        va = d.add_vertex("A")
        vb = d.add_vertex("B")
        d.add_edge(va, vb, "f")
        assert check_commute(d) is True


class TestMorphism:
    def test_create_morphism(self):
        m = Morphism(name="f", domain="A", codomain="B")
        assert m.domain == "A"
        assert m.codomain == "B"

    def test_create_equivalence(self):
        e = Equivalence(
            name="eq", domain="A", codomain="B",
            forward="f", inverse="g",
            homotopy_fwd="eta", homotopy_inv="eps",
        )
        assert e.is_valid()

    def test_is_contr(self):
        result = is_contr("A")
        assert result["predicate"] == "isContr"

    def test_is_prop(self):
        result = is_prop("A")
        assert result["predicate"] == "isProp"

    def test_is_set(self):
        result = is_set("A")
        assert result["predicate"] == "isSet"

    def test_is_grpd(self):
        result = is_grpd("A")
        assert result["predicate"] == "isGrpd"


class TestCoherence:
    def test_segal_condition(self):
        coh = segal_condition("X", [("f", "g")])
        assert coh.name == "segal_X"

    def test_rezk_condition(self):
        coh = rezk_condition("X")
        assert coh.name == "rezk_X"

    def test_coherence_list(self):
        obligations = coherence_list("X", 3)
        assert len(obligations) == 3
        assert obligations[0].name == "coh_X_level_0"
        assert obligations[2].name == "coh_X_level_2"
