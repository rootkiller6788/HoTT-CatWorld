"""Higher-categorical graph representation for proof states.

Represents proof state as a structured graph where:
- nodes: objects, types, terms
- edges: morphisms, paths, equalities
- hyperedges: higher cells, homotopies, coherence data

Uses networkx for graph operations and torch_geometric-style tensors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import torch


@dataclass
class HigherGraph:
    """A higher-dimensional graph representing proof state structure.

    node_types: 0=object, 1=type, 2=term, 3=universe
    edge_types: 0=morphism, 1=path, 2=equality, 3=equivalence
    cell_types: 0=homotopy, 1=naturality, 2=coherence
    """
    nx_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    node_features: Optional[torch.Tensor] = None
    edge_features: Optional[torch.Tensor] = None
    cell_features: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None

    def add_node(
        self, name: str, node_type: int = 0, features: Optional[torch.Tensor] = None,
    ):
        self.nx_graph.add_node(
            name, node_type=node_type,
            features=features.tolist() if features is not None else [],
        )

    def add_edge(
        self, src: str, dst: str, edge_type: int = 0,
        label: str = "", features: Optional[torch.Tensor] = None,
    ):
        self.nx_graph.add_edge(
            src, dst, edge_type=edge_type, label=label,
            features=features.tolist() if features is not None else [],
        )

    def build_tensors(self, feat_dim: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edge_index and node_feature tensors."""
        nodes = list(self.nx_graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        edge_list = []
        for src, dst in self.nx_graph.edges():
            edge_list.append([node_to_idx[src], node_to_idx[dst]])

        node_feats = []
        for n in nodes:
            data = self.nx_graph.nodes[n]
            if isinstance(data, dict):
                feat = torch.zeros(feat_dim)
                feat[0] = data.get("node_type", 0) / 3.0
                node_feats.append(feat)
            else:
                node_feats.append(torch.zeros(feat_dim))

        self.node_features = torch.stack(node_feats) if node_feats else torch.zeros(0, feat_dim)
        self.edge_index = (
            torch.tensor(edge_list, dtype=torch.long).t()
            if edge_list else torch.zeros(2, 0, dtype=torch.long)
        )
        return self.node_features, self.edge_index

    def num_nodes(self) -> int:
        return self.nx_graph.number_of_nodes()

    def num_edges(self) -> int:
        return self.nx_graph.number_of_edges()


def build_higher_graph(goals: list, context: Any) -> HigherGraph:
    """Build a HigherGraph from a list of proof goals and context."""
    g = HigherGraph()
    for i, goal in enumerate(goals):
        g.add_node(f"goal_{i}", node_type=1)
        deps = getattr(goal, "context", [])
        for j, dep in enumerate(deps):
            dep_name = f"hyp_{i}_{j}"
            g.add_node(dep_name, node_type=2)
            g.add_edge(dep_name, f"goal_{i}", edge_type=1, label="witnesses")
    return g
