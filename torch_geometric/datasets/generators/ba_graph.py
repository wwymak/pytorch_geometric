from typing import Callable

from torch_geometric.data import Data
from torch_geometric.utils import barabasi_albert_graph

from .graph_generator import GraphGenerator


class BAGraph(GraphGenerator):
    r"""Generate base graph for Barabasi-Albert (BA) graph.

    Args:
        motif: the Motif object
        num_nodes (int, optional): Specifies the number of the nodes in the
            base graph. (default: :obj:`300`)
        num_edges: number of edges
        num_motifs: number of motifs to attach to the base graph
    """
    def __init__(self, motif: Callable, num_nodes: int = 300,
                 num_edges: int = 5, num_motifs: int = 80):
        self.num_edges = num_edges
        self.num_motifs = num_motifs
        super().__init__(motif, num_nodes)

    def generate_base_graph(self):
        self.edge_index = barabasi_albert_graph(self.num_nodes, self.num_edges)
        self.attach_motif()
        self.generate_feature()

        data = Data(x=self.x, edge_index=self.edge_index, y=self.node_label,
                    expl_mask=self.expl_mask, edge_label=self.edge_label)

        return data
