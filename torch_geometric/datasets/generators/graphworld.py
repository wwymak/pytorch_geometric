import os.path as osp
from typing import Callable, List, Optional, Union

import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.io import read_npz
from torch_geometric.utils import stochastic_blockmodel_graph

from .graph_generator import GraphGenerator


class GraphWorld(GraphGenerator):
    def __init__(
            self,
            motif: Callable,
            block_sizes: Union[List[int], torch.Tensor],
            edge_probs: Union[List[List[float]], torch.Tensor],
            directed: bool = False,
            out_degs=None,
            feature_center_distance=0.0,
            feature_dim=0,
            num_feature_groups=1,
            # feature_group_match_type=MatchType.RANDOM,
            feature_cluster_variance=1.0,
            edge_feature_dim=0,
            edge_center_distance=0.0,
            edge_cluster_variance=1.0,
            normalize_features=True,
            num_nodes: int = 300,
            p_to_q_min=1,
            p_to_q_max=64):
        # self.num_edges = num_edges
        # self.num_motifs = num_motifs
        self.block_sizes = block_sizes
        self.edge_probs = edge_probs
        self.directed = directed
        super().__init__(motif, num_nodes)

    def generate_feature(self, num_features: int = 10):
        self.x = torch.ones((self.num_nodes, num_features), dtype=torch.float)

    def generate_base_graph(self):
        self.edge_index = stochastic_blockmodel_graph(self.block_sizes,
                                                      self.edge_probs,
                                                      self.directed)
        self.attach_motif()
        self.generate_feature()

        data = Data(x=self.x, edge_index=self.edge_index, y=self.node_label,
                    expl_mask=self.expl_mask, edge_label=self.edge_label)

        return data

    def generate_graphs(self):
        pass
    # generate muliple graphs as per paper using p_q ration
