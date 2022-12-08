import os.path as osp
from typing import Callable, List, Optional, Union

import torch
import enum
import collections
import math
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.io import read_npz
from torch_geometric.utils import stochastic_blockmodel_graph

from .graph_generator import GraphGenerator
from sklearn.preprocessing import normalize
import numpy as np

class MatchType(enum.Enum):
    """Indicates type of feature/graph membership matching to do.
    RANDOM: feature memberships are generated randomly.
    NESTED: for # feature groups >= # graph groups. Each feature cluster is a
      sub-cluster of a graph cluster. Multiplicity of sub-clusters per
      graph cluster is kept as uniform as possible.
    GROUPED: for # feature groups <= # graph groups. Each graph cluster is a
      sub-cluster of a feature cluster. Multiplicity of sub-clusters per
      feature cluster is kept as uniform as possible.
    """
    RANDOM = 1
    NESTED = 2
    GROUPED = 3


class StochasticBlockModelGraph(GraphGenerator):
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
            seed:int = None,
    ):
        self.edge_probs = edge_probs
        self.block_sizes = block_sizes
        self.out_degs = out_degs
        self.directed = directed
        self.x = None
        super().__init__(num_nodes, motif, seed)

    def generate_node_features(
            self,
            center_var,
            feature_dim: int,
            num_groups,
            graph_memberships,
            match_type=MatchType.RANDOM,
            cluster_var=1.0,
            normalize_features=True) -> torch.Tensor:
        r"""

        Args:
            center_var: (float) variance of feature cluster centers. When this is 0.0,
                the signal-to-noise ratio is 0.0. When equal to cluster_var, SNR is 1.0.
            feature_dim: (int) dimension of the multivariate normal.
            num_groups: (int) number of centers. Generated by a multivariate normal with
                mean zero and covariance matrix cluster_var * I_{feature_dim}.
            match_type: (MatchType) see sbm_simulator.MatchType for details.
            cluster_var: (float) variance of feature clusters around their centers.

        Returns:

        """
        feature_memberships = self._generate_feature_memberships(
            graph_memberships=graph_memberships,
            num_groups=num_groups,
            match_type=match_type)

        # Get centers
        centers = []
        center_cov = np.identity(feature_dim) * center_var
        cluster_cov = np.identity(feature_dim) * cluster_var
        for _ in range(num_groups):
            center = np.random.multivariate_normal(
                np.zeros(feature_dim), center_cov, 1)[0]
            centers.append(center)
        features = []
        for cluster_index in feature_memberships:
            feature = np.random.multivariate_normal(centers[cluster_index], cluster_cov,
                                                    1)[0]
            features.append(feature)
        features = np.array(features)
        if normalize_features:
            features = normalize(features)
        return torch.tensor(features)

    def generate_edge_features(
            self,
            feature_dim: int,
            graph_memberships: List,
            center_distance: float =0.0,
            cluster_variance: float =1.0
    ) -> torch.Tensor:
        r"""
            Generates edge feature distribution via inter-class vs intra-class.
            Edge features have two centers: one at (0, 0, ....) and one at
            (center_distance, center_distance, ....) for inter-class and intra-class
            edges (respectively). They are generated from a multivariate normal with
            covariance matrix = cluster_variance * I_d.

            Args:
            feature_dim: (int) dimension of the multivariate normal.
            center_distance: (float) per-dimension distance between the intra-class and
              inter-class means. Increasing this makes the edge feature signal stronger.
            cluster_variance: (float) variance of clusters around their centers.
  """
        center0 = np.zeros(shape=(feature_dim,))
        center1 = np.ones(shape=(feature_dim,)) * center_distance
        covariance = np.identity(feature_dim) * cluster_variance
        edge_features = {}
        for idx in self.edge_index.shape[1]:
            vertex1 = self.edge_index[:, idx][0]
            vertex2 = self.edge_index[:, idx][1]
            edge_tuple = tuple(sorted((vertex1, vertex2)))
            if (graph_memberships[vertex1] ==
                    graph_memberships[vertex2]):
                center = center1
            else:
                center = center0
            edge_features[edge_tuple] = np.random.multivariate_normal(
                center, covariance, 1)[0]
        return edge_features

    def generate_base_graph(self) -> Data:
        self.edge_index = stochastic_blockmodel_graph(
            self.block_sizes, self.edge_probs, directed=not self.directed)
        num_samples = int(self.block_sizes.sum())
        num_classes = self.block_sizes.size(0)
        if self.motif:
            self.attach_motif()
        x = self.generate_node_features()
        edge_attr = self.generate_edge_features()

        data = Data(x=x, edge_index=self.edge_index, y=self.node_label,
                    edge_attr=edge_attr)

        return data

    def _generate_feature_memberships(
            self,
            graph_memberships,
            num_groups=None,
            match_type=MatchType.RANDOM):
        """Generates a feature membership assignment.
        Args:
          graph_memberships: (list) the integer memberships for the graph SBM
          num_groups: (int) number of groups. If None, defaults to number of unique
            values in graph_memberships.
          match_type: (MatchType) see the enum class description.
        Returns:
          memberships: a int list - index i contains feature group of node i.
        """
        # Parameter checks
        if num_groups is not None and num_groups == 0:
            raise ValueError("argument num_groups must be None or positive")
        graph_num_groups = len(set(graph_memberships))
        if num_groups is None:
            num_groups = graph_num_groups

        # Compute memberships
        memberships = []
        if match_type == MatchType.GROUPED:
            if num_groups > graph_num_groups:
                raise ValueError(
                    "for match type GROUPED, must have num_groups <= graph_num_groups")
            nesting_map = self._get_nesting_map(graph_num_groups, num_groups)
            # Creates deterministic map from (smaller) graph clusters to (larger)
            # feature clusters.
            reverse_nesting_map = {}
            for feature_cluster, graph_cluster_list in nesting_map.items():
                for cluster in graph_cluster_list:
                    reverse_nesting_map[cluster] = feature_cluster
            for cluster in graph_memberships:
                memberships.append(reverse_nesting_map[cluster])
        elif match_type == MatchType.NESTED:
            if num_groups < graph_num_groups:
                raise ValueError(
                    "for match type NESTED, must have num_groups >= graph_num_groups")
            nesting_map = self._get_nesting_map(num_groups, graph_num_groups)
            # Creates deterministic map from (smaller) feature clusters to (larger)
            # graph clusters.
            for graph_cluster_id, feature_cluster_ids in nesting_map.items():
                sorted_feature_cluster_ids = sorted(feature_cluster_ids)
                num_feature_groups = len(sorted_feature_cluster_ids)
                feature_pi = np.ones(num_feature_groups) / num_feature_groups
                num_graph_cluster_nodes = np.sum(
                    [i == graph_cluster_id for i in graph_memberships])
                sub_memberships = self._generate_node_memberships(num_graph_cluster_nodes,
                                                           feature_pi)
                sub_memberships = [sorted_feature_cluster_ids[i] for i in sub_memberships]
                memberships.extend(sub_memberships)
        else:  # MatchType.RANDOM
            memberships = np.random.choice(range(num_groups), size=len(graph_memberships))
        return np.array(sorted(memberships))


    def _get_nesting_map(self, large_k, small_k):
        """Given two group sizes, computes a "nesting map" between groups.
        This function will produce a bipartite map between two sets of "group nodes"
        that will be used downstream to partition nodes in a bigger graph. The map
        encodes which groups from the larger set are nested in certain groups from
        the smaller set.
        As currently implemented, nesting is assigned as evenly as possible. If
        large_k is an integer multiple of small_k, each smaller-set group will be
        mapped to exactly (large_k/small_k) larger-set groups. If there is a
        remainder r, the first r smaller-set groups will each have one extra nested
        larger-set group.
        Args:
          large_k: (int) size of the larger group set
          small_k: (int) size of the smaller group set
        Returns:
          nesting_map: (dict) map from larger group set indices to lists of
            smaller group set indices
        """
        min_multiplicity = int(math.floor(large_k / small_k))
        max_bloated_group_index = large_k - small_k * min_multiplicity - 1
        nesting_map = collections.defaultdict(list)
        pos = 0
        for i in range(small_k):
            for _ in range(min_multiplicity + int(i <= max_bloated_group_index)):
                nesting_map[i].append(pos)
                pos += 1
        return nesting_map

    def _ComputeExpectedEdgeCounts(self, num_edges, num_vertices,
                                   pi,
                                   prop_mat):
        """Computes expected edge counts within and between communities.
        Args:
          num_edges: expected number of edges in the graph.
          num_vertices: number of nodes in the graph
          pi: interable of non-zero community size proportions. Must sum to 1.0, but
            this check is left to the caller of this internal function.
          prop_mat: square, symmetric matrix of community edge count rates. Entries
            must be non-negative, but this check is left to the caller.
        Returns:
          symmetric matrix with shape prop_mat.shape giving expected edge counts.
        """
        scale = np.matmul(pi, np.matmul(prop_mat, pi)) * num_vertices ** 2
        prob_mat = prop_mat * num_edges / scale
        return np.outer(pi, pi) * prob_mat * num_vertices ** 2

    def _compute_community_sizes(self, num_vertices, pi):
        """Helper function of GenerateNodeMemberships to compute group sizes.
        Args:
          num_vertices: number of nodes in graph.
          pi: interable of non-zero community size proportions.
        Returns:
          community_sizes: np vector of group sizes. If num_vertices * pi[i] is a
            whole number (up to machine precision), community_sizes[i] will be that
            number. Otherwise, this function accounts for rounding errors by making
            group sizes as balanced as possible (i.e. increasing smallest groups by
            1 or decreasing largest groups by 1 if needed).
        """
        community_sizes = [int(x * num_vertices) for x in pi]
        if sum(community_sizes) != num_vertices:
            size_order = np.argsort(community_sizes)
            delta = sum(community_sizes) - num_vertices
            adjustment = np.sign(delta)
            if adjustment == 1:
                size_order = np.flip(size_order)
            for i in range(int(abs(delta))):
                community_sizes[size_order[i]] -= adjustment
        return community_sizes

    def _generate_node_memberships(
            self, num_vertices,pi):
        """Gets node memberships for sbm.
        Args:
          num_vertices: number of nodes in graph.
          pi: interable of non-zero community size proportions. Must sum to 1.0, but
            this check is left to the caller of this internal function.
        Returns:
          np vector of ints representing community indices.
        """
        community_sizes = self._compute_community_sizes(num_vertices, pi)
        memberships = np.zeros(num_vertices, dtype=int)
        node = 0
        for i in range(len(pi)):
            memberships[range(node, node + community_sizes[i])] = i
            node += community_sizes[i]
        return memberships


class GeneratorConfigSampler:
    # Base class for sampling generator configs.
    #
    # A child class C should call the following in its __init__:
    #   super(C, self).__init__(param_sample_specs)
    # Following this line, add sampler_fns to each spec with:
    #   self._AddSampleFn('foo', self._SampleUniformInteger)
    #   self._AddSampleFn('bar', self._SampleUniformFloat)
    # The sampler_fn can also be any function accessible to the child class.
    #
    # Arguments:
    #   param_sampler_specs: a list of ParamSamplerSpecs.

    def _SampleUniformInteger(self, param_sampler):
        low = int(param_sampler.min_val)
        high = int(param_sampler.max_val)
        if high < low:
            raise RuntimeError(
                "integer sampling for %s failed as high < low" % param_sampler.name)
        return low if low == high else np.random.randint(low, high)

    def _SampleUniformFloat(self, param_sampler):
        return np.random.uniform(param_sampler.min_val, param_sampler.max_val)

    def _AddSamplerFn(self, param_name, sampler_fn):
        if param_name not in self._param_sampler_specs:
            raise RuntimeError("param %s not found in input param specs" % param_name)
        self._param_sampler_specs[param_name].sampler_fn = sampler_fn

    def _ChooseMarginalParam(self):
        valid_params = [
            param_name for
            param_name, spec in self._param_sampler_specs.items() if
            spec.min_val != spec.max_val]
        if len(valid_params) == 0:
            return None
        return random.choice(valid_params)

    def __init__(self, param_sampler_specs):
        self._param_sampler_specs = {spec.name: spec for spec in param_sampler_specs}

    def SampleConfig(self, marginal=False):
        config = {}
        marginal_param = None
        if marginal:
            marginal_param = self._ChooseMarginalParam()
        fixed_params = []
        for param_name, spec in self._param_sampler_specs.items():
            param_value = None
            if marginal and marginal_param is not None:
                # If the param is not a marginal param, give it its default (if possible)
                if param_name != marginal_param:
                    if spec.default_val is not None:
                        fixed_params.append(param_name)
                        param_value = spec.default_val
            # If the param val is still None, give it a random value.
            if param_value is None:
                param_value = spec.sampler_fn(spec)
            config[param_name] = param_value
        return config, marginal_param, fixed_params


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
            p_to_q_max=64,
            seed=None,
    ):
        # self.num_edges = num_edges
        # self.num_motifs = num_motifs
        self.block_sizes = block_sizes
        self.edge_probs = edge_probs
        self.directed = directed
        super().__init__(num_nodes=num_nodes, motif=motif, seed=seed)



    # Helper function to create the "PropMat" matrix for the SBM model (square
    # matrix giving inter-community Poisson means) from the config parameters,
    # particularly `p_to_q_ratio`. See the config proto for details.
    def make_prop_matrix(self, num_communities: int, p_to_q_ratio: float) -> np.ndarray:
        prop_mat = np.ones((num_communities, num_communities))
        np.fill_diagonal(prop_mat, p_to_q_ratio)
        return prop_mat

    def generate_config(self, generator_params, num_samples):
        config = {}
        for param_name, spec in generator_params.items():
            min_value = spec['min_value']
            max_value = spec['max_value']
            param_type = spec['param_type']

            if param_type == 'int':
                config[param_name] = np.random.randint(
                    low=min_value,high=max_value, size=num_samples)
            elif param_type=='float':
                config[param_name] = (np.random.random_sample(
                    size=num_samples) * (max_value - min_value)
                                      + min_value)
        if 'p_to_q_ratio' in config.keys():
            config['edge_probs'] = self.make_prop_matrix(config.pop('p_to_q_ratio'))
        else:
            # double check if the default value of 1 is correct
            config['edge_probs'] = self.make_prop_matrix(1)
    def generate_graphs(self, p_q_ratio_min, p_q_ratio_max):
        configs = self.generate_config()
        for config in configs:
            sbm = StochasticBlockModelGraph()
        yield {**config, 'data':sbm}
    # generate muliple graphs as per paper using p_q ration
