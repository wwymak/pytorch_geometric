# flake8: noqa # TODO remove once tests fixed
import pytest
import torch

from torch_geometric.explain import Explainer  # , Explanation
from torch_geometric.explain.algorithm import PGMExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.nn import GCNConv, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        if model_config.mode.value == 'multiclass_classification':
            out_channels = 7
        else:
            out_channels = 1

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, edge_weight, batch=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()

        if self.model_config.task_level.value == 'graph':
            x = global_add_pool(x, batch)

        if self.model_config.mode.value == 'binary_classification':
            x = x.sigmoid()
        elif self.model_config.mode.value == 'multiclass_classification':
            x = x.log_softmax(dim=-1)

        return x


x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])
batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
edge_label_index = torch.tensor([[0, 1, 2], [3, 4, 5]])


@pytest.mark.parametrize('index', [None, 2, torch.arange(3)])
def test_pgm_explainer_node_classification(
    edge_mask_type,
    node_mask_type,
    explanation_type,
    task_level,
    return_type,
    index,
    multi_output,
):
    model_config = ModelConfig(
        mode='binary_classification',
        task_level=task_level,
        return_type=return_type,
    )

    model = GCN(model_config)

    # todo


def test_pgm_explainer_graph_classification(
    edge_mask_type,
    node_mask_type,
    explanation_type,
    task_level,
    return_type,
    index,
    multi_output,
):
    pass
