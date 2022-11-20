import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz
from torch_geometric.datasets.generators import GraphGenerator


class GraphWorld(GraphGenerator):
