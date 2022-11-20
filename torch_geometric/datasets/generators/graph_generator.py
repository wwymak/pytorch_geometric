from torch_geometric.data import Data, InMemoryDataset, download_url
from typing import Optional, Callable
import torch


class GraphGenerator(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def generate_labels(self):
        pass

    def generate_motif(self):
        pass

    def attach_motif(self):
        pass