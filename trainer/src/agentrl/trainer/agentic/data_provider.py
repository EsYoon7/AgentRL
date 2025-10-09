import math
from typing import Sequence

import requests
from torch.utils.data import Dataset

Index = int | str


class AgenticDataset(Dataset):
    def __init__(self, indices: list[Index], name):
        self.indices = indices
        self.name = name

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return {
            "index": self.indices[item],
            "name": self.name,
            "data_source": self.name,
        }


class BalancedDataset(Dataset):
    def __init__(self, datasets: list[Sequence]):
        self.datasets = datasets
        self.total_length = math.lcm(*[len(dataset) for dataset in datasets])

    def __len__(self):
        return self.total_length

    def __getitem__(self, item):
        dataset_idx = item % len(self.datasets)
        sample_idx = item // len(self.datasets) % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]


def get_indices(base_url: str, task_name: str) -> list[Index]:
    url = f"{base_url}/get_indices"
    params = {"name": task_name}

    indices = requests.get(url, params=params).json()
    return indices

def get_agentic_datasets(names, base_url):
    datasets = []
    for name in names:
        indices = get_indices(base_url, name)
        datasets.append(AgenticDataset(indices, name))
    return datasets


