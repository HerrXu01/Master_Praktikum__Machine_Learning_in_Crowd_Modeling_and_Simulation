import numpy as np
import torch
from torch.utils.data import Dataset

class PedestrianDataset(Dataset):
    """
    A PyTorch dataset class for pedestrian data, for supervised learning tasks.

    Initializes the dataset with features and targets, converting them into PyTorch tensors. 
    It supports indexing and provides the length of the dataset.

    Parameters:
    - features (np.ndarray): A 2D array of input features.
    - targets (np.ndarray): A 1D or 2D array of targets.

    The dataset can be used with a DataLoader for batch processing.
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]