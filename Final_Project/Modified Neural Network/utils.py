from torch.utils.data import Dataset, DataLoader, TensorDataset


class CustomDataset1(Dataset):
    """
    Returns dataset with identifier 1
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], 1  
    
class CustomDataset2(Dataset):
    """
    Returns dataset with identifier 2
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], 2