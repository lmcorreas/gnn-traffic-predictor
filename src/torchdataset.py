import torch
from torch.utils.data import Dataset

class TorchDataset(Dataset):
    def __init__(self, x_train, y_train, wrapper):
        self.x_train = x_train
        self.y_train = y_train
        self.wrapper = wrapper
        
    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, index):
        return (
            self.wrapper.create_tensor(self.x_train[index]), 
            self.wrapper.create_tensor(self.y_train[index])
        )
                