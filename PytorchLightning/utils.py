import torch.utils.data as data
import torch
import h5py
import os
import numpy as np
import random
class customDataset(data.Dataset):
    def __init__(self, inputs, labels):
        super(customDataset, self).__init__()
        self.data = inputs
        self.target = labels

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:].astype('float32')).permute(2, 0, 1), torch.from_numpy(self.target[index,:,:,:].astype('float32')).permute(2, 0, 1)

    def __len__(self):
        return self.data.shape[0]

def set_seed(seed=10):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
