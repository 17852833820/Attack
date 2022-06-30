import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
# data loader
class FingerprintDataset(Dataset):
    def __init__(self, csv_file, label):
        self.dt = pd.read_csv(csv_file, header=None)

        if label == "train": # for training
            self.data = self.dt
        else:
            index = (self.dt.loc[:, 104] == label) # for testing

            self.data = self.dt[index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.from_numpy(np.array(self.data.iloc[idx][0:104].values, dtype=np.float)).reshape(1, 2, 52)#.float()

        loc_xy = torch.cat((torch.from_numpy(np.array(self.data.iloc[idx][105:106].values,dtype=float)) / 10.0,torch.from_numpy(np.array(self.data.iloc[idx][106:107].values,dtype=float)) / 10.0),dim=0)

        label = int(self.data.iloc[idx][104])
        return label, loc_xy, features

