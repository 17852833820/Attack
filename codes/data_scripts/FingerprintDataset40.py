import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
# data loader


class FingerprintDataset40(Dataset):
    def __init__(self, csv_file, label):
        self.dt = pd.read_csv(csv_file, header=None)

        if label == "train": # for training
            self.data = self.dt
        else:
            index = (self.dt.loc[:, 112] == label) # for testing
            self.data = self.dt[index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.from_numpy(np.array(self.data.iloc[idx][0:112].values, dtype=np.float)).reshape(1, 2, 56) #.float()
        loc_xy = torch.cat((torch.from_numpy(np.array(self.data.iloc[idx][113:114].values, dtype=float)) / 8.0, torch.from_numpy(np.array(self.data.iloc[idx][114:115].values, dtype=float)) / 5.0), dim=0) #取出x/y轴的数据并作归一化处理

        label = int(self.data.iloc[idx][112])
        return label, loc_xy, features

