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
            #offine
            #index = (self.dt.loc[:, 112] == label) # for testing
            #online
            index = (self.dt.loc[:, 104] == label) # for testing

            self.data = self.dt[index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #offine
        '''features = torch.from_numpy(np.array(self.data.iloc[idx][0:112].values, dtype=np.float)).reshape(1, 2, 56) #.float()
        loc_xy = torch.cat((torch.from_numpy(np.array(self.data.iloc[idx][113:114].values, dtype=float)) / 8.0, torch.from_numpy(np.array(self.data.iloc[idx][114:115].values, dtype=float)) / 5.0), dim=0) #取出x/y轴的数据并作归一化处理

        label = int(self.data.iloc[idx][112])'''
        #online
        features = torch.from_numpy(np.array(self.data.iloc[idx][0:104].values, dtype=np.float)).reshape(1, 2,
                                                                                                         52)  # .float()
        loc_xy = torch.cat((torch.from_numpy(np.array(self.data.iloc[idx][105:106].values, dtype=float)) / 10.0,
                            torch.from_numpy(np.array(self.data.iloc[idx][106:107].values, dtype=float)) / 1.0),
                           dim=0)  # 取出x/y轴的数据并作归一化处理

        label = int(self.data.iloc[idx][104])
        return label, loc_xy, features

