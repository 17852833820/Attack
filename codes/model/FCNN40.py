import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN40_new(nn.Module):
    def __init__(self):
        super(FCNN40_new, self).__init__()
        #offine
        # self.inputs = 1 * 2 * 56  # 输入1x2 SIMO 56个子载?波
        #online
        self.inputs = 1 * 2 * 52
        self.outputs = 2  # 输出大小
        self.l1 = nn.Linear(self.inputs, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 1024)
        self.l4 = nn.Linear(1024, 512)
        self.l5 = nn.Linear(512, 1024)
        self.l6 = nn.Linear(1024, self.outputs)

    def forward(self, x):
        x = torch.nn.functional.normalize(x, float('inf'), dim=3)
        #offine
        # x = x.view(x.shape[0], 1 * 2 * 56)
        #online
        x = x.view(x.shape[0], 1 * 2 * 52)
        x = self.l1(x)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.relu(self.l4(x))
        x = self.l5(x)
        x = self.l6(x)
        x = torch.sigmoid(x)
        return x