import torch
import torch.nn as nn
from torch.utils.data import Dataset
# define CNN model
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.inputs = 1* 2 * 52
        self.outputs = 2
        self.l1 = nn.Linear(self.inputs, 120)
        self.l2 = nn.Linear(120, 80)
        self.l3 = nn.Linear(80, 60)
        self.l4 = nn.Linear(60, 30)
        self.l5 = nn.Linear(30, 10)

        self.l6 = nn.Linear(10, self.outputs)
    def forward(self, x):
        x = x.view(x.shape[0], 1 * 2 * 52)
        x=self.l1(x)
        x = F.relu(x)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        return x
