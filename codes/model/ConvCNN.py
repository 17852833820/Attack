import torch
import torch.nn as nn
from torch.utils.data import Dataset
# define CNN model
import torch.nn.functional as F
class ConvCNN(nn.Module):
    def __init__(self):
        super(ConvCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 2)
        self.conv2 = nn.Conv2d(100, 100, 1)
        self.conv3 = nn.Conv2d(100, 16, 1)

        self.fc1 = nn.Linear(16 * 55 , 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
    def forward(self, x):
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,(1,1))
        x=F.max_pool2d(F.relu(self.conv2(x)),1)
        x=F.relu(self.conv3(x))
        x=x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
