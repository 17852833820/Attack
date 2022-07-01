import torch
import torch.nn as nn
from torch.utils.data import Dataset
# define CNN model
import torch.nn.functional as F


class ConvCNN40(nn.Module):
    def __init__(self):
        super(ConvCNN40, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, (1, 1))
        self.conv2 = nn.Conv2d(256, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))
        #offine
        #self.fc1 = nn.Linear(128 * 112, 512)
        #online
        self.fc1 = nn.Linear(128 * 104, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.nn.functional.normalize(x, float('inf'), dim=3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
