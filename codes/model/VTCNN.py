import torch
import torch.nn as nn
from torch.utils.data import Dataset
# define CNN model
# VT-CNN2, see: https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb


class VTCNN(nn.Module):
    def __init__(self):
        super(VTCNN, self).__init__()
        self.net = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(1,3)),
            # nn.Tanh(),
            # nn.Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(12*4*54, 128),
            # nn.Tanh(),
            nn.Flatten(),
            nn.Linear(2*56, 1024), #4*56,1024
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),  #1024,2
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)   #输入【200,1,2,56】200组CSI数据，输出【200,1】200组，输出位置x坐标
        return x