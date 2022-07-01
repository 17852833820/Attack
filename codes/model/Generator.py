import torch
import torch.nn as nn


class Generator(nn.Module):  #
    def __init__(self):
        super(Generator, self).__init__()
        self.linear_weight = nn.Parameter(torch.rand(1, 56))

    def forward(self, x, delta=0.2):#x:B up
        weight_mapped = torch.tanh(self.linear_weight)*delta + 1
        # x = torch.mul(torch.kron(torch.ones(2, 1).to("cuda:1"), weight_mapped), x)  # 50,1,2,56
        x = torch.mul(torch.kron(torch.ones(2, 1).cpu(), weight_mapped), x)#weight*data
        return x, weight_mapped


class Generator0(nn.Module):  #
    def __init__(self):
        super(Generator0, self).__init__()
        self.linear_weight = nn.Parameter(torch.rand(1, 56))

    def forward(self, x, delta=0.2):
        weight_mapped = torch.tanh(self.linear_weight)*delta + 1
        x = torch.mul(torch.kron(torch.ones(2, 1).cuda(), weight_mapped), x)  # 50,1,2,56
        return x, weight_mapped


class Generator1(nn.Module):  # random perturbation
    def __init__(self):
        super(Generator1, self).__init__()
        self.linear_weight = nn.Parameter(torch.rand(1, 56))

    def forward(self, x, delta=0.2):
        weight_mapped = (self.linear_weight-0.5)*2*delta + 1
        x = torch.mul(torch.kron(torch.ones(2, 1), weight_mapped), x)  # 50,1,2,56
        return x, weight_mapped


