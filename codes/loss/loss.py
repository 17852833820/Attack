import torch
import torch.nn as nn
import torch.nn.functional as F
class MyLoss1(nn.Module):
    def __init__(self):
        super(MyLoss1, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, pos, dmax):
        pos = torch.kron(torch.ones(pred.size()[0], 1).to(self.device), pos.view(1, 2))
        pred_t = torch.ones_like(pred)
        pos_t = torch.ones_like(pos)
        pred_t[:, 0], pred_t[:, 1] = pred[:, 0] * 8.0 * 1.5, pred[:, 1] * 5.0 * 1.5
        pos_t[:, 0], pos_t[:, 1] = pos[:, 0] * 8.0 * 1.5, pos[:, 1] * 5.0 * 1.5
        temp = F.pairwise_distance(pred_t, pos_t, p=2)
        n = nn.ReLU()(temp - dmax)
        return torch.sum(n) / (torch.count_nonzero(n) + 0.01)
class WeightLoss(nn.Module):
    def __init__(self):
        super(WeightLoss, self).__init__()

    def forward(self, weights):
        d = torch.norm(torch.diff(weights), p=2)
        return d
class MyLoss2(nn.Module):
    def __init__(self):
        super(MyLoss2, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    def forward(self, pred, pos, dmin):
        pos = torch.kron(torch.ones(pred.size()[0], 1).to(self.device), pos.view(1, 2))
        pred_t = torch.ones_like(pred)
        pos_t = torch.ones_like(pos)
        pred_t[:, 0], pred_t[:, 1] = pred[:, 0] * 8.0 * 1.5, pred[:, 1] * 5.0 * 1.5
        pos_t[:, 0], pos_t[:, 1] = pos[:, 0] * 8.0 * 1.5, pos[:, 1] * 5.0 * 1.5
        temp = F.pairwise_distance(pred_t, pos_t, p=2)
        n = nn.ReLU()(dmin - temp)
        return torch.sum(n) / (torch.count_nonzero(n) + 0.01)
