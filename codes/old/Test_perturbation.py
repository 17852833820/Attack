import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from codes.data_scripts import create_dataset
import torch.optim as optim
from scipy.io import savemat
import math
import pickle
from codes.model import Generator


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(3)


class MyLoss1(nn.Module):
    def __init__(self):
        super(MyLoss1, self).__init__()

    def forward(self, pred, pos, dmax):
        pos = torch.kron(torch.ones(pred.size()[0], 1), pos.view(1, 2))
        pred_t = torch.ones_like(pred)
        pos_t = torch.ones_like(pos)
        pred_t[:, 0], pred_t[:, 1] = pred[:, 0]*8.0*1.5, pred[:, 1]*5.0*1.5
        pos_t[:, 0], pos_t[:, 1] = pos[:, 0]*8.0*1.5, pos[:, 1]*5.0*1.5
        temp = F.pairwise_distance(pred_t, pos_t, p=2)
        n = nn.ReLU()(temp-dmax)
        return torch.sum(n)/(torch.count_nonzero(n)+0.01)


class MyLoss2(nn.Module):
    def __init__(self):
        super(MyLoss2, self).__init__()

    def forward(self, pred, pos, dmin):
        pos = torch.kron(torch.ones(pred.size()[0], 1), pos.view(1, 2))
        pred_t = torch.ones_like(pred)
        pos_t = torch.ones_like(pos)
        pred_t[:, 0], pred_t[:, 1] = pred[:, 0]*8.0*1.5, pred[:, 1]*5.0*1.5
        pos_t[:, 0], pos_t[:, 1] = pos[:, 0]*8.0*1.5, pos[:, 1]*5.0*1.5
        temp = F.pairwise_distance(pred_t, pos_t, p=2)
        n = nn.ReLU()(dmin-temp)
        return torch.sum(n)/(torch.count_nonzero(n)+0.01)


# search for nearest points outside the circle located at k-th point
def pairing(k, threshold_k):
    array_k = np.array([])
    x = k // 5 + 1
    y = k % 5 + 1
    for d in np.arange(1, 8):
        if d * 1.5 >= threshold_k:
            x_s, y_s = [0, 0, -d, d], [-d, d, 0, 0]
        elif d * 1.5 * math.sqrt(2) >= threshold_k:
            x_s, y_s = [d, d, -d, -d], [d, -d, d, -d]
        else:
            continue
        for (xd, yd) in zip(x_s, y_s):
            x_new, y_new = x + xd, y + yd
            if 1 <= x_new <= 8 and 1 <= y_new <= 5:
                label_new = (x_new - 1) * 5 + y_new - 1
                array_k = np.append(array_k, label_new)
        break
    return array_k


def Test_adv_network(model, network, device, test_loader, k, n, dmax, date):  # model: Loc model, CNN: perturbation model, K: original location, n: targeted location, dmax：threshold
    err_k_b = np.array([])  # localization errors to original location before perturbation
    err_n_b = np.array([])  # localization errors to targeted location before perturbation
    err_k_a = np.array([])  # localization errors to original location after perturbation
    err_n_a = np.array([])  # localization errors to targeted location after perturbation
    target_location = torch.tensor([(n // 5+1)*1.5, (n % 5+1)*1.5]).to(device)
    model = model.to(device)
    network = network.to(device)
    with torch.no_grad():
        for data in test_loader:
            _, pos, inputs = data
            pos, inputs = pos.to(device), inputs.to(device)
            data_per, adv_weight = network(inputs, date)  # add perturbation
            output = model(data_per)   #perturbed results
            predict = model(inputs)   #genuine results

            output[:, 0], output[:, 1] = output[:, 0]*8.0*1.5, output[:, 1]*5.0*1.5
            pos[:, 0], pos[:, 1] = pos[:, 0]*8.0*1.5, pos[:, 1]*5.0*1.5
            predict[:, 0], predict[:, 1] = predict[:, 0]*8.0*1.5, predict[:, 1]*5.0*1.5

            temp_k_b = F.pairwise_distance(predict, pos, p=2)  # localization errors
            temp_n_b = F.pairwise_distance(predict, target_location, p=2)
            temp_k_a = F.pairwise_distance(output, pos, p=2)
            temp_n_a = F.pairwise_distance(output, target_location, p=2)

            err_k_b = np.append(err_k_b, temp_k_b.cpu())
            err_n_b = np.append(err_n_b, temp_n_b.cpu())
            err_k_a = np.append(err_k_a, temp_k_a.cpu())
            err_n_a = np.append(err_n_a, temp_n_a.cpu())
    final_acc_a = np.sum(err_n_a <= dmax) / err_n_a.shape[0]
    final_acc_b = np.sum(err_n_b <= dmax) / err_n_a.shape[0]
    print('【%d-%d】' % (k, n))
    print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_b, 0.5), np.quantile(err_k_b, 0.9)))
    print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_a, 0.5), np.quantile(err_k_a, 0.9)))
    print('Before Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_n_b, 0.5), np.quantile(err_n_b, 0.9)))
    print('After Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_n_a, 0.5), np.quantile(err_n_a, 0.9)))
    print(' Before and After Attack accuracy: %.5f' % final_acc_b, final_acc_a)
    return k, n, err_k_b, err_k_a, err_n_b, err_n_a, final_acc_b, final_acc_a, adv_weight.cpu()


num_classes = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_train = '../datas/Offline_B_down_SIMO.csv'
path_test = '../datas/Offline_B_up_SIMO.csv'
model = torch.load('../offline/fcnn_white/FCNN_white.pth')
CNN = Generator.Generator()
errors90_all = pickle.load(open("../offline/fcnn_white/FCNN_White_meta_error90_info.pkl", 'rb'))
date = 0.2
d_max = 0.75

Errs_k_b = np.empty((1, 2+250))
Errs_n_b = np.empty((1, 2+250))
Errs_k_a = np.empty((1, 2+250))
Errs_n_a = np.empty((1, 2+250))
Accs_b = np.empty((1, 2+1))
Accs_a = np.empty((1, 2+1))
Adv_weights = np.empty((1, 2+56))

for k in np.arange(num_classes):
    data_train = create_dataset('FD40', path_train, k)
    data_test = create_dataset('FD40', path_test, k)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, num_workers=0)
    threshold_k = errors90_all[k] + d_max
    list_k = pairing(k, threshold_k)
    for n in list_k:
        network = torch.load('../offline/adv_fcnn_white/adv_white_fcnn'+'%d-'%k+'%d'%n+'.pth')
        _, _, err_k_b, err_k_a, err_n_b, err_n_a, final_acc_b, final_acc_a, adv_weight = Test_adv_network(model, network, device, dataloader_test, k, n, 0.75, date)
        Errs_k_b = np.append(Errs_k_b, np.array([np.concatenate((np.array([k, n]), err_k_b))]), axis=0)
        Errs_n_b = np.append(Errs_n_b, np.array([np.concatenate((np.array([k, n]), err_n_b))]), axis=0)
        Errs_k_a = np.append(Errs_k_a, np.array([np.concatenate((np.array([k, n]), err_k_a))]), axis=0)
        Errs_n_a = np.append(Errs_n_a, np.array([np.concatenate((np.array([k, n]), err_n_a))]), axis=0)
        Accs_b = np.append(Accs_b, np.array([np.concatenate((np.array([k, n]), np.array([final_acc_b])))]), axis=0)
        Accs_a = np.append(Accs_a, np.array([np.concatenate((np.array([k, n]), np.array([final_acc_a])))]), axis=0)
        Adv_weights = np.append(Adv_weights, np.concatenate((np.array([[k, n]]), adv_weight), axis=1), axis=0)

Errs_k_b = np.delete(Errs_k_b, [0], axis=0)
Errs_n_b = np.delete(Errs_n_b, [0], axis=0)
Errs_k_a = np.delete(Errs_k_a, [0], axis=0)
Errs_n_a = np.delete(Errs_n_a, [0], axis=0)
Accs_b = np.delete(Accs_b, [0], axis=0)
Accs_a = np.delete(Accs_a, [0], axis=0)
Adv_weights = np.delete(Adv_weights, [0], axis=0)
print('Overall Accuracy Before and After: %.5f & %.5f' % (np.mean(Accs_b[:, 2]), np.mean(Accs_a[:, 2])))
print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_k_b, 0.5), np.quantile(Errs_k_b, 0.9)))
print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_k_a, 0.5), np.quantile(Errs_k_a, 0.9)))
print('Before Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_n_b, 0.5), np.quantile(Errs_n_b, 0.9)))
print('After Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_n_a, 0.5), np.quantile(Errs_n_a, 0.9)))

file_name = '../offline/fcnn_white/Attack_Results_all_fcnn_white_1.mat'
savemat(file_name, {'Errors_k_b': Errs_k_b, 'Errors_n_b': Errs_n_b, 'Errors_k_a': Errs_k_a, 'Errors_n_a': Errs_n_a, 'Accuracy_before': Accs_b, 'Accuracy_after': Accs_a, 'Adv_weights': Adv_weights})