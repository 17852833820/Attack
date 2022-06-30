import os
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


# setup_seed(3)


class MyLoss2(nn.Module):
    def __init__(self):
        super(MyLoss2, self).__init__()

    def forward(self, pred, pos, dmin):
        pos = torch.kron(torch.ones(pred.size()[0], 1).to(device), pos.view(1, 2))
        pred_t = torch.ones_like(pred)
        pos_t = torch.ones_like(pos)
        pred_t[:, 0], pred_t[:, 1] = pred[:, 0]*8.0*1.5, pred[:, 1]*5.0*1.5
        pos_t[:, 0], pos_t[:, 1] = pos[:, 0]*8.0*1.5, pos[:, 1]*5.0*1.5
        temp = F.pairwise_distance(pred_t, pos_t, p=2)
        n = nn.ReLU()(dmin-temp)
        return torch.sum(n)/(torch.count_nonzero(n)+0.01)


# train adversarial network
def Train_adv_network(model, network, device, train_loader, k, dmin, date):
    original_location = torch.tensor([(k // 5 + 1)/8, (k % 5 + 1)/5]).to(device)
    model = model.to(device)
    network = network.to(device)
    for param_model in model.parameters():  # fix parameters of loc model
        param_model.requires_grad = False

    myloss2 = MyLoss2().to(device)
    # optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.5)
    optimizer = optim.Adadelta(network.parameters(), lr=0.5)
    for data in train_loader:
        _, pos, inputs = data
        pos, inputs = pos.to(device), inputs.to(device)
        for Epoch in range(2000):  #
            second_loss = []
            optimizer.zero_grad()
            data_per, _ = network(inputs, date)  # add perturbation
            output = model(data_per)  # location predicts

            loss = myloss2(output, original_location, dmin)
            loss.backward()
            optimizer.step()
            second_loss.append(loss.cpu())
            print('[%d][%d] Second loss:  %.6f' %
                  (k, Epoch + 1,  max(second_loss)))
            if max(second_loss) <= 0.001:
                break
    if isinstance(network, torch.nn.DataParallel):
        torch.save(network.module, '../offline/adv_conv_black/ut_adv_black_conv' + '%d-' % k + '.pth')
    else:
        torch.save(network, '../offline/adv_conv_black/ut_adv_black_conv' + '%d-' % k + '.pth')
    return network


def Test_adv_network(model, network, device, test_loader, k, dmin, date):  # model: Loc model, CNN: perturbation model, K: original location, dmin：threshold
    err_k_b = np.array([])  # localization errors to original location before perturbation
    err_k_a = np.array([])  # localization errors to original location after perturbation
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
            temp_k_a = F.pairwise_distance(output, pos, p=2)

            err_k_b = np.append(err_k_b, temp_k_b.cpu())
            err_k_a = np.append(err_k_a, temp_k_a.cpu())
    final_acc_a = np.sum(err_k_a >= dmin) / err_k_a.shape[0]
    final_acc_b = np.sum(err_k_b >= dmin) / err_k_b.shape[0]
    print('【%d】' % k)
    print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_b, 0.5), np.quantile(err_k_b, 0.9)))
    print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_a, 0.5), np.quantile(err_k_a, 0.9)))
    print(' Before and After Attack accuracy: %.5f' % final_acc_b, final_acc_a)
    return k, err_k_b, err_k_a, final_acc_b, final_acc_a, adv_weight.cpu()


num_classes = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
path_train = '../datas/Offline_B_down_SIMO.csv'
path_test = '../datas/Offline_B_up_SIMO.csv'
model_surrogate = torch.load('../offline/fcnn_black/FCNN_black.pth', map_location=torch.device('cpu'))
model_victim = torch.load('../offline/conv_white/ConvCNN_white.pth', map_location=torch.device('cpu'))
CNN = Generator.Generator()
CNN_random = Generator.Generator1()

errors90_all = pickle.load(open("../offline/conv_white/ConvCNN_white_meta_error90_info.pkl", 'rb'))
date = 0.2

Errs_k_b = np.empty((1, 1+250))
Errs_k_a = np.empty((1, 1+250))
Accs_b = np.empty((1, 1+1))
Accs_a = np.empty((1, 1+1))
Adv_weights = np.empty((1, 1+56))

for k in np.arange(num_classes):
    data_train = create_dataset('FD40', path_train, k)
    data_test = create_dataset('FD40', path_test, k)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=250, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=250, shuffle=True)
    d_min = errors90_all[k] + 0.75

    # network = torch.load('../offline/adv_conv_black/ut_adv_black_conv' + '%d-' % k + '.pth')
    # network = Train_adv_network(model_surrogate, network, device, dataloader_train, k, d_min+0.2, date)
    _, err_k_b, err_k_a, final_acc_b, final_acc_a, adv_weight = Test_adv_network(model_victim, CNN_random, device, dataloader_test, k, d_min, date)
    Errs_k_b = np.append(Errs_k_b, np.array([np.concatenate((np.array([k]), err_k_b))]), axis=0)
    Errs_k_a = np.append(Errs_k_a, np.array([np.concatenate((np.array([k]), err_k_a))]), axis=0)
    Accs_b = np.append(Accs_b, np.array([np.concatenate((np.array([k]), np.array([final_acc_b])))]), axis=0)
    Accs_a = np.append(Accs_a, np.array([np.concatenate((np.array([k]), np.array([final_acc_a])))]), axis=0)
    Adv_weights = np.append(Adv_weights, np.concatenate((np.array([[k]]), adv_weight), axis=1), axis=0)

Errs_k_b = np.delete(Errs_k_b, [0], axis=0)
Errs_k_a = np.delete(Errs_k_a, [0], axis=0)
Accs_b = np.delete(Accs_b, [0], axis=0)
Accs_a = np.delete(Accs_a, [0], axis=0)
Adv_weights = np.delete(Adv_weights, [0], axis=0)
print('Overall Accuracy Before and After: %.5f & %.5f' % (np.mean(Accs_b[:, 1]), np.mean(Accs_a[:, 1])))
print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_k_b[:, 1:251], 0.5), np.quantile(Errs_k_b[:, 1:251], 0.9)))
print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_k_a[:, 1:251], 0.5), np.quantile(Errs_k_a[:, 1:251], 0.9)))


# file_name = '../offline/conv_white/ut_Attack_Results_all_conv_black.mat'
# savemat(file_name, {'Errors_k_b': Errs_k_b, 'Errors_k_a': Errs_k_a, 'Accuracy_before': Accs_b, 'Accuracy_after': Accs_a, 'Adv_weights': Adv_weights})