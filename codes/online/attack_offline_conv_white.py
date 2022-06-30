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


setup_seed(3)


class MyLoss1(nn.Module):
    def __init__(self):
        super(MyLoss1, self).__init__()

    def forward(self, pred, pos, dmax):
        pos = torch.kron(torch.ones(pred.size()[0], 1).to(device), pos.view(1, 2))
        pred_t = torch.ones_like(pred)
        pos_t = torch.ones_like(pos)
        pred_t[:, 0], pred_t[:, 1] = pred[:, 0]*10.0*0.6, pred[:, 1]*1.0*0.6
        pos_t[:, 0], pos_t[:, 1] = pos[:, 0]*10.0*0.6, pos[:, 1]*1.0*0.6
        temp = F.pairwise_distance(pred_t, pos_t, p=2)
        n = nn.ReLU()(temp-dmax)
        return torch.sum(n)/(torch.count_nonzero(n)+0.01)


class WeightLoss(nn.Module):
    def __init__(self):
        super(WeightLoss, self).__init__()

    def forward(self, weights):
        d = torch.norm(torch.diff(weights), p=2)
        return d


# search for targeted spots for k-th genuine point
def pairing(k, threshold_k):  #k变化范围为[0,9]
    array_k = np.array([])
    for n in np.arange(1, 11):
        d = np.abs(k-n)*0.6
        if d >= threshold_k:
            array_k = np.append(array_k, n)
    return array_k


# train adversarial network
def Train_adv_network(model, network, device, train_loader, k, n, dmax, date):
    target_location = torch.tensor([(n // 5 + 1)/8.0, (n % 5 + 1)/5.0]).to(device)
    d_new = dmax
    model = model.to(device)
    network = network.to(device)
    for param_model in model.parameters():  # fix parameters of loc model
        param_model.requires_grad = False

    myloss1 = MyLoss1().to(device)
    myloss3 = WeightLoss().to(device)

    # optimizer = optim.SGD(network.parameters(), lr=0.5, momentum=0.5)
    optimizer = optim.Adadelta(network.parameters(), lr=1.0)

    for data in train_loader:
        _, pos, inputs = data
        pos, inputs = pos.to(device), inputs.to(device)
        loss_temp = 0.0
        alpha = 1.0
        for Epoch in range(4000):  #
            first_loss = []
            third_loss = []

            optimizer.zero_grad()
            data_per, weights = network(inputs, date)  # add perturbation
            output = model(data_per)  # location predicts

            loss1 = myloss1(output, target_location, d_new)
            loss3 = myloss3(weights)
            loss = loss1 + alpha * loss3  # total loss

            loss.backward()
            optimizer.step()
            first_loss.append(loss1.cpu())
            third_loss.append(loss3.cpu())
            print('[%d-%d][%d] First loss & Third loss: %.6f & %.6f' %
                  (k, n, Epoch + 1, max(first_loss), max(third_loss)))
            if abs(max(first_loss)-loss_temp) <= 0.00001 and d_new >= dmax/5.0:  #控制阈值，使其更加小，以产生更多满足原始阈值的数据，提高准确率
                d_new = d_new/1.05
            if Epoch > 100 and max(first_loss) <= 0.1 and max(third_loss) <= 0.1:  #
                break
            loss_temp = max(first_loss)

            if max(second_loss) <= 0.1 and max(third_loss) >= 0.1:  # 动态改变权重。前期可将alpha=0.1，重要优化攻击精度。精度达到上限之后，逐渐增大alpha，是的gamma更加平滑
                alpha = 30.0
            else:
                alpha = 1.0

    if isinstance(network, torch.nn.DataParallel):
        torch.save(network.module, '../online/adv_conv_white/adv_white_conv' + '%d-' % k + '%d' % n + '.pth')
    else:
        torch.save(network, '../online/adv_conv_white/adv_white_conv' + '%d-' % k + '%d' % n + '.pth')
    return network


def Test_adv_network(model, network, device, test_loader, k, n, dmax, date):  # model: Loc model, CNN: perturbation model, K: original location, n: targeted location, dmax：threshold
    err_k_b = np.array([])  # localization errors to original location before perturbation
    err_n_b = np.array([])  # localization errors to targeted location before perturbation
    err_k_a = np.array([])  # localization errors to original location after perturbation
    err_n_a = np.array([])  # localization errors to targeted location after perturbation
    loc_prediction_b = np.array([])
    loc_prediction_a = np.array([])
    target_location = torch.tensor([n*0.6, 0.6]).to(device)
    model= model.to(device)
    network = network.to(device)
    with torch.no_grad():
        for data in test_loader:
            _, pos, inputs = data
            pos, inputs = pos.to(device), inputs.to(device)
            data_per, adv_weight = network(inputs, date)  # add perturbation
            output = model(data_per)   #perturbed results
            predict = model(inputs)   #genuine results

            output[:, 0], output[:, 1] = output[:, 0]*10.0*0.6, output[:, 1]*1.0*0.6
            pos[:, 0], pos[:, 1] = pos[:, 0]*10.0*0.6, pos[:, 1]*1.0*0.6
            predict[:, 0], predict[:, 1] = predict[:, 0]*10.0*0.6, predict[:, 1]*1.0*0.6

            temp_k_b = F.pairwise_distance(predict, pos, p=2)  # localization errors
            temp_n_b = F.pairwise_distance(predict, target_location, p=2)
            temp_k_a = F.pairwise_distance(output, pos, p=2)
            temp_n_a = F.pairwise_distance(output, target_location, p=2)

            err_k_b = np.append(err_k_b, temp_k_b.cpu())
            err_n_b = np.append(err_n_b, temp_n_b.cpu())
            err_k_a = np.append(err_k_a, temp_k_a.cpu())
            err_n_a = np.append(err_n_a, temp_n_a.cpu())
            loc_prediction_b = np.append(loc_prediction_b, predict.cpu())
            loc_prediction_a = np.append(loc_prediction_a, output.cpu())
    final_acc_a = np.sum(err_n_a <= dmax) / err_n_a.shape[0]
    final_acc_b = np.sum(err_n_b <= dmax) / err_n_b.shape[0]
    print('【%d-%d】' % (k, n))
    print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_b, 0.5), np.quantile(err_k_b, 0.9)))
    print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_a, 0.5), np.quantile(err_k_a, 0.9)))
    print('Before Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_n_b, 0.5), np.quantile(err_n_b, 0.9)))
    print('After Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_n_a, 0.5), np.quantile(err_n_a, 0.9)))
    print(' Before and After Attack accuracy: %.5f' % final_acc_b, final_acc_a)
    return k, n, err_k_b, err_k_a, err_n_b, err_n_a, final_acc_b, final_acc_a, adv_weight.cpu(), loc_prediction_b, loc_prediction_a


num_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
path_train = '../datas/Offline_B_down_SIMO.csv'
path_test = '../datas/Offline_B_up_SIMO.csv'
model = torch.load('../online/conv_white/ConvCNN_white.pth')
# model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
CNN = Generator.Generator()

errors90_all = pickle.load(open("../offline/conv_white/ConvCNN_white_meta_error90_info.pkl", 'rb'))
date = 0.15
d_max = 0.3

Errs_k_b = np.empty((1, 2+500))
Errs_n_b = np.empty((1, 2+500))
Errs_k_a = np.empty((1, 2+500))
Errs_n_a = np.empty((1, 2+500))
Accs_b = np.empty((1, 2+1))
Accs_a = np.empty((1, 2+1))
Adv_weights = np.empty((1, 2+52))
Perdiction_b = np.empty((1, 1+500*2))
Perdiction_a = np.empty((1, 1+500*2))

for k in np.arange(num_classes):
    data_train = create_dataset('FD', path_train, k)
    data_test = create_dataset('FD', path_test, k)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=500, shuffle=True, num_workers=16, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=500, shuffle=False, num_workers=16, pin_memory=True)
    threshold_k = errors90_all[k] + d_max
    list_k = pairing(k, threshold_k)
    for n in list_k:
        # network = torch.load('../offline/adv_conv_white/adv_white_conv' + '%d-' % k + '%d' % n + '.pth')
        network = Train_adv_network(model, CNN, device, dataloader_train, k, n, d_max, date)
        _, _, err_k_b, err_k_a, err_n_b, err_n_a, final_acc_b, final_acc_a, adv_weight, loc_prediction_b, loc_prediction_a  = Test_adv_network(model, network, device, dataloader_test, k, n, d_max, date)
        Errs_k_b = np.append(Errs_k_b, np.array([np.concatenate((np.array([k, n]), err_k_b))]), axis=0)
        Errs_n_b = np.append(Errs_n_b, np.array([np.concatenate((np.array([k, n]), err_n_b))]), axis=0)
        Errs_k_a = np.append(Errs_k_a, np.array([np.concatenate((np.array([k, n]), err_k_a))]), axis=0)
        Errs_n_a = np.append(Errs_n_a, np.array([np.concatenate((np.array([k, n]), err_n_a))]), axis=0)
        Accs_b = np.append(Accs_b, np.array([np.concatenate((np.array([k, n]), np.array([final_acc_b])))]), axis=0)
        Accs_a = np.append(Accs_a, np.array([np.concatenate((np.array([k, n]), np.array([final_acc_a])))]), axis=0)
        Adv_weights = np.append(Adv_weights, np.concatenate((np.array([[k, n]]), adv_weight), axis=1), axis=0)
        Perdiction_a = np.append(Perdiction_a, np.array([np.concatenate((np.array([k]), loc_prediction_a))]), axis=0)
        Perdiction_b = np.append(Perdiction_b, np.array([np.concatenate((np.array([k]), loc_prediction_b))]), axis=0)

Errs_k_b = np.delete(Errs_k_b, [0], axis=0)
Errs_n_b = np.delete(Errs_n_b, [0], axis=0)
Errs_k_a = np.delete(Errs_k_a, [0], axis=0)
Errs_n_a = np.delete(Errs_n_a, [0], axis=0)
Accs_b = np.delete(Accs_b, [0], axis=0)
Accs_a = np.delete(Accs_a, [0], axis=0)
Adv_weights = np.delete(Adv_weights, [0], axis=0)
Perdiction_b = np.delete(Perdiction_b, [0], axis=0)
Perdiction_a = np.delete(Perdiction_a, [0], axis=0)
print('Overall Accuracy Before and After: %.5f & %.5f' % (np.mean(Accs_b[:, 2]), np.mean(Accs_a[:, 2])))
print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_k_b[:, 2:502], 0.5), np.quantile(Errs_k_b[:, 2:502], 0.9)))
print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_k_a[:, 2:502], 0.5), np.quantile(Errs_k_a[:, 2:502], 0.9)))
print('Before Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_n_b[:, 2:502], 0.5), np.quantile(Errs_n_b[:, 2:502], 0.9)))
print('After Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_n_a[:, 2:502], 0.5), np.quantile(Errs_n_a[:, 2:502], 0.9)))

file_name = '../online/conv_white/Attack_Results_all_conv_white.mat'
savemat(file_name, {'Errors_k_b': Errs_k_b, 'Errors_n_b': Errs_n_b, 'Errors_k_a': Errs_k_a, 'Errors_n_a': Errs_n_a, 'Accuracy_before': Accs_b, 'Accuracy_after': Accs_a, 'Adv_weights': Adv_weights, 'Perdiction_b': Perdiction_b, 'Perdiction_a': Perdiction_a})