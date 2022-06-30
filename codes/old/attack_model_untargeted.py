import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from codes.data_scripts import create_dataset
import torch.optim as optim
from scipy.io import savemat
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


# train adversarial network
def Train_adv_network(model, network, device, train_loader, k, dmin_k, date):
    original_location = torch.tensor([(k // 5 + 1)/8, (k % 5 + 1)/5]).to(device)
    model = model.to(device)
    for param_model in model.parameters():  # fix parameters of loc model
        param_model.requires_grad = False

    myloss2 = MyLoss2().to(device)
    network = network.to(device)
    optimizer = optim.SGD(network.parameters(), lr=0.4, momentum=0.7)

    for Epoch in range(500):  #
        second_loss = []
        for data in train_loader:
            _, pos, inputs = data
            pos, inputs = pos.to(device), inputs.to(device)

            optimizer.zero_grad()
            data_per, _ = network(inputs, date)  # add perturbation
            output = model(data_per)  # location predicts

            loss2 = myloss2(output, original_location, dmin_k)
            loss = loss2
            loss.backward()
            optimizer.step()
            second_loss.append(loss2.item())
        print('[%d][%d] Loss: %.6f' %
              (k, Epoch + 1, max(second_loss)))
        if max(second_loss) <= 0.001:
            break
    torch.save(network, '../offline/adv_fcnn_white/adv_white_fcnn_untargeted'+'%d-'%k+'.pth')
    return network


def Test_adv_network(model, network, device, test_loader, k, dmin_k, date):  # model: Loc model, CNN: perturbation model, K: original location, dmin_k：threshold
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

            err_k_b = np.append(err_k_b, temp_k_b.numpy())
            err_k_a = np.append(err_k_a, temp_k_a.numpy())
    final_acc = np.sum(err_k_a >= dmin_k)/err_k_a.shape[0]
    print('【%d】' % k)
    print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_b, 0.5), np.quantile(err_k_b, 0.9)))
    print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_a, 0.5), np.quantile(err_k_a, 0.9)))
    print('Untargeted Attack accuracy: %.5f' % final_acc)
    return k, err_k_b, err_k_a, final_acc, adv_weight.numpy()


num_classes = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_train = '../datas/Offline_B_down_SIMO.csv'
path_test = '../datas/Offline_B_up_SIMO.csv'
model = torch.load('../offline/fcnn_white/FCNN_white.pth')
model = model.double()
CNN = Generator.Generator()
errors90_all = pickle.load(open("../offline/fcnn_white/FCNN_White_meta_error90_info.pkl", 'rb'))
date = 0.2

Errs_k_b = np.empty((1, 1+250))
Errs_k_a = np.empty((1, 1+250))
Accs_untargeted = np.empty((1, 1+1))
Adv_weights = np.empty((1, 1+56))

for k in np.arange(num_classes):
    data_train = create_dataset('FD40', path_train, k)
    data_test = create_dataset('FD40', path_test, k)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=50, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=50, shuffle=False, num_workers=0)
    threshold_k = errors90_all[k]+0.75

    network = Train_adv_network(model, CNN, device, dataloader_train, k, threshold_k, date)
    _, err_k_b, err_k_a, final_acc, adv_weight = Test_adv_network(model, network, device, dataloader_test, k, threshold_k, date)
    Errs_k_b = np.append(Errs_k_b, np.array([np.concatenate((np.array([k]), err_k_b))]), axis=0)
    Errs_k_a = np.append(Errs_k_a, np.array([np.concatenate((np.array([k]), err_k_a))]), axis=0)
    Accs_untargeted = np.append(Accs_untargeted, np.array([np.concatenate((np.array([k]), np.array([final_acc])))]), axis=0)
    Adv_weights = np.append(Adv_weights, np.concatenate((np.array([[k]]), adv_weight), axis=1), axis=0)

Errs_k_b = np.delete(Errs_k_b, [0], axis=0)
Errs_k_a = np.delete(Errs_k_a, [0], axis=0)
Accs_untargeted = np.delete(Accs_untargeted, [0], axis=0)
Adv_weights = np.delete(Adv_weights, [0], axis=0)
print('Overall Accuracy: %.5f'% (np.mean(Accs_untargeted[:, 1])))
print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_k_b, 0.5), np.quantile(Errs_k_b, 0.9)))
print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(Errs_k_a, 0.5), np.quantile(Errs_k_a, 0.9)))

file_name = 'Attack_Results_all_fcnn_white_untargeted.mat'
savemat(file_name, {'Errors_k_b': Errs_k_b, 'Errors_k_a': Errs_k_a, 'Accuracy': Accs_untargeted, 'Adv_weights': Adv_weights})