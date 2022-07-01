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
import time
from codes.model import Generator
from tensorboardX import SummaryWriter
from codes.loss.loss import MyLoss2,WeightLoss
class UT_offine_conv_white():
    def __init__(self):
        # 设置随机数种子
        self.setup_seed(3)
        self.num_classes = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path_train = '../datas/Online_B_down_SIMO.csv'
        self.path_test = '../datas/Online_B_up_SIMO.csv'
        # model = torch.load('../offline/conv_white/ConvCNN_white.pth', map_location=torch.device('cpu'))
        self.model = torch.load('../online/conv_white/ConvCNN_white.pth')
        self.CNN = Generator.Generator()

        self.errors90_all = pickle.load(open("../online/conv_white/ConvCNN_white_meta_error90_info.pkl", 'rb'))
        self.date = 0.15

        self.Errs_k_b = np.empty((1, 1+500))
        self.Errs_k_a = np.empty((1, 1+500))
        self.Accs_b = np.empty((1, 1 + 1))
        self.Accs_a = np.empty((1, 1 + 1))
        self.Adv_weights = np.empty((1, 1 + 52))
        self.Perdiction_b = np.empty((1, 1+500*2))
        self.Perdiction_a = np.empty((1, 1+500*2))
        self.writer= SummaryWriter('./logs/trainGAN/UT-CNN-white/{0}/tensorboard'.format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))))

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # train adversarial network
    def Train_adv_network(self,model, network, device, train_loader, k, dmin, date):
        original_location = torch.tensor([(k // 5 + 1) / 8, (k % 5 + 1) / 5]).to(device)
        d_new = 5*dmin
        model = model.to(device)
        network = network.to(device)
        for param_model in model.parameters():  # fix parameters of loc model
            param_model.requires_grad = False

        myloss2 = MyLoss2().to(device)
        myloss3 = WeightLoss().to(device)
        # optimizer = optim.SGD(network.parameters(), lr=0.5, momentum=0.5)
        optimizer = optim.Adadelta(network.parameters(), lr=0.5)
        for data in train_loader:
            _, pos, inputs = data
            loss_temp = 0.0
            alpha = 1.0
            pos, inputs = pos.to(device), inputs.to(device)
            for Epoch in range(2000):  #
                second_loss = []
                third_loss = []
                optimizer.zero_grad()
                data_per, weights = network(inputs, date)  # add perturbation
                output = model(data_per)  # location predicts

                loss2 = myloss2(output, original_location, d_new)
                loss3 = myloss3(weights)
                loss = loss2 + alpha * loss3
                loss.backward()
                optimizer.step()
                second_loss.append(loss2.cpu())
                third_loss.append(loss3.cpu())
                print('[%d][%d] Second loss and third loss:  %.6f & %.6f' %
                      (k, Epoch + 1, max(second_loss), max(third_loss)))
                if abs(max(
                        second_loss) - loss_temp) <= 0.000001 and d_new <= 5 * dmin:  # 控制阈值，使其更加大，以产生更多满足原始阈值的数据，提高准确率
                    d_new = d_new * 1.05
                if Epoch > 100 and max(second_loss) <= 0.1 and max(
                        third_loss) <= 0.1:  # 动态改变权重。前期可将alpha=0.1，重要优化攻击精度。精度达到上限之后，逐渐增大alpha，是的gamma更加平滑
                    break
                loss_temp = max(second_loss)
                if max(second_loss) <= 0.1 and max(third_loss) >= 0.1:
                    alpha = 30.0
                else:
                    alpha = 1.0

        if isinstance(network, torch.nn.DataParallel):
            torch.save(network.module, '../online/adv_conv_white/ut_adv_white_conv' + '%d-' % k + '.pth')
        else:
            torch.save(network, '../online/adv_conv_white/ut_adv_white_conv' + '%d-' % k + '.pth')
        return network


    def Test_adv_network(self,model, network, device, test_loader, k, dmin, date):  # model: Loc model, CNN: perturbation model, K: original location, dmin：threshold
        err_k_b = np.array([])  # localization errors to original location before perturbation
        err_k_a = np.array([])  # localization errors to original location after perturbation
        loc_prediction_b = np.array([])  # location estimation before attack
        loc_prediction_a = np.array([])  # location estimation after attack
        with torch.no_grad():
            for data in test_loader:
                _, pos, inputs = data
                pos, inputs = pos.to(device), inputs.to(device)
                data_per, adv_weight = network(inputs, date)  # add perturbation
                output = model(data_per)  # perturbed results
                predict = model(inputs)  # genuine results

                output[:, 0], output[:, 1] = output[:, 0] * 10.0 * 0.6, output[:, 1] * 1.0 * 0.6
                pos[:, 0], pos[:, 1] = pos[:, 0] * 10.0 * 0.6, pos[:, 1] * 1.0 * 0.6
                predict[:, 0], predict[:, 1] = predict[:, 0] * 10.0 * 0.6, predict[:, 1] * 1.0 * 0.6

                temp_k_b = F.pairwise_distance(predict, pos, p=2)  # localization errors
                temp_k_a = F.pairwise_distance(output, pos, p=2)

                err_k_b = np.append(err_k_b, temp_k_b.cpu())
                err_k_a = np.append(err_k_a, temp_k_a.cpu())
                loc_prediction_b = np.append(loc_prediction_b, predict.cpu())
                loc_prediction_a = np.append(loc_prediction_a, output.cpu())
        final_acc_a = np.sum(err_k_a >= dmin) / err_k_a.shape[0]
        final_acc_b = np.sum(err_k_b >= dmin) / err_k_b.shape[0]
        print('【%d】' % k)
        print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_b, 0.5), np.quantile(err_k_b, 0.9)))
        print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(err_k_a, 0.5), np.quantile(err_k_a, 0.9)))
        print(' Before and After Attack accuracy: %.5f' % final_acc_b, final_acc_a)
        return k, err_k_b, err_k_a, final_acc_b, final_acc_a, adv_weight.cpu(), loc_prediction_b, loc_prediction_a
    def run(self):
        for k in np.arange(self.num_classes):
            data_train = create_dataset('FD', self.path_train, k)
            data_test = create_dataset('FD', self.path_test, k)
            dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=500, shuffle=True)
            dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=500, shuffle=False)
            d_min = self.errors90_all[k] + 0.3

            # network = torch.load('../online/adv_conv_white/ut_adv_white_conv' + '%d-' % k + '.pth')
            network = self.Train_adv_network(self.model, self.CNN, self.device, dataloader_train, k, d_min, self.date)
            _, err_k_b, err_k_a, final_acc_b, final_acc_a, adv_weight, loc_prediction_b, loc_prediction_a = self.Test_adv_network(
                self.model, network, self.device, dataloader_test, k, d_min, self.date)
            self.Errs_k_b = np.append(self.Errs_k_b, np.array([np.concatenate((np.array([k]), err_k_b))]), axis=0)
            self.Errs_k_a = np.append(self.Errs_k_a, np.array([np.concatenate((np.array([k]), err_k_a))]), axis=0)
            self.Accs_b = np.append(self.Accs_b, np.array([np.concatenate((np.array([k]), np.array([final_acc_b])))]), axis=0)
            self.Accs_a = np.append(self.Accs_a, np.array([np.concatenate((np.array([k]), np.array([final_acc_a])))]), axis=0)
            self.Adv_weights = np.append(self.Adv_weights, np.concatenate((np.array([[k]]), adv_weight), axis=1), axis=0)
            self.Perdiction_a = np.append(self.Perdiction_a, np.array([np.concatenate((np.array([k]), loc_prediction_a))]),
                                     axis=0)
            self.Perdiction_b = np.append(self.Perdiction_b, np.array([np.concatenate((np.array([k]), loc_prediction_b))]),
                                     axis=0)

        self.Errs_k_b = np.delete(self.Errs_k_b, [0], axis=0)
        self.Errs_k_a = np.delete(self.Errs_k_a, [0], axis=0)
        self.Accs_b = np.delete(self.Accs_b, [0], axis=0)
        self.Accs_a = np.delete(self.Accs_a, [0], axis=0)
        self.Adv_weights = np.delete(self.Adv_weights, [0], axis=0)
        self.Perdiction_b = np.delete(self.Perdiction_b, [0], axis=0)
        self.Perdiction_a = np.delete(self.Perdiction_a, [0], axis=0)
        print('Overall Accuracy Before and After: %.5f & %.5f' % (np.mean(self.Accs_b[:, 1]), np.mean(self.Accs_a[:, 1])))
        print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(self.Errs_k_b[:, 1:251], 0.5), np.quantile(self.Errs_k_b[:, 1:251], 0.9)))
        print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(self.Errs_k_a[:, 1:251], 0.5), np.quantile(self.Errs_k_a[:, 1:251], 0.9)))

        file_name = '../offline/conv_white/ut_Attack_Results_all_conv_white_new.mat'
        savemat(file_name, {'Errors_k_b': self.Errs_k_b, 'Errors_k_a': self.Errs_k_a, 'Accuracy_before': self.Accs_b, 'Accuracy_after': self.Accs_a, 'Adv_weights': self.Adv_weights , 'Perdiction_b': self.Perdiction_b, 'Perdiction_a': self.Perdiction_a})
'''if __name__ == '__main__':
    attacker=UT_offine_conv_white()
    attacker.run()'''