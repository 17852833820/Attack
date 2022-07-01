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
from tensorboardX import SummaryWriter
from codes.loss.loss import MyLoss2,WeightLoss
import time
class UT_offine_fcnn_black():
    # 设置随机数种子
    def __init__(self):
        # self.setup_seed(3)
        self.num_classes = 40
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path_train = '../datas/old6.30/Offline_B_down_SIMO.csv'
        self.path_test = '../datas/old6.30/Offline_B_up_SIMO.csv'
        self.model_surrogate = torch.load('../offline/conv_black/ConvCNN_black.pth', map_location=torch.device('cpu'))
        self.model_victim = torch.load('../offline/fcnn_white/FCNN_white.pth', map_location=torch.device('cpu'))
        self.CNN = Generator.Generator()
        self.CNN_random = Generator.Generator1()

        self.errors90_all = pickle.load(open("../offline/fcnn_white/FCNN_white_meta_error90_info.pkl", 'rb'))
        self.date = 0.15

        self.Errs_k_b = np.empty((1, 1 + 250))
        self.Errs_k_a = np.empty((1, 1 + 250))
        self.Accs_b = np.empty((1, 1 + 1))
        self.Accs_a = np.empty((1, 1 + 1))
        self.Adv_weights = np.empty((1, 1 + 56))
        self.Perdiction_b = np.empty((1, 1 + 500))
        self.Perdiction_a = np.empty((1, 1 + 500))
        self.writer= SummaryWriter('./logs/trainGAN/UT-FCNN-black/{0}/tensorboard'.format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))))

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # train adversarial network
    def Train_adv_network(self,model, network, device, train_loader, k, dmin, date):
        original_location = torch.tensor([(k // 5 + 1)/8, (k % 5 + 1)/5]).to(device)
        d_new = dmin
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
                loss = loss2 + alpha*loss3
                self.writer.add_scalar('train/loss', loss, Epoch)
                self.writer.add_scalar('train/loss2', loss2, Epoch)
                self.writer.add_scalar('train/loss3', loss3, Epoch)
                loss.backward()
                optimizer.step()
                second_loss.append(loss2.cpu())
                third_loss.append(loss3.cpu())
                print('[%d][%d] Second loss and third loss:  %.6f & %.6f' %
                      (k, Epoch + 1,  max(second_loss), max(third_loss)))
                if abs(max(second_loss)-loss_temp) <= 0.000001 and d_new <= 3*dmin:
                    d_new = d_new*1.05
                if Epoch > 100 and max(second_loss) <= 0.05 and max(third_loss) <= 0.1:
                    break
                loss_temp = max(second_loss)
                if max(second_loss) <= 0.05 and max(third_loss) >= 0.1:
                    alpha = 30.0
                else:
                    alpha = 1.0

        if isinstance(network, torch.nn.DataParallel):
            torch.save(network.module, '../offline/adv_fcnn_black/ut_adv_black_fcnn_new' + '%d-' % k + '.pth')
        else:
            torch.save(network, '../offline/adv_fcnn_black/ut_adv_black_fcnn_new' + '%d-' % k + '.pth')
        return network


    def Test_adv_network(self,model, network, device, test_loader, k, dmin, date):  # model: Loc model, CNN: perturbation model, K: original location, dmin：threshold
        err_k_b = np.array([])  # localization errors to original location before perturbation
        err_k_a = np.array([])  # localization errors to original location after perturbation
        loc_prediction_b = np.array([])
        loc_prediction_a = np.array([])
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
            data_train = create_dataset('FD40', self.path_train, k)
            data_test = create_dataset('FD40', self.path_test, k)
            dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=250, shuffle=True)
            dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=250, shuffle=True)
            d_min = self.errors90_all[k] + 0.3

            network = torch.load('../offline/adv_fcnn_black/ut_adv_black_fcnn_new' + '%d-' % k + '.pth')
            network = self.Train_adv_network(model_surrogate, network, self.device, dataloader_train, k, d_min, date)
            _, err_k_b, err_k_a, final_acc_b, final_acc_a, adv_weight, loc_prediction_b, loc_prediction_a = self.Test_adv_network(model_victim, network, device, dataloader_test, k, d_min, date)
            self.Errs_k_b = np.append(self.Errs_k_b, np.array([np.concatenate((np.array([k]), err_k_b))]), axis=0)
            self.Errs_k_a = np.append(self.Errs_k_a, np.array([np.concatenate((np.array([k]), err_k_a))]), axis=0)
            self.Accs_b = np.append(self.Accs_b, np.array([np.concatenate((np.array([k]), np.array([final_acc_b])))]), axis=0)
            self.Accs_a = np.append(self.Accs_a, np.array([np.concatenate((np.array([k]), np.array([final_acc_a])))]), axis=0)
            self.Adv_weights = np.append(self.Adv_weights, np.concatenate((np.array([[k]]), adv_weight), axis=1), axis=0)
            self.Perdiction_a = np.append(self.Perdiction_a,np.array([np.concatenate((np.array([k]), loc_prediction_a))]), axis=0)
            self.Perdiction_b = np.append(self.Perdiction_b,np.array([np.concatenate((np.array([k]), loc_prediction_b))]), axis=0)
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


        file_name = '../offline/conv_white/ut_Attack_Results_all_fcnn_black_new.mat'
        savemat(file_name, {'Errors_k_b': self.Errs_k_b, 'Errors_k_a': self.Errs_k_a, 'Accuracy_before': self.Accs_b, 'Accuracy_after': self.Accs_a, 'Adv_weights': self.Adv_weights , 'Perdiction_b': self.Perdiction_b, 'Perdiction_a': self.Perdiction_a})
'''if __name__ == '__main__':
    attacker=UT_offine_fcnn_black()
    attacker.run()'''