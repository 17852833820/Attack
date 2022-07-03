import os
import numpy as np
import torch
import random
import torch.nn.functional as F
from codes.data_scripts import create_dataset
import torch.optim as optim
from scipy.io import savemat
import math
import pickle
from codes.model import Generator
from tensorboardX import SummaryWriter
import time
from codes.loss.loss import MyLoss1,WeightLoss
class T_offine_conv_white():
    def __init__(self):
        self.num_classes = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path_train = '../datas/Online_B_down_SIMO.csv'
        self.path_test = '../datas/Online_B_up_SIMO.csv'
        self.model = torch.load('../online_new/conv_white/ConvCNN_white.pth', map_location=torch.device('cpu'))#DNNA,train first
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        self.CNN = Generator.Generator()
        self.CNN = self.CNN.to(self.device)#GAN generator
        self.errors90_all = pickle.load(open("../online_new/conv_white/ConvCNN_white_meta_error90_info.pkl", 'rb'))
        self.date = 0.15
        self.d_max = 0.3
        self.Perdiction_b = np.empty((1, 1 + 500 * 2))
        self.Perdiction_a = np.empty((1, 1 + 500 * 2))
        self.Errs_k_b = np.empty((1, 2 + 500))
        self.Errs_n_b = np.empty((1, 2 + 500))
        self.Errs_k_a = np.empty((1, 2 + 500))
        self.Errs_n_a = np.empty((1, 2 + 500))
        self.Accs_b = np.empty((1, 2 + 1))
        self.Accs_a = np.empty((1, 2 + 1))
        self.Adv_weights = np.empty((1, 2 + 52))
        self.writer= SummaryWriter('../logs/trainGAN/T-CNN-white/{0}/tensorboard'.format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))))

        self.setup_seed(3)

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # search for nearest points outside the circle located at k-th point
    def pairing(self, k, threshold_k):
        array_k = np.array([])
        x = (k + 1)
        y = 0
        for d in np.arange(9, -1, -1):  # d=1,...10
            d_x = abs(x * 0.6 - (d + 1) * 0.6)
            d_y = abs(y - 0)
            if int(d_x * 10) == int(3 * 0.6 * 10) and len(array_k) == 0:
                array_k = np.append(array_k, d)

        return array_k


    # train adversarial network
    def Train_adv_network(self,model, network, device, train_loader, k, n, dmax, date):
        target_location = torch.tensor([(n+1) / 10.0, 0.0/1.0]).to(device)
        d_new = dmax-0.2
        model = model.to(device)
        network = network.to(device)
        for param_model in model.parameters():  # fix parameters of loc model
            param_model.requires_grad = False

        myloss1 = MyLoss1().to(device)
        myloss3 = WeightLoss().to(device)

        #optimizer = optim.SGD(network.parameters(), lr=0.5, momentum=0.5)
        optimizer = optim.Adadelta(network.parameters(), lr=0.1)

        for data in train_loader:
            _, pos, inputs = data
            pos, inputs = pos.to(device), inputs.to(device)
            loss_temp = 0.0
            alpha = 0.1
            first_loss = []
            third_loss = []
            for Epoch in range(8000):  #


                optimizer.zero_grad()
                data_per, weights = network(inputs, date)  # add perturbation
                output = model(data_per)  # location predicts

                loss1 = myloss1(output, target_location, d_new)
                loss3 = myloss3(weights)
                loss = loss1 + alpha * loss3  # total loss
                self.writer.add_scalar('train{0}-{1}/loss'.format(k, n), loss, Epoch)
                self.writer.add_scalar('train{0}-{1}/loss1'.format(k, n), loss1, Epoch)
                self.writer.add_scalar('train{0}-{1}/loss3'.format(k, n), loss3, Epoch)
                loss.backward()
                optimizer.step()
                first_loss.append(loss1.cpu().item())
                third_loss.append(loss3.cpu().item())
                print('[%d-%d][%d] First loss & Third loss: %.6f & %.6f alpha %6f' %
                      (k, n, Epoch + 1, loss1, loss3, alpha))
                '''if abs(max(first_loss) - loss_temp) <= 0.00001 and d_new >= dmax / 5.0:  # 控制阈值，使其更加小，以产生更多满足原始阈值的数据，提高准确率
                    d_new = d_new / 1.05
                if Epoch > 100 and max(first_loss) <= 0.1 and max(third_loss) <= 0.1:  #
                    break
                loss_temp = max(first_loss)'''
                if loss1 <= 0.01 and loss3 <= 0.01:
                    break
                if loss1 <= 0.1 and loss3 >= 0.1:  # 动态改变权重。前期可将alpha=0.1，重要优化攻击精度。精度达到上限之后，逐渐增大alpha，是的gamma更加平滑
                    alpha = 30.0
                else:
                    alpha = 0.001
                if Epoch == 4000:
                    mean_first = np.mean(first_loss)
                    std_first = np.std(first_loss)
                if Epoch == 6000 and mean_first - 2 * std_first <= loss1.cpu() <= mean_first + 2 * std_first:
                    alpha = 200.0

        if isinstance(network, torch.nn.DataParallel):
            torch.save(network.module, '../online_new/adv_conv_white/adv_white_conv' + '%d-' % k + '%d' % n + '.pth')
        else:
            torch.save(network, '../online_new/adv_conv_white/adv_white_conv' + '%d-' % k + '%d' % n + '.pth')
        return network


    def Test_adv_network(self,model, network, device, test_loader, k, n, dmax, date):  # model: Loc model, CNN: perturbation model, K: original location, n: targeted location, dmax：threshold
        err_k_b = np.array([])  # localization errors to original location before perturbation
        err_n_b = np.array([])  # localization errors to targeted location before perturbation
        err_k_a = np.array([])  # localization errors to original location after perturbation
        err_n_a = np.array([])  # localization errors to targeted location after perturbation
        loc_prediction_b = np.array([])
        loc_prediction_a = np.array([])
        target_location = torch.tensor([(n+1) * 0.6, 0.0]).to(device)#target location modify
        model = model.to(device)
        network = network.to(device)
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


    def run(self):
        for k in np.arange(self.num_classes):
            data_train = create_dataset('FD', self.path_train, k)
            data_test = create_dataset('FD', self.path_test, k)
            dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=500, shuffle=True, num_workers=0,
                                                           pin_memory=True)
            dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=500, shuffle=False, num_workers=0,
                                                          pin_memory=True)
            threshold_k = self.errors90_all[k] + self.d_max
            list_k = self.pairing(k, threshold_k)
            for n in list_k:
                t1 = time.time()

                # network = torch.load('../offline/adv_conv_white/adv_white_conv' + '%d-' % k + '%d' % n + '.pth')
                network = self.Train_adv_network(self.model, self.CNN, self.device, dataloader_train, k, n, self.d_max, self.date)
                t2 = time.time()
                print("time{0}".format(t2-t1))
                _, _, err_k_b, err_k_a, err_n_b, err_n_a, final_acc_b, final_acc_a, adv_weight, loc_prediction_b, loc_prediction_a = self.Test_adv_network(
                    self.model, network, self.device, dataloader_test, k, n, self.d_max, self.date)
                self.Errs_k_b = np.append(self.Errs_k_b, np.array([np.concatenate((np.array([k, n]), err_k_b))]), axis=0)
                self.Errs_n_b = np.append(self.Errs_n_b, np.array([np.concatenate((np.array([k, n]), err_n_b))]), axis=0)
                self.Errs_k_a = np.append(self.Errs_k_a, np.array([np.concatenate((np.array([k, n]), err_k_a))]), axis=0)
                self.Errs_n_a = np.append(self.Errs_n_a, np.array([np.concatenate((np.array([k, n]), err_n_a))]), axis=0)
                self.Accs_b = np.append(self.Accs_b, np.array([np.concatenate((np.array([k, n]), np.array([final_acc_b])))]),
                                   axis=0)
                self.Accs_a = np.append(self.Accs_a, np.array([np.concatenate((np.array([k, n]), np.array([final_acc_a])))]),
                                   axis=0)
                self.Adv_weights = np.append(self.Adv_weights, np.concatenate((np.array([[k, n]]), adv_weight), axis=1), axis=0)
                self.Perdiction_a = np.append(self.Perdiction_a, np.array([np.concatenate((np.array([k]), loc_prediction_a))]),
                                         axis=0)
                self.Perdiction_b = np.append(self.Perdiction_b, np.array([np.concatenate((np.array([k]), loc_prediction_b))]),
                                         axis=0)

        self.Errs_k_b = np.delete(self.Errs_k_b, [0], axis=0)
        self.Errs_n_b = np.delete(self.Errs_n_b, [0], axis=0)
        self.Errs_k_a = np.delete(self.Errs_k_a, [0], axis=0)
        self.Errs_n_a = np.delete(self.Errs_n_a, [0], axis=0)
        self.Accs_b = np.delete(self.Accs_b, [0], axis=0)
        self.Accs_a = np.delete(self.Accs_a, [0], axis=0)
        self.Adv_weights = np.delete(self.Adv_weights, [0], axis=0)
        self.Prediction_b = np.delete(self.Prediction_b,[0],axis=0)
        self.Prediction_a = np.delete(self.Prediction_a,[0],axis=0)

        print('Overall Accuracy Before and After: %.5f & %.5f' % (np.mean(self.Accs_b[:, 2]), np.mean(self.Accs_a[:, 2])))
        print('Before Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(self.Errs_k_b[:, 2:252], 0.5), np.quantile(self.Errs_k_b[:, 2:252], 0.9)))
        print('After Error_k 0.5 & 0.9: %.5f & %.5f' % (np.quantile(self.Errs_k_a[:, 2:252], 0.5), np.quantile(self.Errs_k_a[:, 2:252], 0.9)))
        print('Before Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(self.Errs_n_b[:, 2:252], 0.5), np.quantile(self.Errs_n_b[:, 2:252], 0.9)))
        print('After Error_n 0.5 & 0.9: %.5f & %.5f' % (np.quantile(self.Errs_n_a[:, 2:252], 0.5), np.quantile(self.Errs_n_a[:, 2:252], 0.9)))

        file_name = '../online_new/conv_white/Attack_Results_all_conv_white_new.mat'
        savemat(file_name, {'Errors_k_b': self.Errs_k_b, 'Errors_n_b': self.Errs_n_b, 'Errors_k_a': self.Errs_k_a, 'Errors_n_a': self.Errs_n_a, 'Accuracy_before': self.Accs_b, 'Accuracy_after': self.Accs_a, 'Adv_weights': self.Adv_weights,'Prediction_b':self.Prediction_b,"Prediction_a":self.Prediction_a})
if __name__ == '__main__':
    attacker=T_offine_conv_white()
    attacker.run()
