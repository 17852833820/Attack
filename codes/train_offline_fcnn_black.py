import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import torch.optim as optim
import numpy as np
import pickle
import time
from codes.model import FCNN40, ConvCNN40
from codes.data_scripts import create_dataset
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class DNN_offine_fcnn_black():
    def __init__(self):
        self.setup_seed(2)
        self.Num_classes = 10
        self.Num_epochs = 300  # number of training epochs
        #self.network = FCNN40.FCNN40_new()
        #self.network = self.network.double()
        self.network = torch.load('../online_new/fcnn_black/FCNN_black.pth')
        self.writer= SummaryWriter('../logs/trainDNN/FCNN/black/{0}/tensorboard'.format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))))

        self.path_train = '../datas/Online_B_down_SIMO.csv'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # fix random seeds
    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True




    # train localization models
    def Train_loc(self,model, dataloader_train, device, num_epochs=200):
        criterion = nn.MSELoss() #loss
        # optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.7)
        optimizer = optim.Adadelta(model.parameters(), lr=0.1)
        model = model.to(device)
        for Epoch in range(num_epochs):
            running_loss = 0.0
            for data in dataloader_train:
                _, pos, inputs = data  # loc_xy, features
                pos, inputs = pos.to(device), inputs.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, pos)
                self.writer.add_scalar('loss', loss, Epoch )
                loss.backward()
                optimizer.step()
                running_loss += loss.cpu()
            if Epoch%50==0:
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module, '../online_new/fcnn_black/FCNN_black{0}.pth'.format(Epoch))
                else:
                    torch.save(model, '../online_new/fcnn_black/FCNN_black{0}.pth'.format(Epoch))
            self.writer.add_scalar('running_loss', running_loss, Epoch + 1)
            print('[%d] loss: %.6f' % (Epoch + 1, running_loss))
        print('Finished Training')
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module, '../online_new/fcnn_black/FCNN_black.pth')
        else:
            torch.save(model, '../online_new/fcnn_black/FCNN_black.pth')


    # test localization model
    def Test_loc(self,model, device, testdatapath, Num_classes):
        errs_all = np.array([])  # localization errors of all test samples
        errs_90_all = np.array([])  # 90% percentile errors of each position
        model = model.to(device)
        for k in range(Num_classes):
            dataset = create_dataset('FD40', testdatapath, k)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, num_workers=0, pin_memory=True)
            errs_k = np.array([])  # localization errors of k-th position
            with torch.no_grad():
                for data in dataloader:
                    _, loc_gt, in_feats = data
                    loc_gt, in_feats = loc_gt.to(device), in_feats.to(device)

                    loc_pred = model(in_feats)
                    loc_pred[:, 0], loc_pred[:, 1] = loc_pred[:, 0]*10.0*0.6, loc_pred[:, 1]*1.0*0.6
                    loc_gt[:, 0], loc_gt[:, 1] = loc_gt[:, 0]*10.0*0.6, loc_gt[:, 1]*1.0*0.6
                    temp = F.pairwise_distance(loc_pred, loc_gt, p=2)
                    errs_k = np.append(errs_k, temp.cpu())
            if len(errs_k)==0:
                print('temp None')
                continue
            errs_all = np.append(errs_all, errs_k)
            errs_90_all = np.append(errs_90_all, np.quantile(errs_k, 0.9))
            print('[%d] 0.5 & 0.9 errors: %.5f & %.5f'% (k, np.quantile(errs_k, 0.5),  np.quantile(errs_k, 0.9)))

        pickle.dump(errs_all, open("../online_new/fcnn_black/FCNN_black_meta_error_all_info.pkl", "wb"))
        pickle.dump(errs_90_all, open("../online_new/fcnn_black/FCNN_black_meta_error90_info.pkl", "wb"))
        print('[Total] 0.5 & 0.9 errors: %.5f & %.5f' % (np.quantile(errs_all, 0.5), np.quantile(errs_all, 0.9)))
    def run(self):
        time_start = time.time()

        network = torch.nn.DataParallel(self.network, device_ids=[0])
        data_train = create_dataset('FD40', self.path_train, "train")
        dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
        self.Train_loc(network, dataloader_train, self.device, self.Num_epochs)

        model = torch.load('../online_new/fcnn_black/FCNN_black.pth')
        model = model.double()
        path_test = '../datas/Online_B_up_SIMO.csv'
        self.Test_loc(model, self.device, path_test, self.Num_classes)
        time_end = time.time()
        time_cost = time_end - time_start
        print('Time cost', time_cost, 's')
if __name__ == '__main__':
    location=DNN_offine_fcnn_black()
    location.run()