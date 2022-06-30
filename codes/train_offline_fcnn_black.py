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


# fix random seeds
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2)


# train localization models
def Train_loc(model, dataloader_train, device, num_epochs=200):
    criterion = nn.MSELoss() #loss
    # optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.7)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    model = model.to(device)
    for Epoch in range(num_epochs):
        running_loss = 0.0
        for data in dataloader_train:
            _, pos, inputs = data  # loc_xy, features
            pos, inputs = pos.to(device), inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, pos)
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu()
        print('[%d] loss: %.6f' % (Epoch + 1, running_loss))
    print('Finished Training')
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module, '../offline/fcnn_black/FCNN_black.pth')
    else:
        torch.save(model, '../offline/fcnn_black/FCNN_black.pth')


# test localization model
def Test_loc(model, device, testdatapath, Num_classes):
    errs_all = np.array([])  # localization errors of all test samples
    errs_90_all = np.array([])  # 90% percentile errors of each position
    model = model.to(device)
    for k in range(Num_classes):
        dataset = create_dataset('FD40', testdatapath, k)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=250, shuffle=False, num_workers=16, pin_memory=True)
        errs_k = np.array([])  # localization errors of k-th position
        with torch.no_grad():
            for data in dataloader:
                _, loc_gt, in_feats = data
                loc_gt, in_feats = loc_gt.to(device), in_feats.to(device)

                loc_pred = model(in_feats)
                loc_pred[:, 0], loc_pred[:, 1] = loc_pred[:, 0]*8.0*1.5, loc_pred[:, 1]*5.0*1.5
                loc_gt[:, 0], loc_gt[:, 1] = loc_gt[:, 0]*8.0*1.5, loc_gt[:, 1]*5.0*1.5
                temp = F.pairwise_distance(loc_pred, loc_gt, p=2)
                errs_k = np.append(errs_k, temp.cpu())
        errs_all = np.append(errs_all, errs_k)
        errs_90_all = np.append(errs_90_all, np.quantile(errs_k, 0.9))
        print('[%d] 0.5 & 0.9 errors: %.5f & %.5f'% (k, np.quantile(errs_k, 0.5),  np.quantile(errs_k, 0.9)))

    pickle.dump(errs_all, open("../offline/fcnn_black/FCNN_black_meta_error_all_info.pkl", "wb"))
    pickle.dump(errs_90_all, open("../offline/fcnn_black/FCNN_black_meta_error90_info.pkl", "wb"))
    print('[Total] 0.5 & 0.9 errors: %.5f & %.5f' % (np.quantile(errs_all, 0.5), np.quantile(errs_all, 0.9)))


time_start = time.time()
Num_classes = 40
Num_epochs = 300  # number of training epochs
network = FCNN40.FCNN40_new()
network = network.double()
# network = torch.load('../offline/fcnn_black/FCNN_black.pth')

path_train = '../datas/Offline_B_down_SIMO.csv'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
network = torch.nn.DataParallel(network, device_ids=[2, 3, 1, 0])
data_train = create_dataset('FD40', path_train, "train")
dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)
Train_loc(network, dataloader_train, device, Num_epochs)

model = torch.load('../offline/fcnn_black/FCNN_black.pth')
model = model.double()
path_test = '../datas/Offline_B_up_SIMO.csv'
Test_loc(model, device, path_test, Num_classes)

time_end = time.time()
time_cost = time_end - time_start
print('Time cost', time_cost, 's')