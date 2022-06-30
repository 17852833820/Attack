import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import torch.optim as optim
import numpy as np
import pickle
from codes.model import VTCNN, FCNN, ConvCNN
from codes.data_scripts import create_dataset
import statsmodels.api as sm
import matplotlib.pyplot as plt


def train(CNN,dataloader_train,device,num_epochs=20):
    # CNN = torch.load('./Baseline_CNN48_amp.pkl')
    criterion = nn.HuberLoss() #loss
    optimizer = optim.SGD(CNN.parameters(), lr=0.5,momentum=0.7)
    CNN.to(device)
    #train
    for Epoch in range(num_epochs):

        running_loss = 0.0
        total = 0

        for Data in dataloader_train:
            _, pos, inputs = Data   #label, loc_xy, features
            pos, inputs = pos.to(device), inputs.to(device)

            optimizer.zero_grad()
            outputs = CNN(inputs)   #输入【200,1,3,56】
            loss = criterion(outputs, pos)    #输出【200,2】
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # total += labels.size(0)
            total += 1

        print('[%d] loss: %.6f' %
            (Epoch + 1, running_loss / total))

    print('Finished Training')
    #torch.save(CNN,'./Baseline_CNN48_amp.pkl')
    torch.save(CNN, '../online/model_ConvCNN_BLACK_lr=1.0.pth')


def test_models(model,device,testdatapath,Num_classes):

    errs = []
    errs_cdf=[]
    keys1 = []
    keys2 = []
    for k in range(Num_classes):
        dataset_odd = create_dataset('FD', testdatapath, k)
        dataloader_odd = torch.utils.data.DataLoader(dataset_odd, batch_size=1, shuffle=False, num_workers=0)
        key1 = []
        key2 = []
        err = []
        with torch.no_grad():
            for data in dataloader_odd:
               _, loc_gt, in_feats = data
               loc_gt, in_feats=  loc_gt.to(device), in_feats.to(device)

               loc_pred = model(in_feats)  #1,1，2，56
               loc_pred[:,0],loc_pred[:,1]=loc_pred[:,0]*10.0,loc_pred[:,1]*10.0
               loc_gt[:,0],loc_gt[:,1]=loc_gt[:,0]*10.0,loc_gt[:,1]*10.0
               err.append(torch.mean(torch.norm(loc_pred-loc_gt, p=2, dim=1)*0.5).item())
               errs_cdf.append(torch.mean(torch.norm(loc_pred-loc_gt, p=2, dim=1)*0.5).item())
        ecdf = sm.distributions.ECDF(err)
        x = np.linspace(min(err), max(err))
        y = ecdf(x)
        ax1 = plt.figure().add_subplot(111)
        ax1.plot(x, y)
        ax1.hlines(0.5, x[0], x[-1:],
                  linestyles='-.', colors='red')
        ax1.hlines(0.9, x[0], x[-1:],
                  linestyles='-.', colors='orange')
        for i, j in zip(x, y):
            if j - 0.5 > 0.01:
                key1.append(i)
            if j - 0.9 > 0.01:
                key2.append(i)
        keys1.append(key1[0])
        keys2.append(key2[0])
        ax1.set_title('The CDF  of the number %d spot' % (k))
        plt.savefig("../online/ConvCNN_CDF_[%d].png"% (k))
        plt.show()
        print('[%d] Localization error: %.5f' % (k + 1, sum(err)/len(err)))
        errs.append(sum(err)/len(err))
    print('Overall localization error: %.3f' % (sum(errs)/len(errs)))
    ecdf = sm.distributions.ECDF(errs_cdf)
    x = np.linspace(min(errs_cdf), max(errs_cdf))
    y = ecdf(x)
    ax = plt.figure().add_subplot(111)
    ax.plot(x, y)
    ax.hlines(0.5, x[0], x[-1:],
              linestyles='-.', colors='red')
    ax.hlines(0.9, x[0], x[-1:],
              linestyles='-.', colors='orange')
    ax.set_title('The CDF of all samples')
    key1_all = []
    key2_all = []
    for i, j in zip(x, y):
        if j - 0.5 > 0.01:
            key1_all.append(i)
        if j - 0.9 > 0.01:
            key2_all.append(i)
    # plt.step(x, y)
    plt.savefig("../online/ConvCNN_CDF_all.png")
    plt.show()
    info={}
    info['50%_all']=key1_all[0]
    info['90%_all']=key2_all[0]
    info['50%_samples']=keys1
    info['90%_samples']=keys2
    #pickle.dump(info, open("../online/FCNN_BLACK_meta_info.pkl", "wb"))
    pickle.dump(info, open("../online/ConvCNN_meta_info.pkl", "wb"))
    #select_info = pickle.load(open("D:/PycharmFiles/Adversarial_attack/results/meta_info.pkl", 'rb'))
    #print(select_info)



Num_classes = 10
CNN = ConvCNN.ConvCNN()
#CNN= torch.load('../online/model_FCNN_BLACK_lr=1.0.pth')
CNN = CNN.double()

train_data='../datas/B_DOWN_ALL.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_train = create_dataset('FD',train_data, "train") #17000,115
dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=200, shuffle=True, num_workers=0)
#train(CNN,dataloader_train,device)

model = torch.load('../online/model_ConvCNN_lr=1.0.pth')
model=model.double()
testdatapath='../datas/B_UP_ALL.csv'
test_models(CNN,device,testdatapath,Num_classes)
