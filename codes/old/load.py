import numpy as np
import scipy.io
import torch
import torch.nn as nn
import numpy
import scipy
import torch.nn.functional as F
from codes.data_scripts import create_dataset
import matplotlib.pyplot as plt
import torch.optim as optim
import pickle
from pandas import Series, DataFrame

import statsmodels.api as sm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_CSI(data,data_per,ave0,ave1):
    amp1 = []  #
    amp1_per = []
    amp2 = []
    amp2_per = []
    d1 = []
    d2 = []
    d = []  # 210,112
    for i, j in zip(ave0, ave1):  # 210,1,2,56
        for x, y in zip(i, j):
            x=np.array(x.tolist())
            y=np.array(y.tolist())
            x=x.reshape(104)
            y=y.reshape(104)
            d.append(np.hstack((x,y)))
    dataframe = DataFrame(d)
    dataframe.to_csv("../ON_RESULTS/ConvCNN_inputs_untargeted[%d].csv" % (k))
    x_axix = list(range(0, 52))
    f = plt.figure()
    ax1 = f.add_subplot(211)
    for i, j in zip(data, data_per):  # 50,1,2,56
        for x in i:  # 2，,56
            amp1 = (x[0, :].tolist())
            amp2 = (x[1, :].tolist())

        for y in j:
            amp1_per = (y[0, :].tolist())
            amp2_per = (y[1, :].tolist())
    d1.append(amp1)  # 2,56
    d1.append(amp1_per)
    d2.append(amp2)
    d2.append(amp2_per)

    data0 = DataFrame(np.array(d1).reshape(52, 2), columns=list('12'))
    plt.title('The CSI(Amplitude):')
    # ax1.legend(('raw', 'per'), loc='best', handlelength=15, shadow=True)
    plt.ylim(0, 1)
    ax1.plot(data0, alpha=0.9)
    ax2 = f.add_subplot(212)
    data1 = DataFrame(np.array(d2).reshape(52, 2), columns=list('12'))
    ax2.set_xlabel("# of subcarriers")
    plt.ylim(0, 1)
    ax2.plot(data1, alpha=0.9)
    plt.savefig("CSI_targeted1_[%d].png" % (k))
    plt.show()
    dataframe1 = DataFrame(d1)
    dataframe1.to_csv("../ON_RESULTS/ConvCNN1_untargeted[%d].csv"% (k))
    dataframe2 = DataFrame(d2)
    dataframe2.to_csv("../ON_RESULTS/ConvCNN2_untargeted[%d].csv"% (k))
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear_weight = nn.Parameter(torch.ones(1, 52))

    def forward(self, x):
        x = torch.mul(torch.kron(torch.ones(2, 1).to(device), self.linear_weight), x)  #50,1,2,56
        return torch.clamp(x, min=0, max=1)


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        print("1")
    def forward(self, pred, pos,TARGETED,dmax,dmin):
            #k=F.pairwise_distance(pred, pos, p=2)
            if TARGETED==0:
                pos=torch.kron(torch.ones(pred.size()[0], 1).to(device), pos.view(1, 2))

            pred[:, 0], pred[:, 1] = pred[:, 0]*10.0, pred[:, 1]*10.0
            pos[:,0],pos[:,1]=pos[:,0]*10.0,pos[:,1]*10.0
            temp = F.pairwise_distance(pred, pos, p=2)
            m = dmin-temp
            n = temp-dmax
            return torch.mean((1 - TARGETED) * nn.ReLU()(n) + TARGETED * nn.ReLU()(m))


def test(model,CNN,device,test_loader,TARGETED,k,dmax,dmin):
    correct=0.0
    err=[]
    num=0
    ave0=[]
    ave1=[]
    with torch.no_grad():

        for Data in test_loader:
            _, pos, inputs = Data
            # label, loc_xy, features
            pos, inputs = pos.to(device), inputs.to(device)
            #data_per = CNN(inputs)  # add perturbation
            #save_per.append(data_per.tolist())
            #for i in inputs:
            #    ave0.append(i)
            #for i in data_per:
            #    ave1.append(i)
            #output=model(data_per)   #predict
            #predict=model(inputs)
            output = model(inputs)
            output[:,0], output[:,1] = output[:,0] * 10.0, output[:,1] * 10.0
            pos[:,0], pos[:,1] = pos[:,0] * 10.0, pos[:,1] * 10.0
            #predict[:,0], predict[:,1] = predict[:,0] * 10.0, predict[:,1] * 10.0

            if TARGETED == 1:  # no
                #print(F.pairwise_distance(predict, pos, p=2))
                print(F.pairwise_distance(output, pos, p=2))

                correct = correct+torch.count_nonzero(torch.ge(F.pairwise_distance(output, pos, p=2), dmin).type(torch.uint8)).item()
            elif TARGETED == 0:
                temp = F.pairwise_distance(output, Q*torch.tensor([10.0, 10.0]).to(device), p=2)
                correct = correct+torch.count_nonzero(torch.le(temp, dmax).type(torch.uint8)).item()

            err.append(torch.mean(torch.norm(output - pos, p=2, dim=1)*0.5).item())

            num=num+inputs.size()[0]
    final_acc=correct/float(num)
    print('[%d] Localization error: %.5f' % (k + 1, sum(err)/ len(err)))
    #绘制该点的某一个样本数据的CSI幅值
    #plot_CSI(inputs, data_per,ave0,ave1)

    return final_acc,err
MAX_ITERATIONS = 100 # number of iterations to perform gradient
LEARNING_RATE = 1e-1  # larger values converge faster to less accurate results
TARGETED = 1        # should we target one specific class? or just be wrong?
path='../datas/B_DOWN_ALL.csv'
display_step = 50
Num_classes = 10
errs = []
accs = []
W_sets = {}
Q=torch.from_numpy(np.array([5.0/10.0,0.0/10.0])).to(device)   #归一化坐标
myloss = MyLoss()
model=torch.load('../online/model_FCNN_lr=1.0.pth')
model=model.double()
#modelA = torch.load('../online/model_ConvCNN_BLACK_BA_lr=1.0.pth')
info = pickle.load(open("../online/FCNN_meta_info.pkl", 'rb'))
#infoA=pickle.load(open("ConvCNN_BLACK_BA_meta_info.pkl", 'rb'))
batch_size=50
err_cdf=[]
save_per=[]
for k in range(Num_classes):

    W_sets[k]=[]
    CNN=torch.load( "../ON_RESULTS/modelFCNN_untargeted_"+'%d'%k+".pth")

    #data_attack = create_dataset('FD', path, k)  # 17000,115
    #date = 0.5
    #CNN = Generator().to(device)
    optimizer = optim.SGD(CNN.parameters(), lr=LEARNING_RATE, momentum=0.7)
    #optimizer=optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20,T_mult=1)
    #d_max=(info['50%_all']*0.9)/0.5
    d_max = 0.9*1.2
    d_min=1.1*(info['90%_samples'][k])/0.5
    #d_maxA=0.9*(infoA['50%_samples'][k])/0.5
    #d_minA=1.1*(infoA['90%_samples'][k])/0.5
    #dataloader_attack = torch.utils.data.DataLoader(data_attack, batch_size=batch_size, shuffle=False, num_workers=0)

    #torch.save(CNN,"../ON_RESULTS/modelFCNN_untargeted_"+'%d'%k+".pth")
    for param in CNN.parameters():
        W_sets[k] = param
        save_per.append(param.tolist())
    path_test='../online_results/FCNN_UNTARGET_IUP_ALL.csv'
    data_attack_TEST = create_dataset('FD', path_test, k)  # 17000,115
    dataloader_attack_TEST = torch.utils.data.DataLoader(data_attack_TEST, batch_size=1, shuffle=False, num_workers=0)

    acc,err=test(model,CNN,device,dataloader_attack_TEST,TARGETED,k,d_max/0.9,d_min/1.1)
    err_cdf.append(err)

    errs.append(sum(err)/len(err))
    accs.append(acc)
dataframe = DataFrame(save_per)
#dataframe.to_csv("../ON_RESULTS/ConvCNN_per_targeted.csv" )

#绘制所有样本点的误差cdf
ecdf = sm.distributions.ECDF(errs)
x = np.linspace(min(errs), max(errs))
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
plt.savefig("../ON_RESULTS/FCNN_err_untargted_all.png")
plt.show()

#绘制成功率的cdf图
ecdf = sm.distributions.ECDF(accs)
x = np.linspace(min(accs), max(accs))
y = ecdf(x)
ax = plt.figure().add_subplot(111)
ax.plot(x, y)
ax.hlines(0.5, x[0], x[-1:],
              linestyles='-.', colors='red')
ax.hlines(0.9, x[0], x[-1:],
              linestyles='-.', colors='orange')
ax.set_title('The CDF of all samples')
key3_all = []
key4_all = []
for i, j in zip(x, y):
        if j - 0.5 > 0.01:
            key3_all.append(i)
        if j - 0.9 > 0.01:
            key4_all.append(i)
    # plt.step(x, y)
plt.savefig("../ON_RESULTS/FCNN_acc_untargted_all.png")
plt.show()


results={}
#所有样本点的误差的cdf图
results['50%_all']=key1_all[0]
results['90%_all']=key2_all[0]
results['所有样本平均定位误差']=sum(errs)/len(errs)
results['每个点的平均定位误差']=errs
results['每个样本的定位误差']=err_cdf

results['攻击成功率']=sum(accs)/len(accs)
results['每个点的平均攻击成功率']=accs
results['50%_acc']=key3_all[0]
results['90%_acc']=key4_all[0]
print(results)
print('无目标攻击扰动参数')
for param in CNN.parameters():
    results['param']=param
    print(param)
print('所有样本平均定位误差')
print(sum(errs)/len(errs))
#每个点的平均定位误差
print('每个点的平均定位误差')
print(errs)
print('攻击成功率')
print(sum(accs)/len(accs))
pickle.dump(results, open("../ON_RESULTS/FCNN_results_untargeted.pkl", "wb"))

