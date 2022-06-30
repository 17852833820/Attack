import pickle
from pandas import Series, DataFrame

path="../online_results/FCNN_results_untargeted.pkl"
results=pickle.load(open(path,"rb"))
err_50=results['50%_all']
err_90=results['90%_all']
err_mean=results['所有样本平均定位误差']
err_spot=results['每个点的平均定位误差']
err_cdf=results['每个样本的定位误差']
err=[]
for i in err_cdf:
    for j in i:
        err.append(j)
acc=results['攻击成功率']
acc_spot=results['每个点的平均攻击成功率']
acc_50=results['50%_acc']
acc_90=results['90%_acc']
d=[]
d.append(err_50)
d.append(err_90)
d.append(err_mean)
d.append(err_spot)
d.append(err)
d.append(err_cdf)
d.append(acc)
d.append(acc_spot)
d.append(acc_50)
d.append(acc_90)
dataframe = DataFrame(d)
dataframe.to_csv("../online_results/FCNN_results_untargeted.csv")
