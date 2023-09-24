from Finetune_hep.python import ParT_mlp
from Finetune_hep.python import Mlp
from Finetune_hep.python import definitions as df
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import roc_curve, auc
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys


fpr_mlpHlXbb=[]
auc_mlpHlXbb=[]
fpr_ParTevent=[]
auc_ParTevent=[]

for i in range(5):

    file = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTevent/test_ParTevent_score_training_{i+1}.h5'
    #target_ParTevent = data['evt_score']
    #yi_ParTevent = data['evt_label']
    with h5py.File(filename, 'r') as Data:
        target_ParTevent = Data['Xbb']
        yi_ParTevent = Data['X_label']
    fpr_i, tpr_i, threshold_i = roc_curve(target_ParTevent, yi_ParTevent,drop_intermediate=False)

    if i==0: 
        tpr_common = tpr_i
    fpr_ParTevent.append(np.interp(tpr_common, tpr_i, fpr_i))
    auc_ParTevent.append(auc(fpr_i,tpr_i))


    file = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/mlpHlXbb/test_mlpHlXbb_score_training_{i+1}.h5'
    with h5py.File(filename, 'r') as Data:
        yi_mlpHlXbb = Data['evt_score']
        target_mlpHlXbb = Data['evt_label']
    fpr_i, tpr_i, threshold_i = roc_curve(target_mlpHlXbb, yi_mlpHlXbb,drop_intermediate=False)
    fpr_mlpHlXbb.append(np.interp(tpr_common, tpr_i, fpr_i))
    auc_mlpHlXbb.append(auc(fpr_i,tpr_i))
    

fpr_ParTevent_mean = np.mean(fpr_ParTevent,axis=-1)
fpr_ParTevent_std = np.std(fpr_ParTevent,axis=-1)
auc_ParTevent_mean = np.mean(auc_ParTevent,axis=-1)
auc_ParTevent_std = np.std(auc_ParTevent,axis=-1)

fpr_mlpHlXbb_mean = np.mean(fpr_mlpHlXbb,axis=-1)
fpr_mlpHlXbb_std = np.std(fpr_mlpHlXbb,axis=-1)
auc_mlpHlXbb_mean = np.mean(auc_mlpHlXbb,axis=-1)
auc_mlpHlXbb_std = np.std(auc_mlpHlXbb,axis=-1)

b=np.linspace(0,1,101)
fig, ax = plt.subplots()

# plot ROC curve

fig = plt.figure()
ax = fig.add_subplot(4,1,(1,3))
ax.plot(tpr_common, 1/fpr_ParTevent_mean, lw=0.8, label=f'EndToEnd_mean AUC = {auc_ParTevent_mean} +/- {auc_ParTevent_std}',color=blue)
ax.fill_between(tpr_common, (1/fpr_ParTevent_mean-1/fpr_ParTevent_std), (1/fpr_ParTevent_mean+1/fpr_ParTevent_std),color=blue,alpha=0.2)
ax.plot(tpr_common, 1/fpr_mlpHlXbb_mean, lw=0.8, label=f'mlpHlXbb_mean AUC = {auc_mlpHlXbb_mean} +/- {auc_mlpHlXbb_std}',color=red)
ax.fill_between(tpr_common, (1/fpr_mlpHlXbb_mean-1/fpr_mlpHlXbb_std), (1/fpr_mlpHlXbb_mean+1/fpr_mlpHlXbb_std),color=red,alpha=0.2)
#ax.plot(tpr[0], 1/fpr[0], lw=0.8, label=('baseline AUC = {:.3f}%'.format(auc(fpr[0],tpr[0])*100)))
ax.set_ylabel(r'Background rejection')
ax.semilogy()
ax.set_xlim(0.4,1)
ax.grid(True)
ax.legend(loc='lower left')
plt.setp(ax.get_xticklabels(), visible=False)
ax = fig.add_subplot(4,1,4)
ax.plot(tpr_common,(1/fpr_ParTevent_mean)/(1/fpr_mlpHlXbb_mean),lw=0.8,color=blue)
ax.fill_between( tpr_common, (1/(fpr_ParTevent_mean-fpr_ParTevent_std))/(1/fpr_mlpHlXbb_mean),
                 (1/(fpr_ParTevent_mean+fpr_ParTevent_std))/(1/fpr_mlpHlXbb_mean),color=blue,alpha=0.2)
ax.plot(tpr_common,(1/fpr_mlpHlXbb_mean)/(1/fpr_mlpHlXbb_mean),lw=0.8)
ax.fill_between( tpr_common, (1/(fpr_mlpHlXbb_mean-fpr_mlpHlXbb_std))/(1/fpr_mlpHlXbb_mean),
                 (1/(fpr_mlpHlXbb_mean+fpr_mlpHlXbb_std))/(1/fpr_mlpHlXbb_mean),color=red,alpha=0.2)
ax.set_ylim(0.8,5)
ax.set_xlim(0.4,1)
ax.set_ylabel(r'Ratio')
ax.grid(True)
ax.set_xlabel(r'Signal efficiency',loc="right")
fig.savefig(f"../../Finetune_hep/plots/Final_ROC.pdf")

