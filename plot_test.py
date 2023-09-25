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
import h5py


fpr_mlpHlXbb=[]
auc_mlpHlXbb=[]
fpr_ParTevent=[]
auc_ParTevent=[]

for i in range(5):

    filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTevent/test_ParTevent_score_training_{i+1}.h5'
    with h5py.File(filename, 'r') as Data:
        yi_ParTevent = Data['evt_score'][:].reshape(-1,1)
        target_ParTevent = Data['evt_label'][:].reshape(-1,1)
    fpr_i, tpr_i, threshold_i = roc_curve(target_ParTevent, yi_ParTevent,drop_intermediate=False)

    if i==0: 
        tpr_common = tpr_i
    fpr_ParTevent.append(np.interp(tpr_common, tpr_i, fpr_i))
    auc_ParTevent.append(auc(fpr_i,tpr_i))


    filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/mlpHlXbb/test_mlpHlXbb_score_training_{i+1}.h5'
    with h5py.File(filename, 'r') as Data:
        yi_mlpHlXbb = Data['evt_score'][:].reshape(-1,1)
        target_mlpHlXbb = Data['evt_label'][:].reshape(-1,1)
    fpr_i, tpr_i, threshold_i = roc_curve(target_mlpHlXbb, yi_mlpHlXbb,drop_intermediate=False)
    fpr_mlpHlXbb.append(np.interp(tpr_common, tpr_i, fpr_i))
    auc_mlpHlXbb.append(auc(fpr_i,tpr_i))
    

fpr_ParTevent_mean = np.mean(fpr_ParTevent,axis=0)
fpr_ParTevent_std = np.std(fpr_ParTevent,axis=0)
auc_ParTevent_mean = np.mean(auc_ParTevent,axis=0)
auc_ParTevent_std = np.std(auc_ParTevent,axis=0)

fpr_mlpHlXbb_mean = np.mean(fpr_mlpHlXbb,axis=0)
fpr_mlpHlXbb_std = np.std(fpr_mlpHlXbb,axis=0)
auc_mlpHlXbb_mean = np.mean(auc_mlpHlXbb,axis=0)
auc_mlpHlXbb_std = np.std(auc_mlpHlXbb,axis=0)

b=np.linspace(0,1,101)
fig, ax = plt.subplots()

# plot ROC curve

fig = plt.figure()
ax = fig.add_subplot(4,1,(1,3))
ax.plot(tpr_common, 1/fpr_ParTevent_mean, lw=0.8, label=f'EndToEnd_mean AUC = {auc_ParTevent_mean:.4f} +/- {auc_ParTevent_std:.4f}',color='b')
ax.fill_between(tpr_common, (1/(fpr_ParTevent_mean-fpr_ParTevent_std)), (1/(fpr_ParTevent_mean+fpr_ParTevent_std)),color='b',alpha=0.2)
ax.plot(tpr_common, 1/fpr_mlpHlXbb_mean, lw=0.8, label=f'mlpHlXbb_mean AUC = {auc_mlpHlXbb_mean:.4f} +/- {auc_mlpHlXbb_std:.4f}',color='r')
ax.fill_between(tpr_common, (1/(fpr_mlpHlXbb_mean-fpr_mlpHlXbb_std)), (1/(fpr_mlpHlXbb_mean+fpr_mlpHlXbb_std)),color='r',alpha=0.2)
#ax.plot(tpr[0], 1/fpr[0], lw=0.8, label=('baseline AUC = {:.3f}%'.format(auc(fpr[0],tpr[0])*100)))
ax.set_ylabel(r'Background rejection')
ax.semilogy()
ax.set_xlim(0.4,1)
ax.grid(True)
ax.legend(loc='lower left')
plt.setp(ax.get_xticklabels(), visible=False)
ax = fig.add_subplot(4,1,4)
ax.plot(tpr_common,(1/fpr_ParTevent_mean)/(1/fpr_mlpHlXbb_mean),lw=0.8,color='b')
ax.fill_between( tpr_common, (1/(fpr_ParTevent_mean-fpr_ParTevent_std))/(1/fpr_mlpHlXbb_mean),
                 (1/(fpr_ParTevent_mean+fpr_ParTevent_std))/(1/fpr_mlpHlXbb_mean),color='b',alpha=0.2)
ax.plot(tpr_common,(1/fpr_mlpHlXbb_mean)/(1/fpr_mlpHlXbb_mean),lw=0.8,color='r')
ax.fill_between( tpr_common, (1/(fpr_mlpHlXbb_mean-fpr_mlpHlXbb_std))/(1/fpr_mlpHlXbb_mean),
                 (1/(fpr_mlpHlXbb_mean+fpr_mlpHlXbb_std))/(1/fpr_mlpHlXbb_mean),color='r',alpha=0.2)
ax.set_ylim(0.8,5)
ax.set_xlim(0.4,1)
ax.set_ylabel(r'Ratio')
ax.grid(True)
ax.set_xlabel(r'Signal efficiency',loc="right")
fig.savefig(f"../../Finetune_hep/plots/Final_ROC.pdf")

