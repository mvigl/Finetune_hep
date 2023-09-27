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
fpr_mlpLatent=[]
auc_mlpLatent=[]
fpr_baseline=[]
auc_baseline=[]
fpr_mlpLatentHl=[]
auc_mlpLatentHl=[]

for i in range(1):

    filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/mlpHlXbb/val_mlpHlXbb_score_training_{i+1}.h5'
    with h5py.File(filename, 'r') as Data:
        yi_mlpHlXbb = Data['evt_score'][:].reshape(-1,1)
        target_mlpHlXbb = Data['evt_label'][:].reshape(-1,1)
    fpr_i, tpr_i, threshold_i = roc_curve(target_mlpHlXbb, yi_mlpHlXbb,drop_intermediate=False)
    fpr_mlpHlXbb.append(np.interp(tpr_common, tpr_i, fpr_i))
    auc_mlpHlXbb.append(auc(fpr_i,tpr_i))

    filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/mlpXbb/val_mlpXbb_score_training_{i+1}.h5'
    with h5py.File(filename, 'r') as Data:
        yi_baseline = Data['evt_score'][:].reshape(-1,1)
        target_baseline = Data['evt_label'][:].reshape(-1,1)
    fpr_i, tpr_i, threshold_i = roc_curve(target_baseline, yi_baseline,drop_intermediate=False)
    fpr_baseline.append(np.interp(tpr_common, tpr_i, fpr_i))
    auc_baseline.append(auc(fpr_i,tpr_i))

    filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/mlpLatent/val_mlpLatent_score_training_{i+1}.h5'
    with h5py.File(filename, 'r') as Data:
        yi_mlpLatent = Data['evt_score'][:].reshape(-1,1)
        target_mlpLatent = Data['evt_label'][:].reshape(-1,1)
    fpr_i, tpr_i, threshold_i = roc_curve(target_mlpLatent, yi_mlpLatent,drop_intermediate=False)
    fpr_mlpLatent.append(np.interp(tpr_common, tpr_i, fpr_i))
    auc_mlpLatent.append(auc(fpr_i,tpr_i))

    filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/mlpLatentHl/val_mlpLatentHl_score_training_{i+1}.h5'
    with h5py.File(filename, 'r') as Data:
        yi_mlpLatentHl = Data['evt_score'][:].reshape(-1,1)
        target_mlpLatentHl = Data['evt_label'][:].reshape(-1,1)
    fpr_i, tpr_i, threshold_i = roc_curve(target_mlpLatentHl, yi_mlpLatentHl,drop_intermediate=False)
    fpr_mlpLatentHl.append(np.interp(tpr_common, tpr_i, fpr_i))
    auc_mlpLatentHl.append(auc(fpr_i,tpr_i))
    

fpr_mlpLatent_mean = fpr_mlpLatent[0]
auc_mlpLatent_mean = auc_mlpLatent[0]

fpr_mlpLatentHl_mean = fpr_mlpLatentHl[0]
auc_mlpLatentHl_mean = auc_mlpLatentHl[0]

fpr_mlpHlXbb_mean = fpr_mlpHlXbb[0]
auc_mlpHlXbb_mean = auc_mlpHlXbb[0]

fpr_baseline_mean = fpr_baseline[0]
auc_baseline_mean = auc_baseline[0]

b=np.linspace(0,1,101)
fig, ax = plt.subplots()

# plot ROC curve

fig = plt.figure()
ax = fig.add_subplot(4,1,(1,3))
ax.plot(tpr_common, 1/fpr_mlpLatentHl_mean, lw=0.8, label=f'mlpLatentHl AUC = {auc_mlpLatentHl_mean:.4f}',color='b')

ax.plot(tpr_common, 1/fpr_mlpLatent_mean, lw=0.8, label=f'mlpLatent AUC = {auc_mlpLatent_mean:.4f}',color='r')

ax.plot(tpr_common, 1/fpr_mlpHlXbb_mean, lw=0.8, label=f'baseline AUC = {auc_mlpHlXbb_mean:.4f}',color='o')

ax.plot(tpr_common, 1/fpr_baseline_mean, lw=0.8, label=f'baseline AUC = {auc_baseline_mean:.4f}',color='g')

ax.set_ylabel(r'Background rejection')
ax.semilogy()
ax.set_ylim(1,10000000)
ax.set_xlim(0.4,1)
ax.grid(True)
ax.legend(loc='lower left')
plt.setp(ax.get_xticklabels(), visible=False)
ax = fig.add_subplot(4,1,4)
ax.plot(tpr_common,(1/fpr_mlpLatentHl_mean)/(1/fpr_mlpHlXbb_mean),lw=0.8,color='b')
ax.plot(tpr_common,(1/fpr_mlpLatent_mean)/(1/fpr_mlpHlXbb_mean),lw=0.8,color='r')
ax.plot(tpr_common,(1/fpr_mlpHlXbb_mean)/(1/fpr_mlpHlXbb_mean),lw=0.8,color='o')
ax.plot(tpr_common,(1/fpr_baseline_mean)/(1/fpr_mlpHlXbb_mean),lw=0.8,color='g')
         
ax.set_ylim(0.5,5)
ax.set_xlim(0.4,1)
ax.set_ylabel(r'Ratio')
ax.grid(True)
ax.set_xlabel(r'Signal efficiency',loc="right")
fig.savefig(f"../../Finetune_hep/plots/Final_ROC_val.pdf")

