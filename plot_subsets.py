from Finetune_hep.python import ParT_mlp
from Finetune_hep.python import Mlp
from Finetune_hep.python import definitions as df
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
import h5py


acc_ete=[]
acc_mlpHlXbb=[]
filelist_test = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/config/subset_config/test_list_check.txt'
sizes = [
1730,
19332,
195762,
1959955,
2704,
29145,
293774,
2940006,
4665,
48752,
489801,
4900263,
5880252,
6860297,
777,
7840400,
8820463,
9547,
97752,
979854]
sizes = np.sort(sizes)
thr = 0.5
for i in range(len(sizes)):
    print(sizes[i])
    yi_ParTevent=[]
    target_ParTevent=[]
    yi_mlpHlXbb=[]
    target_mlpHlXbb=[]
    with open(filelist_test) as f:
        for line in f:
            filename = line.strip()
            #print('reading : ',filename)
            data_index = filename.index("Data")
            sample_name = filename[data_index:]
            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/ParTevent/{sizes[i]}/{sample_name}'
            #print('loading : ',name)
            with h5py.File(name, 'r') as Data:
                yi_ParTevent.append(Data['evt_score'][:].reshape(-1))
                target_ParTevent.append(Data['evt_label'][:].reshape(-1))
            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/mlpHlXbb/{sizes[i]}/{sample_name}'
            #print('loading : ',name)
            with h5py.File(name, 'r') as Data:    
                yi_mlpHlXbb.append(Data['evt_score'][:].reshape(-1))
                target_mlpHlXbb.append(Data['evt_label'][:].reshape(-1))
    target_ParTevent = np.concatenate(target_ParTevent).reshape(-1)
    target_ParTevent[:100] = 1
    yi_ParTevent = np.concatenate(yi_ParTevent).reshape(-1)
    fpr, tpr, thresholds = roc_curve(target_mlpHlXbb,yi_mlpHlXbb)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    acc_ete.append(accuracy_score(target_ParTevent,(yi_ParTevent>= optimal_threshold).astype(int)))  

    target_mlpHlXbb = np.concatenate(target_mlpHlXbb).reshape(-1)
    target_mlpHlXbb[:100] = 1
    yi_mlpHlXbb = np.concatenate(yi_mlpHlXbb).reshape(-1)    
    fpr, tpr, thresholds = roc_curve(target_mlpHlXbb,yi_mlpHlXbb)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]    
    acc_mlpHlXbb.append(accuracy_score(target_mlpHlXbb,(yi_mlpHlXbb>= optimal_threshold).astype(int)))   

acc_ete = np.array(acc_ete).reshape(-1)
acc_mlpHlXbb = np.array(acc_mlpHlXbb).reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(4,1,(1,3))
ax.plot(sizes, acc_ete, lw=0.8, label=f'E2e',color='b',marker='o')
ax.plot(sizes, acc_mlpHlXbb, lw=0.8, label=f'mlpHlXbb',color='r',marker='o')
ax.set_ylabel(r'balanced_accuracy_score')
ax.semilogx()
ax.semilogy()
ax.set_ylim(0.6,1)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel(r'Signal efficiency',loc="right")
fig.savefig(f"../../Finetune_hep/plots/Subsets.pdf")

