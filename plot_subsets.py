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

sizes = [1730,19332,195762,1959955,2704,29145,
293774,2940006,4665,48752,489801,400263,5880252,
6860297,777,7840400,8820463,9547,97752,979854]
thr = 0.5
for i in range(len(sizes)):
    with open(filelist) as f:
        for line in f:
            filename = line.strip()
            print('reading : ',filename)
            data_index = filename.index("Data")
            filename = filename[data_index:]
            filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/ParTevent/{sizes[i]}/{filename}.h5'
            with h5py.File(filename, 'r') as Data:
                yi_ParTevent.append(Data['evt_score'][:].reshape(-1))
                target_ParTevent.append(Data['evt_label'][:].reshape(-1))
            filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/mlHlXbb/{sizes[i]}/{filename}.h5'
            with h5py.File(filename, 'r') as Data:    
                yi_mlpHlXbb.append(Data['evt_score'][:].reshape(-1))
                target_mlpHlXbb.append(Data['evt_label'][:].reshape(-1))
    acc_ete.append(balanced_accuracy_score(target_ParTevent.reshape(-1),yi_ParTevent.reshape(-1)))  
    acc_mlpHlXbb.append(balanced_accuracy_score(target_mlpHlXbb.reshape(-1),yi_mlpHlXbb.reshape(-1)))   

fig = plt.figure()
ax = fig.add_subplot(4,1,(1,3))
ax.plot(sizes, acc_ete.reshape(-1), lw=0.8, label=f'E2e',color='b')
ax.plot(sizes, acc_mlpHlXbb.reshape(-1), lw=0.8, label=f'mlpHlXbb',color='r')
ax.set_ylabel(r'balanced_accuracy_score')
ax.semilogx()
ax.set_ylim(0,1)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel(r'Signal efficiency',loc="right")
fig.savefig(f"../../Finetune_hep/plots/Subsets.pdf")

