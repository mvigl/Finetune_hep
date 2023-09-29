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

for i in range(len(sizes)):

    filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subset/test_{sizes[i]}.h5'
    with h5py.File(filename, 'r') as Data:
        yi_ParTevent= Data['ParTevent_evt_score'][:].reshape(-1,1)
        target_ParTevent = Data['ParTevent_evt_label'][:].reshape(-1,1)
        yi_mlpHlXbb = Data['mlpHlXbb_evt_score'][:].reshape(-1,1)
        target_mlpHlXbb = Data['mlpHlXbb_evt_label'][:].reshape(-1,1)
    acc_ete.append(balanced_accuracy_score(target_ParTevent,yi_ParTevent))  
    acc_mlpHlXbb.append(balanced_accuracy_score(target_mlpHlXbb,yi_mlpHlXbb))   

# plot ROC curve

fig = plt.figure()
ax = fig.add_subplot(4,1,(1,3))
ax.plot(sizes, acc_ete, lw=0.8, label=f'Internal+Feats',color='b')
ax.plot(sizes, acc_mlpHlXbb, lw=0.8, label=f'Internal',color='r')
ax.set_ylabel(r'balanced_accuracy_score')
ax.semilogx()
ax.set_ylim(0,1)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel(r'Signal efficiency',loc="right")
fig.savefig(f"../../Finetune_hep/plots/Subsets.pdf")

