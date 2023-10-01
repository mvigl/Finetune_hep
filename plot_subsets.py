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
acc_ete_scratch=[]
acc_mlpLatent=[]
filelist_test = '/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt'
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
979854,
1089208]

sizes_latent = [
1960151,
196015,
19601,
2940227,
294022,
29402,
2940,
4900379,
490037,
49003,
4900,
5880454,
6860530,
7840606,
8820682,
980075,
98007,
9800,
980,
]

sizes_latent = np.sort(sizes_latent)
sizes = np.sort(sizes)
for i in range(len(sizes)-1):
    print(sizes[i])
    yi_ParTevent=[]
    target_ParTevent=[]
    yi_ParTevent_scratch=[]
    target_ParTevent_scratch=[]
    yi_mlpHlXbb=[]
    target_mlpHlXbb=[]
    yi_mlpLatent=[]
    target_mlpLatent=[]
    with open(filelist_test) as f:
        for line in f:
            filename = line.strip()

            data_index = filename.index("Data")
            sample_name = filename[data_index:]
            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/ParTevent/{sizes[i]}/{sample_name}'

            with h5py.File(name, 'r') as Data:
                yi_ParTevent.append(Data['evt_score'][:].reshape(-1))
                target_ParTevent.append(Data['evt_label'][:].reshape(-1))      
            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/mlpHlXbb/{sizes[i]}/{sample_name}'

            with h5py.File(name, 'r') as Data:    
                yi_mlpHlXbb.append(Data['evt_score'][:].reshape(-1))
                target_mlpHlXbb.append(Data['evt_label'][:].reshape(-1))
    #        name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/ParTevent_scratch/{sizes[i]}/{sample_name}'

    #        with h5py.File(name, 'r') as Data:
    #            yi_ParTevent_scratch.append(Data['evt_score'][:].reshape(-1))
    #            target_ParTevent_scratch.append(Data['evt_label'][:].reshape(-1))      
    #name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/mlpLatent/{sizes_latent[i]}/test_mlpLatent.h5'

    #with h5py.File(name, 'r') as Data:    
    #    yi_mlpLatent.append(Data['evt_score'][:].reshape(-1))
    #    target_mlpLatent.append(Data['evt_label'][:].reshape(-1))    

    #target_mlpLatent = np.array(target_mlpLatent).reshape(-1)
    #yi_mlpLatent = np.array(yi_mlpLatent).reshape(-1)    
    #fpr, tpr, thresholds = roc_curve(target_mlpLatent,yi_mlpLatent)
    #optimal_threshold = thresholds[np.argmax(tpr - fpr)]    
    #acc_mlpLatent.append(balanced_accuracy_score(target_mlpLatent,(yi_mlpLatent>= optimal_threshold).astype(int)))   

    target_ParTevent = np.concatenate(target_ParTevent).reshape(-1)
    yi_ParTevent = np.concatenate(yi_ParTevent).reshape(-1)
    fpr, tpr, thresholds = roc_curve(target_ParTevent,yi_ParTevent)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    acc_ete.append(balanced_accuracy_score(target_ParTevent,(yi_ParTevent>= optimal_threshold).astype(int)))

    #target_ParTevent_scratch = np.concatenate(target_ParTevent_scratch).reshape(-1)
    #yi_ParTevent_scratch = np.concatenate(yi_ParTevent_scratch).reshape(-1)
    #fpr, tpr, thresholds = roc_curve(target_ParTevent_scratch,yi_ParTevent_scratch)
    #optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    #acc_ete_scratch.append(balanced_accuracy_score(target_ParTevent_scratch,(yi_ParTevent_scratch>= optimal_threshold).astype(int)))  

    target_mlpHlXbb = np.concatenate(target_mlpHlXbb).reshape(-1)
    yi_mlpHlXbb = np.concatenate(yi_mlpHlXbb).reshape(-1)    
    fpr, tpr, thresholds = roc_curve(target_mlpHlXbb,yi_mlpHlXbb)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]    
    acc_mlpHlXbb.append(balanced_accuracy_score(target_mlpHlXbb,(yi_mlpHlXbb>= optimal_threshold).astype(int)))   


filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTevent/test_ParTevent_score_training_1.h5'
with h5py.File(filename, 'r') as Data:
    yi_ParTevent_full = Data['evt_score'][:].reshape(-1,1)
    target_ParTevent_full = Data['evt_label'][:].reshape(-1,1)
fpr_ParTevent_full, tpr_ParTevent_full, thresholds_ParTevent_full = roc_curve(target_ParTevent_full, yi_ParTevent_full,drop_intermediate=False)
optimal_threshold = thresholds_ParTevent_full[np.argmax(tpr_ParTevent_full - fpr_ParTevent_full)]    
acc_ete_full = (balanced_accuracy_score(target_ParTevent_full,(yi_ParTevent_full>= optimal_threshold).astype(int)))   

filename = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/mlpHlXbb/test_mlpHlXbb_score_training_1.h5'
with h5py.File(filename, 'r') as Data:
    yi_mlpHlXbb_full = Data['evt_score'][:].reshape(-1,1)
    target_mlpHlXbb_full = Data['evt_label'][:].reshape(-1,1)
fpr_mlpHlXbb_full, tpr_mlpHlXbb_full, thresholds_mlpHlXbb_full = roc_curve(target_mlpHlXbb_full, yi_mlpHlXbb_full,drop_intermediate=False)
optimal_threshold = thresholds_mlpHlXbb_full[np.argmax(tpr_mlpHlXbb_full - fpr_mlpHlXbb_full)]    
acc_mlpHlXbb_full = (balanced_accuracy_score(target_mlpHlXbb_full,(yi_mlpHlXbb_full>= optimal_threshold).astype(int))) 

acc_ete = np.array(acc_ete).reshape(-1)
acc_mlpHlXbb = np.array(acc_mlpHlXbb).reshape(-1)
#acc_ete_scratch = np.array(acc_ete_scratch).reshape(-1)
#acc_mlpLatent = np.array(acc_mlpLatent).reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(4,1,(1,3))
ax.plot(sizes, acc_ete, lw=0.8, label=f'E2e',color='b',marker='o')
#ax.plot(sizes_latent, acc_mlpLatent, lw=0.8, label=f'mlpHlXbb',color='c',marker='o')
ax.plot(sizes, acc_mlpHlXbb, lw=0.8, label=f'mlpHlXbb',color='r',marker='o')
#ax.plot(sizes, acc_ete_scratch, lw=0.8, label=f'E2e',color='g',marker='o')
ax.set_ylabel(r'balanced_accuracy_score')
ax.semilogx()
ax.set_ylim(np.min(acc_mlpHlXbb)-0.001,1)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel(r'Signal efficiency',loc="right")
fig.savefig(f"../../Finetune_hep/plots/Subsets.pdf")

