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



def get_metrics(filelist_test,modeltype,size,Ntraining):
    with open(filelist_test) as f:
        for line in f:
            filename = line.strip()
            data_index = filename.index("Data")
            sample_name = filename[data_index:]
            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_{modeltype}_{Ntraining}/{size}/{sample_name}'
            with h5py.File(name, 'r') as Data:
                yi_ParTevent.append(Data['evt_score'][:].reshape(-1))
                target_ParTevent.append(Data['evt_label'][:].reshape(-1)) 



acc_ete=[]
auc_ete=[]
fpr_ete=[]
tpr_ete=[]
optimal_threshold_ete=[]

acc_mlpHlXbb=[]
auc_mlpHlXbb=[]
fpr_mlpHlXbb=[]
tpr_mlpHlXbb=[]
optimal_threshold_mlpHlXbb=[]

acc_ete_scratch=[]
auc_ete_scratch=[]
fpr_ete_scratch=[]
tpr_ete_scratch=[]
optimal_threshold_ete_scratch=[]

acc_mlpLatent=[]
auc_mlpLatent=[]
fpr_mlpLatent=[]
tpr_mlpLatent=[]
optimal_threshold_mlpLatent=[]

acc_mlpLatentHl=[]
auc_mlpLatentHl=[]
fpr_mlpLatentHl=[]
tpr_mlpLatentHl=[]
optimal_threshold_mlpLatentHl=[]

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
979854]

sizes = np.sort(sizes)
for i in range(len(sizes)):
    print(sizes[i])
    yi_ParTevent=[]
    target_ParTevent=[]
    yi_ParTevent_scratch=[]
    target_ParTevent_scratch=[]
    yi_mlpHlXbb=[]
    target_mlpHlXbb=[]
    yi_mlpLatent=[]
    target_mlpLatent=[]
    yi_mlpLatentHl=[]
    target_mlpLatentHl=[]
    with open(filelist_test) as f:
        for line in f:
            filename = line.strip()

            data_index = filename.index("Data")
            sample_name = filename[data_index:]
            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_ParTevent/{sizes[i]}/{sample_name}'

            with h5py.File(name, 'r') as Data:
                yi_ParTevent.append(Data['evt_score'][:].reshape(-1))
                target_ParTevent.append(Data['evt_label'][:].reshape(-1))      
            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_mlpHlXbb/{sizes[i]}/{sample_name}'

            with h5py.File(name, 'r') as Data:    
                yi_mlpHlXbb.append(Data['evt_score'][:].reshape(-1))
                target_mlpHlXbb.append(Data['evt_label'][:].reshape(-1))

            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_ParTevent_scratch/{sizes[i]}/{sample_name}'
            with h5py.File(name, 'r') as Data:
                yi_ParTevent_scratch.append(Data['evt_score'][:].reshape(-1))
                target_ParTevent_scratch.append(Data['evt_label'][:].reshape(-1))

            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_mlpLatent/{sizes[i]}/{sample_name}'
            with h5py.File(name, 'r') as Data:
                yi_mlpLatent.append(Data['evt_score'][:].reshape(-1))
                target_mlpLatent.append(Data['evt_label'][:].reshape(-1))

            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_mlpLatentHl/{sizes[i]}/{sample_name}'
            with h5py.File(name, 'r') as Data:
                yi_mlpLatent.append(Data['evt_score'][:].reshape(-1))
                target_mlpLatent.append(Data['evt_label'][:].reshape(-1))           

    target_ParTevent = np.concatenate(target_ParTevent).reshape(-1)
    yi_ParTevent = np.concatenate(yi_ParTevent).reshape(-1)
    fpr, tpr, thresholds = roc_curve(target_ParTevent,yi_ParTevent)
    if i==0: 
        tpr_common = tpr  
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    acc_ete.append(balanced_accuracy_score(target_ParTevent,(yi_ParTevent>= optimal_threshold).astype(int)))
    auc_ete.append(auc(fpr,tpr))
    fpr = np.interp(tpr_common, tpr, fpr)
    fpr_ete.append(fpr)
    tpr_ete.append(tpr_common)
    optimal_threshold_ete.append(optimal_threshold)

                       
    if i < len(sizes_latent):
        target_mlpLatent = np.concatenate(target_mlpLatent).reshape(-1)
        yi_mlpLatent = np.concatenate(yi_mlpLatent).reshape(-1)    
        fpr, tpr, thresholds = roc_curve(target_mlpLatent,yi_mlpLatent)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]    
        acc_mlpLatent.append(balanced_accuracy_score(target_mlpLatent,(yi_mlpLatent>= optimal_threshold).astype(int)))   
        auc_mlpLatent.append(auc(fpr,tpr))
        fpr = np.interp(tpr_common, tpr, fpr)
        fpr_mlpLatent.append(fpr)
        tpr_mlpLatent.append(tpr_common)
        optimal_threshold_mlpLatent.append(optimal_threshold)

    target_ParTevent_scratch = np.concatenate(target_ParTevent_scratch).reshape(-1)
    yi_ParTevent_scratch = np.concatenate(yi_ParTevent_scratch).reshape(-1)
    fpr, tpr, thresholds = roc_curve(target_ParTevent_scratch,yi_ParTevent_scratch)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    acc_ete_scratch.append(balanced_accuracy_score(target_ParTevent_scratch,(yi_ParTevent_scratch>= optimal_threshold).astype(int)))  
    auc_ete_scratch.append(auc(fpr,tpr))
    fpr = np.interp(tpr_common, tpr, fpr)
    fpr_ete_scratch.append(fpr)
    tpr_ete_scratch.append(tpr_common)
    optimal_threshold_ete_scratch.append(optimal_threshold)

    target_mlpHlXbb = np.concatenate(target_mlpHlXbb).reshape(-1)
    yi_mlpHlXbb = np.concatenate(yi_mlpHlXbb).reshape(-1)    
    fpr, tpr, thresholds = roc_curve(target_mlpHlXbb,yi_mlpHlXbb)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]    
    acc_mlpHlXbb.append(balanced_accuracy_score(target_mlpHlXbb,(yi_mlpHlXbb>= optimal_threshold).astype(int)))   
    auc_mlpHlXbb.append(auc(fpr,tpr))
    fpr = np.interp(tpr_common, tpr, fpr)
    fpr_mlpHlXbb.append(fpr)
    tpr_mlpHlXbb.append(tpr_common)
    optimal_threshold_mlpHlXbb.append(optimal_threshold)

acc_ete = np.array(acc_ete).reshape(-1)
acc_mlpHlXbb = np.array(acc_mlpHlXbb).reshape(-1)
acc_ete_scratch = np.array(acc_ete_scratch).reshape(-1)
acc_mlpLatent = np.array(acc_mlpLatent).reshape(-1)

auc_ete = np.array(auc_ete).reshape(-1)
auc_mlpHlXbb = np.array(auc_mlpHlXbb).reshape(-1)
auc_ete_scratch = np.array(auc_ete_scratch).reshape(-1)
auc_mlpLatent = np.array(auc_mlpLatent).reshape(-1)

optimal_threshold_ete = np.array(optimal_threshold_ete).reshape(-1)
optimal_threshold_mlpHlXbb = np.array(optimal_threshold_mlpHlXbb).reshape(-1)
optimal_threshold_ete_scratch = np.array(optimal_threshold_ete_scratch).reshape(-1)
optimal_threshold_mlpLatent = np.array(optimal_threshold_mlpLatent).reshape(-1)

fpr_ete = np.array(fpr_ete)
fpr_mlpHlXbb = np.array(fpr_mlpHlXbb)
fpr_ete_scratch = np.array(fpr_ete_scratch)
fpr_mlpLatent = np.array(fpr_mlpLatent)

tpr_ete = np.array(tpr_ete)
tpr_mlpHlXbb = np.array(tpr_mlpHlXbb)
tpr_ete_scratch = np.array(tpr_ete_scratch)
tpr_mlpLatent = np.array(tpr_mlpLatent)

with h5py.File('/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/metrics/final_subsets.h5', 'w') as data:
        model_group = data.create_group('ete')
        model_group.create_dataset('acc', data=acc_ete)
        model_group.create_dataset('auc', data=auc_ete)
        model_group.create_dataset('optimal_threshold', data=optimal_threshold_ete)
        model_group.create_dataset('fpr', data=fpr_ete)
        model_group.create_dataset('tpr', data=tpr_ete)

        model_group = data.create_group('ete_scratch')
        model_group.create_dataset('acc', data=acc_ete_scratch)
        model_group.create_dataset('auc', data=auc_ete_scratch)
        model_group.create_dataset('optimal_threshold', data=optimal_threshold_ete_scratch)
        model_group.create_dataset('fpr', data=fpr_ete_scratch)
        model_group.create_dataset('tpr', data=tpr_ete_scratch)

        model_group = data.create_group('ete_frozen')
        model_group.create_dataset('acc', data=acc_mlpLatent)
        model_group.create_dataset('auc', data=auc_mlpLatent)
        model_group.create_dataset('optimal_threshold', data=optimal_threshold_mlpLatent)
        model_group.create_dataset('fpr', data=fpr_mlpLatent)
        model_group.create_dataset('tpr', data=tpr_mlpLatent)

        model_group = data.create_group('mlpHlXbb')
        model_group.create_dataset('acc', data=acc_mlpHlXbb)
        model_group.create_dataset('auc', data=auc_mlpHlXbb)
        model_group.create_dataset('optimal_threshold', data=optimal_threshold_mlpHlXbb)
        model_group.create_dataset('fpr', data=fpr_mlpHlXbb)
        model_group.create_dataset('tpr', data=tpr_mlpHlXbb)



#fig = plt.figure()
#ax = fig.add_subplot(4,1,(1,3))
#ax.plot(sizes, acc_ete, lw=0.2, label=f'E2e',color='b',marker='.')
#ax.plot(sizes_latent, acc_mlpLatent, lw=0.2, label=f'E2e Frozen',color='c',marker='.')
#ax.plot(sizes, acc_mlpHlXbb, lw=0.2, label=f'mlpHlXbb',color='r',marker='.')
#ax.plot(sizes, acc_ete_scratch, lw=0.2, label=f'E2e scratch',color='g',marker='.')
#ax.set_ylabel(r'balanced_accuracy_score')
#ax.semilogx()
#ax.set_ylim(np.min(acc_ete_scratch)-0.001,1)
#ax.grid(True)
#ax.legend(loc='lower right')
#ax.set_xlabel(r'# Training data (evts)',loc="right")
#fig.savefig(f"../../Finetune_hep/plots/Subsets.pdf")

