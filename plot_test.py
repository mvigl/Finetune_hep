import sys, os
sys.path.append('/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/python')

import ParT_Xbb
import ParT_mlp
import Mlp
import definitions as df
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import cm
import numpy as np
import argparse
import pickle
import h5py
from torch.utils.data import Dataset, DataLoader

subset=False
filelist = '/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/config/test_list.txt'
idxmap = df.get_idxmap(filelist)
integer_file_map = df.create_integer_file_map(idxmap)
config_path='../../Finetune_hep/config/myJetClass_full.yaml'
device = df.get_device()

fpr=[]
tpr=[]
threshold=[]


baseline_model = Mlp.InvariantModel(    phi=Mlp.make_mlp(6,24,3,binary=False),
                                        rho=Mlp.make_mlp(24,48,3,for_inference=True))
baseline_model.to(device)

baseline_model.load_state_dict(torch.load('../../run/Final_baseline/models/baseline_hl3_nodes24_nj2_lr0.001_bs512_training_0.pt'))

Dataset = Mlp.CustomDataset(filelist,
                            device,
                            scaler_path='../../run/Final_baseline/models/baseline_hl3_nodes24_nj2_lr0.001_bs512_training_0.pkl',
                            Xbb_scores_path='no',
                            test=True)

train_loader_baseline = DataLoader(Dataset, batch_size=512, shuffle=True)

print(baseline_model)
yi_baseline,target_baseline = Mlp.get_preds(baseline_model,train_loader_baseline,subset,device)    
fpr_i, tpr_i, threshold_i = roc_curve(target_baseline, yi_baseline,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)

baseline_model.load_state_dict(torch.load('../../run/Final_mlpHlXbb/models/mlpHlXbb_hl3_nodes24_nj2_lr0.001_bs512_training_0.pt'))

Dataset_mlpHlXbb = Mlp.CustomDataset(filelist,
                            device,
                            scaler_path='../../run/Final_mlpHlXbb/models/mlpHlXbb_hl3_nodes24_nj2_lr0.001_bs512_training_0.pkl',
                            Xbb_scores_path='../../Finetune_hep/models/ParTXbb/Final_ParTXbb_test.h5',
                            test=True)

train_loader_mlpHlXbb = DataLoader(Dataset_mlpHlXbb, batch_size=512, shuffle=True)

yi_mlpHlXbb,target_mlpHlXbb = Mlp.get_preds(baseline_model,train_loader_mlpHlXbb,subset,device)    
fpr_i, tpr_i, threshold_i = roc_curve(target_mlpHlXbb, yi_mlpHlXbb,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)


with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  
ParTevent_model = ParT_mlp.get_model(data_config,for_inference=True)  

ParTevent_model.to(device)
ParTevent_model.eval()
ParTevent_model.load_state_dict(torch.load(f'/home/iwsatlas1/mavigl/Finetune_hep_dir/run/Final_ParTevent/models/ParTevent_hl3_nodes128_nj2_lr0.001_bs256_WparT_training_0.pt'))
Dataset = df.CustomDataset(idxmap,integer_file_map)

train_loader = DataLoader(Dataset, batch_size=512, shuffle=True,num_workers=6)
build_features = df.build_features_and_labels
yi_ete,target_ete = ParT_mlp.get_preds(ParTevent_model,train_loader,device,subset,build_features)

fpr_i, tpr_i, threshold_i = roc_curve(target_ete, yi_ete,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)

fpr_ratio=[]
tpr_ratio = tpr[2]
for i in range(3):
    fpr_ratio.append(np.interp(tpr_ratio, tpr[i], fpr[i]))

b=np.linspace(0,1,101)
fig, ax = plt.subplots()

ax.hist(yi_mlpHlXbb, lw=0.8,bins=b, weights=target_mlpHlXbb, label=('mlpHlXbb sig'.format(auc(fpr[0],tpr[0])*100)),histtype='step', density=True, alpha=0.7)
ax.hist(yi_mlpHlXbb, lw=0.8,bins=b, weights=(target_mlpHlXbb==0)*1, label=('mlpHlXbb bkg'.format(auc(fpr[0],tpr[0])*100)),histtype='step', density=True, alpha=0.7)
ax.hist(yi_ete, lw=0.8,bins=b, weights=target_ete, label=('E2e sig'.format(auc(fpr[2],tpr[2])*100)),histtype='step', density=True, alpha=0.7, color='blue',linestyle='dashed')
ax.hist(yi_ete, lw=0.8,bins=b, weights=(target_ete==0)*1, label=('E2e bkg'.format(auc(fpr[2],tpr[2])*100)),histtype='step', density=True, alpha=0.7, color='red',linestyle='dashed')
ax.set_xlabel(r'Signal Efficiency')
ax.set_ylabel(r'')
ax.semilogy()
ax.set_xlim(0,1)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/Final_preds_test.pdf")


# plot ROC curve

fig = plt.figure()
ax = fig.add_subplot(4,1,(1,3))
ax.plot(tpr[2], 1/fpr[2], lw=0.8, label=('E2e AUC = {:.3f}%'.format(auc(fpr[2],tpr[2])*100)))
ax.plot(tpr[1], 1/fpr[1], lw=0.8, label=('mlpHlXbb AUC = {:.3f}%'.format(auc(fpr[1],tpr[1])*100)))
ax.plot(tpr[0], 1/fpr[0], lw=0.8, label=('baseline AUC = {:.3f}%'.format(auc(fpr[0],tpr[0])*100)))
ax.set_ylabel(r'Background rejection')
ax.semilogy()
ax.set_xlim(0.4,1)
ax.grid(True)
ax.legend(loc='lower left')
plt.setp(ax.get_xticklabels(), visible=False)
ax = fig.add_subplot(4,1,4)
ax.plot(tpr_ratio,(1/fpr_ratio[2])/(1/fpr_ratio[1]),lw=0.8)
ax.plot(tpr_ratio,(1/fpr_ratio[1])/(1/fpr_ratio[1]),lw=0.8)
ax.set_ylim(0.8,5)
ax.set_xlim(0.4,1)
ax.set_ylabel(r'Ratio')
ax.grid(True)
ax.set_xlabel(r'Signal efficiency',loc="right")
fig.savefig(f"../../Finetune_hep/plots/Final_ROC_test.pdf")

