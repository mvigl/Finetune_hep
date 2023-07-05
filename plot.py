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
from torch.utils.data import Dataset, DataLoader

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]

def get_vars(filelist,subset):
    idxmap = df.get_idxmap(filelist)
    integer_file_map = df.create_integer_file_map(idxmap)
    batch_size = 512
    Dataset = Mlp.CustomDataset(idxmap,integer_file_map,'cpu',scaler_path='no')
    loader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for i, batch in enumerate( loader ):
            if i==0:
                data, target = batch
                data = data.cpu().numpy()
                target = target.cpu().numpy()
            else: 
                data = np.concatenate((data,batch[0].cpu().numpy()),axis=0)
                target = np.concatenate((target,batch[1].cpu().numpy()),axis=0)
            if (subset and i > 10): break    
    return data,target

subset=True
filelist = '/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/config/test_list.txt'
data,target = get_vars(filelist,subset)
print(jVars.index('fj_pt'))
fig, ax = plt.subplots()
bins=np.linspace(0,4500,201)
ax.hist(data[:,jVars.index('fj_pt')], bins=bins, lw=0.8, weights=target, label=('sig'),density=True)
ax.hist(data[:,jVars.index('fj_pt')], bins=bins, lw=0.8, weights=(target==0)*1, label=('bkg'),density=True)
ax.set_xlabel(r'pt')
ax.set_ylabel(r'')
ax.semilogy()
ax.legend(loc='lower left')
fig.savefig(f"../../Finetune_hep/plots/pt.pdf")

fig, ax = plt.subplots()
bins=np.linspace(0,4500,201)
ax.hist(data[:,jVars.index('fj_pt')], bins=bins, lw=0.8, weights=target, label=('sig'),density=False)
ax.hist(data[:,jVars.index('fj_pt')], bins=bins, lw=0.8, weights=(target==0)*1, label=('bkg'),density=False)
ax.set_xlabel(r'pt')
ax.set_ylabel(r'')
ax.semilogy()
ax.legend(loc='lower left')
fig.savefig(f"../../Finetune_hep/plots/pt_not_norm.pdf")

fpr=[]
tpr=[]
threshold=[]
config_path='../../Finetune_hep/config/myJetClass_full.yaml'
device = df.get_device()
device='cpu'
with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  


baseline_model = ParT_mlp.make_mlp(12,24,6)
baseline_model.to(device)

baseline_model.load_state_dict(torch.load('../../run/baseline_test/models/baseline_hl6_nodes24_nj2_lr0.001_bs512_training_0.pt'))

idxmap = df.get_idxmap(filelist)
integer_file_map = df.create_integer_file_map(idxmap)
batch_size = 512
Dataset = Mlp.CustomDataset(idxmap,integer_file_map,device,scaler_path='no')
train_loader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

print(baseline_model)
yi,target = Mlp.get_preds(baseline_model,train_loader,subset,device)    
fpr_i, tpr_i, threshold_i = roc_curve(target, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)


print(tpr[0])

print(fpr[0])


b=np.linspace(0,1,101)
fig, ax = plt.subplots()
ax.hist(yi, lw=0.8,bins=b, weights=target, label=('sig'.format(auc(fpr[0],tpr[0])*100)),density=True)
ax.hist(yi, lw=0.8,bins=b, weights=(target==0)*1, label=('bkg'.format(auc(fpr[0],tpr[0])*100)),density=True)
ax.set_xlabel(r'Signal Efficiency')
ax.set_ylabel(r'')
ax.semilogy()
ax.set_xlim(0,1)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/test2.pdf")

fig, ax = plt.subplots()
ax.hist(yi, lw=0.8,bins=b, weights=target, label=('sig'.format(auc(fpr[0],tpr[0])*100)),density=False)
ax.hist(yi, lw=0.8,bins=b, weights=(target==0)*1, label=('bkg'.format(auc(fpr[0],tpr[0])*100)),density=False)
ax.set_xlabel(r'Signal Efficiency')
ax.set_ylabel(r'')
ax.semilogy()
ax.set_xlim(0,1)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/test3.pdf")
# plot ROC curve
color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))

fig, ax = plt.subplots()
ax.plot(tpr[0], 1/fpr[0], lw=0.8, label=('baseline AUC = {:.3f}%'.format(auc(fpr[0],tpr[0])*100)), color=next(color))
ax.set_xlabel(r'Signal Efficiency')
ax.set_ylabel(r'1/FPR')
ax.semilogy()
ax.set_xlim(0,1)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/test.pdf")

