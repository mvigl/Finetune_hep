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

filelist = '/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/config/test_list.txt'
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]      
def get_vars(filelist):
    i=0
    with open(filelist) as f:
        for line in f:
            filename = line.strip()
            print('reading : ',filename)
            with h5py.File(filename, 'r') as Data:
                if i ==0:
                    data = Data['X_jet'][:]
                    target = Data['labels'][:]
                    target_jet = Data['X_label'][:]
                else:
                    data = np.concatenate((data,Data['X_jet'][:]),axis=0)
                    target = np.concatenate((target,Data['labels'][:]),axis=0) 
                    target_jet = np.concatenate((target_jet,Data['X_label'][:]),axis=0) 
            i+=1
    return data,target,target_jet

data,target,target_jet = get_vars(filelist)

fpr=[]
tpr=[]
threshold=[]

yi_doubleb = (data[:,:,jVars.index('fj_doubleb')].reshape(-1,1)+1)/2
target_doubleb = target_jet[:,:,labelVars.index('label_H_bb')].reshape(-1,1)
fpr_i, tpr_i, threshold_i = roc_curve(target_doubleb, yi_doubleb,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)


with h5py.File('../../Finetune_hep/models/ParTXbb/ParTXbb_test_full.h5', 'r') as test_data:
    yi_ParTXbb,target_ParTXbb = test_data['Xbb'][:].reshape(-1),test_data['X_label'][:].reshape(-1)
fpr_i, tpr_i, threshold_i = roc_curve(target_ParTXbb, yi_ParTXbb,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)

b=np.linspace(0,1,101)
fig, ax = plt.subplots()
ax.hist(yi_ParTXbb, lw=0.8,bins=b, weights=target_ParTXbb, label=('ParTXbb sig'),alpha=0.8,density=True)
ax.hist(yi_ParTXbb, lw=0.8,bins=b, weights=(target_ParTXbb==0)*1, label=('ParTXbb bkg'),alpha=0.8,density=True)
ax.hist(yi_doubleb, lw=0.8,bins=b, weights=target_doubleb, label=('fj_doubleb sig'),alpha=0.8,density=True)
ax.hist(yi_doubleb, lw=0.8,bins=b, weights=(target_doubleb==0)*1, label=('fj_doubleb bkg'),alpha=0.8,density=True)
ax.set_xlabel(r'Signal Efficiency')
ax.set_ylabel(r'')
ax.semilogy()
ax.set_xlim(0,1)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/Full_preds_Xbb.pdf")

# plot ROC curve
color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))

fig, ax = plt.subplots()
ax.plot(tpr[1], 1/fpr[1], lw=0.8, label=('ParTXbb AUC = {:.2f}%'.format(auc(fpr[1],tpr[1])*100)), color=next(color))
ax.plot(tpr[0], 1/fpr[0], lw=0.8, label=('Xbb AUC = {:.2f}%'.format(auc(fpr[0],tpr[0])*100)), color=next(color))
ax.set_xlabel(r'Signal Efficiency')
ax.set_ylabel(r'1/FPR')
ax.semilogy()
ax.set_xlim(0,1)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/Full_ROC_Xbb.pdf")


