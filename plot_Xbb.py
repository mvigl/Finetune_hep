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


subset=False

X_pfo_test , X_jet_test , njets_test , labels_test , X_label_test , evts_test  = df.get_test_data('/home/iwsatlas1/mavigl/Hbb/ParT/Dataset',subset)

X_jet_test = np.reshape(X_jet_test[evts_test],(-1,len(df.jVars)))
X_pfo_test = np.reshape(X_pfo_test[evts_test],(-1,110,len(df.pVars)))
X_label_test = np.reshape(X_label_test[evts_test],(-1,len(df.labelVars)))
evts_test = np.where(X_pfo_test[:,0,df.pVars.index('pfcand_ptrel')] != 0 )[0]
labels_test = X_label_test[evts_test,df.labelVars.index('label_H_bb')]

Xbb_scores = np.reshape(X_jet_test[evts_test,df.jVars.index('fj_doubleb')],(-1))
ParTXbb_scores = np.reshape(df.getXbb_scores('../../Finetune_hep/models/ParTXbb/ParTXbb_scores_0_test.npy',evts_test),(-1))[evts_test] 
print(len(ParTXbb_scores))
print(len(labels_test))
print(ParTXbb_scores)
fpr=[]
tpr=[]
threshold=[]

device = df.get_device()



LatentXbb_test = df.get_latent_feat_Xbb('../../Finetune_hep/models/ParTXbb/ParT_latent_scores_0_test.npy')[evts_test]
LatentXbb_model = ParT_mlp.make_mlp(len(LatentXbb_test[0]),128,0)
LatentXbb_model.to(device)
LatentXbb_model.load_state_dict(torch.load('../../Finetune_hep/models/LatentXbb/LatentXbb_hl0_nodes128_nj1_lr4e-05_bs512_training_0.pt'))
LatentXbb_model.eval()
'''
LatentXbb_6_model = ParT_mlp.make_mlp(len(LatentXbb_test[0]),128,6)
LatentXbb_6_model.to(device)
LatentXbb_6_model.load_state_dict(torch.load('../../Finetune_hep/models/LatentXbb/LatentXbb_hl6_nodes128_nj1_lr4e-05_bs512_training_0.pt'))
LatentXbb_6_model.eval()
'''

yi = ParTXbb_scores
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)
'''
yi = Mlp.get_preds(LatentXbb_6_model,LatentXbb_test,evts_test,device)
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)
'''
yi = Mlp.get_preds(LatentXbb_model,LatentXbb_test,evts_test,device)
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)


yi = Xbb_scores
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)

# plot ROC curve
color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))

fig, ax = plt.subplots()
ax.plot(tpr[0], 1/fpr[0], lw=0.8, label=('ParTXbb AUC = {:.2f}%'.format(auc(fpr[0],tpr[0])*100)), color=next(color))
ax.plot(tpr[1], 1/fpr[1], lw=0.8, label=('LatentXbb AUC = {:.2f}%'.format(auc(fpr[1],tpr[1])*100)), color=next(color))
ax.plot(tpr[2], 1/fpr[2], lw=0.8, label=('Xbb AUC = {:.2f}%'.format(auc(fpr[2],tpr[2])*100)), color=next(color))
#ax.plot(tpr[3], 1/fpr[3], lw=0.8, label=('LatentXbb_6hl AUC = {:.3f}%'.format(auc(fpr[3],tpr[3])*100)), color=next(color))
ax.set_xlabel(r'Signal Efficiency')
ax.set_ylabel(r'1/FPR')
ax.semilogy()
ax.set_xlim(0,1)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/ROC_Xbb.pdf")
fig.savefig(f"../../Finetune_hep/plots/ROC_Xbb.png")

