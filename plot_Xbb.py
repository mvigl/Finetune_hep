import sys, os
sys.path.append('/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/python')

import ParT_Xbb
import ParT_mlp
import ParT_mlp_aux
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
print('oi')
X_pfo_test , X_jet_test , njets_test , labels_test , X_label_test , evts_test  = df.get_test_data('/home/iwsatlas1/mavigl/Hbb/ParT/Dataset',subset)
print('oi')
config_path='../../Finetune_hep/config/myJetClass_full.yaml'
device = df.get_device()
with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  
print('oi')
#Aux_model = ParT_mlp_aux.get_model(data_config,for_inference=False)  
Aux3_model = ParT_mlp_aux.get_model(data_config,for_inference=False)  
Aux_test = df.build_features_and_labels(X_pfo_test[evts_test][:,:2],X_jet_test[evts_test][:,:2],X_label_test[evts_test][:,:2],2)
evts_test_aux = np.copy(evts_test)
#Aux_model.to(device)
Aux3_model.to(device)
print('oi')
X_jet_test = np.reshape(X_jet_test[evts_test][:,:2],(-1,len(df.jVars)))
X_pfo_test = np.reshape(X_pfo_test[evts_test][:,:2],(-1,110,len(df.pVars)))
X_label_test = np.reshape(X_label_test[evts_test][:,:2],(-1,len(df.labelVars)))
evts_test = np.where(X_pfo_test[:,0,df.pVars.index('pfcand_ptrel')] != 0 )[0]
labels_test = X_label_test[evts_test,df.labelVars.index('label_H_bb')]

Xbb_scores = np.reshape(X_jet_test[evts_test,df.jVars.index('fj_doubleb')],(-1))
ParTXbb_scores = np.reshape(df.getXbb_scores('../../Finetune_hep/models/ParTXbb/ParTXbb_scores_0_test.npy',evts_test)[:,:2],(-1))[evts_test] 
print(len(ParTXbb_scores))
print(len(labels_test))
print(ParTXbb_scores)
fpr=[]
tpr=[]
threshold=[]



LatentXbb_test = df.get_latent_feat_Xbb('../../Finetune_hep/models/LatentXbb/LatentXbb_scores_0_test.npy')[evts_test]
LatentXbb_model = ParT_mlp.make_mlp(len(LatentXbb_test[0]),128,0)
LatentXbb_model.to(device)
LatentXbb_model.load_state_dict(torch.load('../../Finetune_hep/models/LatentXbb/LatentXbb_hl0_nodes128_nj1_lr4e-05_bs512_training_0.pt'))
LatentXbb_model.eval()

LatentXbb_Aux_test = df.get_latent_feat_Xbb('../../Finetune_hep/models/LatentXbb_Aux/LatentXbb_Aux_scores_0_test.npy')[evts_test]
LatentXbb_Aux_model = ParT_mlp.make_mlp(len(LatentXbb_test[0]),128,0)
LatentXbb_Aux_model.to(device)
LatentXbb_Aux_model.load_state_dict(torch.load('../../Finetune_hep/models/LatentXbb_Aux/LatentXbb_Aux_hl0_nodes128_nj1_lr4e-05_bs512_training_0.pt'))
LatentXbb_Aux_model.eval()

LatentXbb_6_model = ParT_mlp.make_mlp(len(LatentXbb_test[0]),128,0)
LatentXbb_6_model.to(device)
LatentXbb_6_model.load_state_dict(torch.load('../../Finetune_hep/models/LatentXbb/LatentXbb_hl0_nodes128_nj1_lr4e-05_bs512_training_0.pt'))
LatentXbb_6_model.eval()

#Aux_model.load_state_dict(torch.load(f'../../Finetune_hep/models/Aux/Aux_hl3_nodes128_nj2_lr4e-05_bs512_alpha0.01_WparT_training_0.pt'))
Aux3_model.load_state_dict(torch.load(f'../../Finetune_hep/models/Aux/Aux_hl3_nodes128_nj2_lr4e-05_bs512_alpha0.01_WparT_hlXbb3_training_0.pt'))


yi = ParTXbb_scores
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)

yi = Mlp.get_preds(LatentXbb_model,LatentXbb_test,evts_test,device)
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)

yi = Mlp.get_preds(LatentXbb_Aux_model,LatentXbb_Aux_test,evts_test,device)
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)


yi = Xbb_scores
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)

'''
yi = ParT_mlp_aux.get_preds(Aux_model,Aux_test,evts_test_aux,device)[1]
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)
'''

yi = ParT_mlp_aux.get_preds(Aux3_model,Aux_test,evts_test_aux,device)[1]
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)


yi = Mlp.get_preds(LatentXbb_6_model,LatentXbb_test,evts_test,device)
fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
fpr.append(fpr_i)
tpr.append(tpr_i)
threshold.append(threshold_i)

# plot ROC curve
color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))

fig, ax = plt.subplots()
ax.plot(tpr[0], 1/fpr[0], lw=0.8, label=('ParTXbb AUC = {:.2f}%'.format(auc(fpr[0],tpr[0])*100)), color=next(color))
ax.plot(tpr[1], 1/fpr[1], lw=0.8, label=('LatentXbb AUC = {:.2f}%'.format(auc(fpr[1],tpr[1])*100)), color=next(color))
ax.plot(tpr[2], 1/fpr[2], lw=0.8, label=('LatentXbb_Aux AUC = {:.2f}%'.format(auc(fpr[2],tpr[2])*100)), color=next(color))
ax.plot(tpr[3], 1/fpr[3], lw=0.8, label=('Xbb AUC = {:.2f}%'.format(auc(fpr[3],tpr[3])*100)), color=next(color))
#ax.plot(tpr[4], 1/fpr[4], lw=0.8, label=('Aux AUC = {:.2f}%'.format(auc(fpr[4],tpr[4])*100)), color=next(color))
ax.plot(tpr[4], 1/fpr[4], lw=0.8, label=('Aux3 AUC = {:.2f}%'.format(auc(fpr[4],tpr[4])*100)), color=next(color))
#ax.plot(tpr[6], 1/fpr[6], lw=0.8, label=('LatentXbb_6hl AUC = {:.2f}%'.format(auc(fpr[6],tpr[6])*100)), color=next(color))
ax.set_xlabel(r'Signal Efficiency')
ax.set_ylabel(r'1/FPR')
ax.semilogy()
ax.set_xlim(0,1)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/ROC_Xbb.pdf")
fig.savefig(f"../../Finetune_hep/plots/ROC_Xbb.png")

