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

fpr=[]
tpr=[]
threshold=[]
config_path='../../Finetune_hep/config/myJetClass_full.yaml'
device = df.get_device()

with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  

ParTevent_model = ParT_mlp.get_model(data_config,for_inference=False)  
ParTevent_test = df.build_features_and_labels(X_pfo_test[evts_test][:,:2],X_jet_test[evts_test][:,:2],X_label_test[evts_test][:,:2],2)
ParTevent_model.to(device)


mlpXbb_test = df.get_mlp_feat(X_jet_test[evts_test],2,'mlpXbb',evts_test,'../../Finetune_hep/models/ParTXbb/ParTXbb_scores_test.npy',subset)
mlpHlXbb_test = df.get_mlp_feat(X_jet_test[evts_test],2,'mlpHlXbb',evts_test,'../../Finetune_hep/models/ParTXbb/ParTXbb_scores_test.npy',subset)
baseline_test = df.get_mlp_feat(X_jet_test[evts_test],2,'baseline',evts_test,'no',subset)
mlpLatent_test = df.get_latent_feat('../../Finetune_hep/models/ParTXbb/ParT_latent_scores_test.npy',2,subset)

mlpXbb_model = ParT_mlp.make_mlp(len(mlpXbb_test[0]),12,6)
mlpHlXbb_model = ParT_mlp.make_mlp(len(mlpHlXbb_test[0]),24,6)
baseline_model = ParT_mlp.make_mlp(len(baseline_test[0]),24,6)
mlpLatent_model = ParT_mlp.make_mlp(len(mlpLatent_test[0]),128,6)
mlpXbb_model.to(device)
mlpHlXbb_model.to(device)
baseline_model.to(device)
mlpLatent_model.to(device)

mlpXbb_model.load_state_dict(torch.load('../../Finetune_hep/models/mlpXbb/mlpXbb_hl6_nodes12_nj2_lr0.0006_bs512_training_0.pt'))
mlpHlXbb_model.load_state_dict(torch.load('../../Finetune_hep/models/mlpHlXbb/mlpHlXbb_hl6_nodes24_nj2_lr0.0006_bs512_training_0.pt'))
baseline_model.load_state_dict(torch.load('../../Finetune_hep/models/baseline/baseline_hl6_nodes24_nj2_lr0.0006_bs512_training_0.pt'))
mlpLatent_model.load_state_dict(torch.load('../../Finetune_hep/models/mlpLatent/mlpLatent_hl6_nodes128_nj2_lr4e-05_bs512_training_0.pt'))

with open('../../Finetune_hep/models/mlpXbb/mlpXbb_hl6_nodes12_nj2_lr0.0006_bs512_training_0.pkl','rb') as f:
    scaler = pickle.load(f)
mlpXbb_test = scaler.transform(mlpXbb_test)  
with open('../../Finetune_hep/models/mlpHlXbb/mlpHlXbb_hl6_nodes24_nj2_lr0.0006_bs512_training_0.pkl','rb') as f:
    scaler = pickle.load(f)
mlpHlXbb_test = scaler.transform(mlpHlXbb_test)  
with open('../../Finetune_hep/models/baseline/baseline_hl6_nodes24_nj2_lr0.0006_bs512_training_0.pkl','rb') as f:
    scaler = pickle.load(f)
baseline_test = scaler.transform(baseline_test)  


models = [ParTevent_model,mlpLatent_model,mlpHlXbb_model,mlpXbb_model,baseline_model]
test_set = [ParTevent_test,mlpLatent_test,mlpHlXbb_test,mlpXbb_test,baseline_test]
models_name = ['ParTevent','mlpLatent','mlpHlXbb','mlpXbb','baseline']
m=0
for mod,test in zip(models,test_set):
    print(m)
    mod.eval()
    if models_name[m] == 'ParTevent':
        for i in range(5):
            mod.load_state_dict(torch.load(f'../../Finetune_hep/models/ParTevent/ParTevent_hl3_nodes128_nj2_lr4e-05_bs512_WparT_training_{i}.pt'))
            yi = ParT_mlp.get_preds(mod,test,evts_test,device)
            fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi)
            fpr.append(fpr_i)
            tpr.append(tpr_i)
            threshold.append(threshold_i)
    else: 
        yi = Mlp.get_preds(mod,test,evts_test,device)    
        fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi)
        fpr.append(fpr_i)
        tpr.append(tpr_i)
        threshold.append(threshold_i)
    m+=1

# plot ROC curve
color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))
plt.figure()
start=0
for i in range(len(tpr)):
    plt.plot(tpr[i], 1/fpr[i], lw=0.8, label=(f'{models_name[i-4] if i>4 else models_name[0]} ' + "AUC = {:.3f}%".format(auc(fpr[i],tpr[i])*100)), color=next(color))
plt.xlabel(r'Signal Efficiency')
plt.semilogy()
plt.xlim(0,1)
plt.grid(True)
plt.legend(loc='lower left')
plt.title(f"ROC")
plt.show()
plt.savefig(f"../../Finetune_hep/plots/ROC.pdf")

color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))
plt.figure()
start=0
for i in range(len(tpr)):
    plt.plot(tpr[i], 1/fpr[i], lw=0.8, label=(f'{models_name[i-4] if i>4 else models_name[0]} ' + "AUC = {:.3f}%".format(auc(fpr[i],tpr[i])*100)), color=next(color))
plt.xlabel(r'Signal Efficiency')
plt.semilogy()
plt.xlim(0.8,1)
plt.ylim(1,1000)
plt.grid(True)
plt.legend(loc='lower left')
plt.title(f"ROC")
plt.show()
plt.savefig(f"../../Finetune_hep/plots/ROC_zoom.pdf")