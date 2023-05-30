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


mlpXbb_test = df.get_mlp_feat(X_jet_test[evts_test],2,'mlpXbb',evts_test,'../../Finetune_hep/models/ParTXbb/ParTXbb_scores_0_test.npy',subset)
mlpHlXbb_test = df.get_mlp_feat(X_jet_test[evts_test],2,'mlpHlXbb',evts_test,'../../Finetune_hep/models/ParTXbb/ParTXbb_scores_0_test.npy',subset)
baseline_test = df.get_mlp_feat(X_jet_test[evts_test],2,'baseline',evts_test,'no',subset)
mlpLatent_test = df.get_latent_feat('../../Finetune_hep/models/ParTXbb/ParT_latent_scores_0_test.npy',2,subset)

mlpXbb_model = ParT_mlp.make_mlp(len(mlpXbb_test[0]),12,6)
mlpHlXbb_model = ParT_mlp.make_mlp(len(mlpHlXbb_test[0]),24,6)
baseline_model = ParT_mlp.make_mlp(len(baseline_test[0]),24,6)
mlpLatent_model = ParT_mlp.make_mlp(len(mlpLatent_test[0]),128,6)
mlpXbb_model.to(device)
mlpHlXbb_model.to(device)
baseline_model.to(device)
mlpLatent_model.to(device)

mlpXbb_model.load_state_dict(torch.load('../../Finetune_hep/models/mlpXbb/mlpXbb_hl6_nodes12_nj2_lr0.0006_bs512_training_0.pt'))
baseline_model.load_state_dict(torch.load('../../Finetune_hep/models/baseline/baseline_hl6_nodes24_nj2_lr0.0006_bs512_training_0.pt'))

with open('../../Finetune_hep/models/mlpXbb/mlpXbb_hl6_nodes12_nj2_lr0.0006_bs512_training_0.pkl','rb') as f:
    scaler = pickle.load(f)
mlpXbb_test = scaler.transform(mlpXbb_test)  
with open('../../Finetune_hep/models/baseline/baseline_hl6_nodes24_nj2_lr0.0006_bs512_training_0.pkl','rb') as f:
    scaler = pickle.load(f)
baseline_test = scaler.transform(baseline_test)  

Nsigma = 10
models = [ParTevent_model,mlpLatent_model,mlpHlXbb_model,mlpXbb_model,baseline_model]
test_set = [ParTevent_test,mlpLatent_test,mlpHlXbb_test,mlpXbb_test,baseline_test]
models_name = ['ParTevent_0',
               'ParTevent_1',
               'ParTevent_2',
               'ParTevent_3',
               'ParTevent_4',
               'mlpLatent_0',
               'mlpLatent_1',
               'mlpLatent_2',
               'mlpLatent_3',
               'mlpLatent_4',
               'mlpHlXbb_0',
               'mlpHlXbb_1',
               'mlpHlXbb_2',
               'mlpHlXbb_3',
               'mlpHlXbb_4',
               'mlpXbb',
               'baseline']
m=0
for mod,test in zip(models,test_set):
    print(m)
    mod.eval()
    if m == 0:
        for i in range(Nsigma):
            mod.load_state_dict(torch.load(f'../../Finetune_hep/models/ParTevent/ParTevent_hl3_nodes128_nj2_lr4e-05_bs512_WparT_training_{i}.pt'))
            yi = ParT_mlp.get_preds(mod,test,evts_test,device)
            fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
            fpr.append(fpr_i)
            tpr.append(tpr_i)
            threshold.append(threshold_i)
    elif m == 1:
        for i in range(1):
            mod.load_state_dict(torch.load(f'../../Finetune_hep/models/mlpLatent/mlpLatent_hl6_nodes128_nj2_lr4e-05_bs512_training_{i}.pt'))
            yi = Mlp.get_preds(mod,test,evts_test,device)    
            fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
            fpr.append(fpr_i)
            tpr.append(tpr_i)
            threshold.append(threshold_i)

    elif m == 2:
        for i in range(Nsigma):
            with open(f'../../Finetune_hep/models/mlpHlXbb/mlpHlXbb_hl6_nodes24_nj2_lr0.0006_bs512_training_{i}.pkl','rb') as f:
                scaler = pickle.load(f)
            mlpHlXbb_test = df.get_mlp_feat(X_jet_test[evts_test],2,'mlpHlXbb',evts_test,f'../../Finetune_hep/models/ParTXbb/ParTXbb_scores_{i}_test.npy',subset)    
            mlpHlXbb_test = scaler.transform(mlpHlXbb_test)  
            mod.load_state_dict(torch.load(f'../../Finetune_hep/models/mlpHlXbb/mlpHlXbb_hl6_nodes24_nj2_lr0.0006_bs512_training_{i}.pt'))
            yi = Mlp.get_preds(mod,mlpHlXbb_test,evts_test,device)    
            fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
            fpr.append(fpr_i)
            tpr.append(tpr_i)
            threshold.append(threshold_i)

    else:
        yi = Mlp.get_preds(mod,test,evts_test,device)    
        fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi,drop_intermediate=False)
        fpr.append(fpr_i)
        tpr.append(tpr_i)
        threshold.append(threshold_i)
    m+=1

interp_fpr_ParTevent=[]
interp_fpr_mlpHlXbb=[]
ParTevent_bkg_rej=[]
mlpHlXbb_bkg_rej=[]
tpr_plot = np.arange(0,1,0.001)
for i in range(Nsigma):   
    print(i)
    interp_fpr_ParTevent.append(np.interp(tpr_plot, tpr[i], fpr[i]))
    interp_fpr_mlpHlXbb.append(np.interp(tpr_plot, tpr[Nsigma+1+i], fpr[Nsigma+1+i]))

    ParTevent_bkg_rej.append(np.interp(tpr_plot, tpr[i], 1/fpr[i]))
    mlpHlXbb_bkg_rej.append(np.interp(tpr_plot, tpr[Nsigma+1+i], 1/fpr[Nsigma+1+i]))


ParTevent_bkg_rej_mean = np.mean(np.vstack(ParTevent_bkg_rej),axis=0) 
mlpHlXbb_bkg_rej_mean = np.mean(np.vstack(mlpHlXbb_bkg_rej),axis=0) 

ParTevent_bkg_rej_std = np.std(np.vstack(ParTevent_bkg_rej),axis=0) 
mlpHlXbb_bkg_rej_std = np.std(np.vstack(mlpHlXbb_bkg_rej),axis=0) 

ParTevent_fpr_mean = np.mean(np.vstack(interp_fpr_ParTevent),axis=0) 
mlpHlXbb_fpr_mean = np.mean(np.vstack(interp_fpr_mlpHlXbb),axis=0) 

# plot ROC curve
color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))

fig, ax = plt.subplots()

ax.plot(tpr_plot, ParTevent_bkg_rej_mean, lw=0.8, label=('ParTevent_mean AUC = {:.3f}%'.format(auc(ParTevent_fpr_mean,tpr_plot)*100)), color=next(color))
ax.plot(tpr_plot, mlpHlXbb_bkg_rej_mean, lw=0.8, label=('mlpHlXbb_mean AUC = {:.3f}%'.format(auc(mlpHlXbb_fpr_mean,tpr_plot)*100)), color=next(color))

ax.fill_between(tpr_plot, (ParTevent_bkg_rej_mean-ParTevent_bkg_rej_std), (ParTevent_bkg_rej_mean+ParTevent_bkg_rej_std),color=next(color))

ax.set_xlabel(r'Signal Efficiency')
ax.semilogy()
ax.set_xlim(0,1)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/ROC.pdf")

color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))
fig, ax = plt.subplots()

ax.plot(tpr_plot, ParTevent_bkg_rej_mean, lw=0.8, label=('ParTevent_mean AUC = {:.3f}%'.format(auc(ParTevent_fpr_mean,tpr_plot)*100)), color=next(color))
ax.fill_between(tpr_plot, (ParTevent_bkg_rej_mean-ParTevent_bkg_rej_std), (ParTevent_bkg_rej_mean+ParTevent_bkg_rej_std),color=next(color),alpha=0.2)

ax.plot(tpr_plot, mlpHlXbb_bkg_rej_mean, lw=0.8, label=('mlpHlXbb_mean AUC = {:.3f}%'.format(auc(mlpHlXbb_fpr_mean,tpr_plot)*100)), color=next(color))
ax.fill_between(tpr_plot, (mlpHlXbb_bkg_rej_mean-mlpHlXbb_bkg_rej_std), (mlpHlXbb_bkg_rej_mean+mlpHlXbb_bkg_rej_std),color=next(color),alpha=0.2)

ax.set_xlabel(r'Signal Efficiency')
ax.semilogy()
ax.set_xlim(0.7,0.9)
ax.set_ylim(100,1200)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/ROC_zoom.pdf")

color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))
fig, ax = plt.subplots()

ax.plot(tpr_plot, (ParTevent_bkg_rej_mean)/(mlpHlXbb_bkg_rej_mean), lw=0.8, label=('ratio'), color=next(color))
ax.plot(tpr_plot, (mlpHlXbb_bkg_rej_mean)/(mlpHlXbb_bkg_rej_mean), lw=0.8, label=('ratio'), color=next(color))

ax.fill_between(tpr_plot, (ParTevent_bkg_rej_mean-ParTevent_bkg_rej_std)/(mlpHlXbb_bkg_rej_mean), (ParTevent_bkg_rej_mean+ParTevent_bkg_rej_std)/(mlpHlXbb_bkg_rej_mean),color=next(color),alpha=0.2)
ax.fill_between(tpr_plot, (mlpHlXbb_bkg_rej_mean-mlpHlXbb_bkg_rej_std)/(mlpHlXbb_bkg_rej_mean), (mlpHlXbb_bkg_rej_mean+mlpHlXbb_bkg_rej_std)/(mlpHlXbb_bkg_rej_mean),color=next(color),alpha=0.2)

ax.set_xlabel(r'Signal Efficiency')
ax.set_ylim(0.9,2)
ax.grid(True)
ax.legend(loc='lower left')
ax.set_title(f"ROC")
fig.savefig(f"../../Finetune_hep/plots/ROC_ratio.pdf")
