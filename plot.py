import sys, os
sys.path.append('/home/iwsatlas1/mavigl/Finetune_hep/source/python')

import ParT_Xbb
import ParT_mlp
import definitions as df
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
import torch.nn as nn
import yaml


hyper_params = {
   "learning_rate": 0.00004,
   "steps": 35,
   "batch_size": 512,
}

device = df.get_device()
with open('/home/iwsatlas1/mavigl/Finetune_hep/source/config/myJetClass_full.yaml') as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)
model = ParT_mlp.get_model(data_config,for_inference=False)    
model.to(device)
model.eval()

model_params_path=(f'/home/iwsatlas1/mavigl/Finetune_hep/source/models/COMBINED_TRAINING_{hyper_params["learning_rate"]}_{hyper_params["batch_size"]}.pt')
model.load_state_dict(torch.load(model_params_path))

X_pfo_test , X_jet_test , njets_test , labels_test , X_label_test , evts_test  = df.get_test_data('/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')
test = df.build_features_and_labels(X_pfo_test[evts_test][:,:2],X_jet_test[evts_test][:,:2],X_label_test[evts_test][:,:2],2)
yi_ParT_test = df.get_preds(model,test,evts_test,device)

from sklearn.metrics import roc_curve, auc
disc_ParT = yi_ParT_test
fpr_ParT, tpr_ParT, threshold_ParT = roc_curve(labels_test, disc_ParT)
# plot ROC curve
plt.figure()
start=0
plt.plot(tpr_ParT[start:], 1/fpr_ParT[start:], lw=0.8, label="hl feats + Part_Xbb , AUC = {:.2f}%".format(auc(fpr_ParT,tpr_ParT)*100), color = 'orange')
plt.xlabel(r'Signal Efficiency')
plt.semilogy()
plt.xlim(0,1)
plt.grid(True)
plt.legend(loc='lower left')
plt.show()
plt.savefig(f"ROC_COMBINED_TRAINING_{ hyper_params['learning_rate'] }_{hyper_params['batch_size']}.png")