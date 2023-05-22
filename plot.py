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
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import cm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float,  help='learning rate',default='0.00004')
parser.add_argument('--bs', type=int,  help='learning rate',default='512')
parser.add_argument('--ep', type=int,  help='learning rate',default='30')
parser.add_argument('--mess', help='message',default='')
parser.add_argument('--Ntrainings', type=int, help='Ntrainings',default='10')

args = parser.parse_args()

learning_rate = args.lr
batch_size = args.bs
epochs = args.ep
message = args.mess
Ntrainings = args.Ntrainings

hyper_params = {
   "learning_rate": learning_rate,
   "steps": epochs,
   "batch_size": batch_size,
}

X_pfo_test , X_jet_test , njets_test , labels_test , X_label_test , evts_test  = df.get_test_data('/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')

test = df.build_features_and_labels(X_pfo_test[evts_test][:,:2],X_jet_test[evts_test][:,:2],X_label_test[evts_test][:,:2],2)

fpr=[]
tpr=[]
threshold=[]
models = [f'/home/iwsatlas1/mavigl/Finetune_hep/run/21_may_sigma/models/COMBINED_TRAINING_{hyper_params["learning_rate"]}_{hyper_params["batch_size"]}_training_{i}.pt' for i in range(Ntrainings)]

device = df.get_device()
with open('/home/iwsatlas1/mavigl/Finetune_hep/source/config/myJetClass_full.yaml') as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)
model = ParT_mlp.get_model(data_config,for_inference=False)    
model.to(device)

for mod in models:
    model.eval()
    model_params_path=mod
    model.load_state_dict(torch.load(model_params_path))

    yi = df.get_preds(model,test,evts_test,device)
    fpr_i, tpr_i, threshold_i = roc_curve(labels_test, yi)
    fpr.append(fpr_i)
    tpr.append(tpr_i)
    threshold.append(threshold_i)

# plot ROC curve
color = iter(cm.rainbow(np.linspace(0, 1, len(tpr))))
plt.figure()
start=0
for i in range(len(tpr)):
    plt.plot(tpr[i], 1/fpr[i], lw=0.8, label="AUC_{:.0f} = {:.3f}%".format(i,auc(fpr[i],tpr[i])*100), color=next(color))
plt.xlabel(r'Signal Efficiency')
plt.semilogy()
plt.xlim(0,1)
plt.grid(True)
plt.legend(loc='lower left')
plt.title(f"ROC_COMBINED_TRAINING_{ hyper_params['learning_rate'] }_{hyper_params['batch_size']}")
plt.show()
plt.savefig(f"ROC_COMBINED_TRAINING_{ hyper_params['learning_rate'] }_{hyper_params['batch_size']}.png")