#!/opt/anaconda3/bin/python

import sys, os
sys.path.append('/home/iwsatlas1/mavigl/Finetune_hep/source/python')

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import ParT_Xbb
import ParT_mlp
import definitions as df
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float,  help='learning rate',default='0.00004')
parser.add_argument('--bs', type=int,  help='learning rate',default='512')
parser.add_argument('--ep', type=int,  help='learning rate',default='1')
parser.add_argument('--mess', help='message',default='')

args = parser.parse_args()

learning_rate = args.lr
batch_size = args.bs
epochs = args.ep
message = args.mess

import yaml

device = df.get_device()
with open('/home/iwsatlas1/mavigl/Finetune_hep/source/config/myJetClass_full.yaml') as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)
model = ParT_mlp.get_model(data_config,for_inference=False)    
model.to(device)

model_params_path="/home/iwsatlas1/mavigl/Hbb/models/myPart_modelsbest_TRAINING.pt"
for i, layer in enumerate(torch.load(model_params_path).keys()):
    if i > 1:
        model.state_dict()[layer].copy_(torch.load(model_params_path)[layer])

print(model)
print(device)

X_pfo_train, X_jet_train, njets_train, labels_train, X_label_train, evts_train = df.get_train_data('/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')
train = df.build_features_and_labels(X_pfo_train[evts_train][:,:2],X_jet_train[evts_train][:,:2],X_label_train[evts_train][:,:2],2)

experiment = Experiment(
  api_key = "r1SBLyPzovxoWBPDLx3TAE02O",
  project_name = "part",
  workspace="mvigl",
  log_graph=True, # Can be True or False.
  auto_metric_logging=True # Can be True or False
)

hyper_params = {
   "learning_rate": learning_rate,
   "steps": epochs,
   "batch_size": batch_size,
}

model_path = (f'models/COMBINED_TRAINING_{hyper_params["learning_rate"]}_{hyper_params["batch_size"]}_{message}.pt' )
experiment.log_parameters(hyper_params)


evals_part, model_part = df.train_loop(
    model,
    train,
    labels_train,
    device,
    experiment,
    model_path,
    config = dict(    
        LR = hyper_params['learning_rate'],
        batch_size = hyper_params['batch_size'],
        epochs = (hyper_params['steps'])
    )
)

log_model(experiment, model, model_name= (f"COMBINED_TRAINING_{ hyper_params['learning_rate'] }_{hyper_params['batch_size']}.pt" ) )

figure(figsize=(5, 4), dpi=80)
df.plot_evals(evals_part, 'combined')
plt.legend()
plt.semilogy()
plt.savefig(f"plots/LOSS_COMBINED_TRAINING_{ hyper_params['learning_rate'] }_{hyper_params['batch_size']}_{message}.png")