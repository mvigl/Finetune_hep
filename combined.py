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
X_pfo_test , X_jet_test , njets_test , labels_test , X_label_test , evts_test  = df.get_train_data('/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')


train = df.build_features_and_labels(X_pfo_train[evts_train][:,:2],X_jet_train[evts_train][:,:2],X_label_train[evts_train][:,:2],2)
test = df.build_features_and_labels(X_pfo_test[evts_test][:,:2],X_jet_test[evts_test][:,:2],X_label_test[evts_test][:,:2],2)

experiment = Experiment(
  api_key = "r1SBLyPzovxoWBPDLx3TAE02O",
  project_name = "part",
  workspace="mvigl",
  log_graph=True, # Can be True or False.
  auto_metric_logging=False # Can be True or False
)

hyper_params = {
   "learning_rate": 0.00004,
   "steps": 5,
   "batch_size": 512,
}
experiment.log_parameters(hyper_params)


evals_part, model_part = df.train_loop(
    model,
    train,
    labels_train,
    device,
    experiment,
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
plt.savefig(f"COMBINED_TRAINING_{ hyper_params['learning_rate'] }_{hyper_params['batch_size']}.png")