import sys, os
sys.path.append('/home/iwsatlas1/mavigl/Finetune_hep/source/python')

import ParT_Xbb
import ParT_mlp
import definitions as df

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import yaml

device = df.get_device()
with open('/home/iwsatlas1/mavigl/Finetune_hep/source/config/myJetClass_full.yaml') as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)
model = ParT_mlp.get_model(data_config,for_inference=False)    
model.to(device)
print(model)
print(device)

X_pfo_train, X_jet_train, njets_train, labels_train, X_label_train, evts_train = df.get_train_data('/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')
X_pfo_test , X_jet_test , njets_test , labels_test , X_label_test , evts_test  = df.get_train_data('/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')


train = df.build_features_and_labels(X_pfo_train[evts_train][:,:2],X_jet_train[evts_train][:,:2],X_label_train[evts_train][:,:2],njets = 2)
test = df.build_features_and_labels(X_pfo_test[evts_test][:,:2],X_jet_test[evts_test][:,:2],X_label_test[evts_test][:,:2],njets = 2)

