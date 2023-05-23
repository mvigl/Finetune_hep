#!/opt/anaconda3/bin/python

import sys, os
sys.path.append('/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/python')

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import ParT_mlp
import ParT_Xbb
import Mlp
import definitions as df
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
import torch.nn as nn
import argparse
import yaml
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--modeltype', help='modeltype',default='ParTevent')
parser.add_argument('--nlayer_mlp', type=int, help='nlayer_mlp',default=6)
parser.add_argument('--nodes_mlp', type=int, help='nodes_mlp',default=128)
parser.add_argument('--weights',  help='weights',default='no')
parser.add_argument('--config', help='config',default='../../Finetune_hep/config/myJetClass_full.yaml')
parser.add_argument('--data', help='data',default='/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')


args = parser.parse_args()

nlayer_mlp = args.nlayer_mlp
nodes_mlp = args.nodes_mlp
config_path = args.config
modeltype = args.modeltype
weights = args.weights
data = args.data


X_pfo_train, X_jet_train, njets_train, labels_train, X_label_train, evts_train = df.get_train_data(data)

device = df.get_device()
if modeltype == 'ParTXbb':
    with open(config_path) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)  

    model = ParT_Xbb.get_model(data_config,for_inference=False) 
    X_jet_train = np.reshape(X_jet_train,(-1,len(df.jVars)))
    X_pfo_train = np.reshape(X_pfo_train,(-1,110,len(df.pVars)))
    X_label_train = np.reshape(X_label_train,(-1,len(df.labelVars)))
    evts_train = np.where(X_pfo_train[:,0,df.pVars.index('pfcand_ptrel')] != 0 )[0]
    model.to(device)
    model.load_state_dict(torch.load(weights))
    train = df.build_features_and_labels_single_jet(X_pfo_train[evts_train],X_jet_train[evts_train],X_label_train[evts_train])

else:
    print('specify a model (ParTXbb,...)')    




name = f'{modeltype}_weights'
model_path = (f'../../Finetune_hep/models/{modeltype}/{name}.npy' )

preds = ParT_mlp.get_preds(model,train,evts_train,device)

with open(model_path, 'wb') as f:
    np.save(f, preds)
