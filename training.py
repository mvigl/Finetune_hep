#!/opt/anaconda3/bin/python

import sys, os
sys.path.append('/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/python')

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import ParT_mlp
import ParT_mlp_aux
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
parser.add_argument('--lr', type=float,  help='learning rate',default='0.00004')
parser.add_argument('--bs', type=int,  help='batch size',default='512')
parser.add_argument('--ep', type=int,  help='epochs',default='50')
parser.add_argument('--mess', help='message',default='training0')
parser.add_argument('--modeltype', help='modeltype',default='ParTevent')
parser.add_argument('--nlayer_mlp', type=int, help='nlayer_mlp',default=6)
parser.add_argument('--nodes_mlp', type=int, help='nodes_mlp',default=128)
parser.add_argument('--njets_mlp', type=int, help='njets_mlp',default=2)
parser.add_argument('--ParT_weights',  help='ParT_weights',default='../../Finetune_hep/models/ParT_full.pt')
parser.add_argument('--mlp_weights',  help='mlp_weights',default='no')
parser.add_argument('--config', help='config',default='../../Finetune_hep/config/myJetClass_full.yaml')
parser.add_argument('--data', help='data',default='/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')
parser.add_argument('--Xbb', help='Xbb_scores_path',default='/home/iwsatlas1/mavigl/Hbb/ParT/Trained_ParT/data/ParT_Xbb.npy')
parser.add_argument('--project_name', help='project_name',default='Finetune_ParT')
parser.add_argument('--subset',  action='store_true', help='subset', default=False)
parser.add_argument('--api_key', help='api_key',default='r1SBLyPzovxoWBPDLx3TAE02O')
parser.add_argument('--ws', help='workspace',default='mvigl')
parser.add_argument('--alpha', type=float,  help='alpha',default=0.01)

args = parser.parse_args()

learning_rate = args.lr
batch_size = args.bs
epochs = args.ep
message = args.mess
nlayer_mlp = args.nlayer_mlp
nodes_mlp = args.nodes_mlp
njets_mlp = args.njets_mlp
config_path = args.config
modeltype = args.modeltype
ParT_weights = args.ParT_weights
mlp_weights = args.mlp_weights
data = args.data
Xbb_scores_path = args.Xbb
project_name = args.project_name
subset = args.subset
api_key = args.api_key
workspace = args.ws
alpha = args.alpha
filelist = args.filelist

for m,w in zip(['Wmlp_','WparT_'],[mlp_weights,ParT_weights]):  
    if w != 'no':
        message = m + message

idxmap = df.get_idxmap(filelist)
device = df.get_device()

if modeltype in ['ParTevent','ParTXbb','Aux']:
    with open(config_path) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)  

    if modeltype == 'ParTevent':
        model = ParT_mlp.get_model(data_config,for_inference=False)  
        model.to(device)
        model = df.load_weights_ParT_mlp(model,modeltype,mlp_layers=1,ParT_params_path=ParT_weights,mlp_params_path=mlp_weights)  

    elif modeltype == 'ParTXbb':
        model = ParT_Xbb.get_model(data_config,for_inference=False) 
        model.to(device)
        model = df.load_weights_ParT_mlp(model,modeltype,mlp_layers=1,ParT_params_path=ParT_weights,mlp_params_path=mlp_weights)

elif modeltype in ['mlpXbb','mlpHlXbb','baseline']:
    model = ParT_mlp.make_mlp(12,nodes_mlp,nlayer_mlp)
    if modeltype == 'mlpXbb': model = ParT_mlp.make_mlp(2,nodes_mlp,nlayer_mlp)
    model.to(device)
    if mlp_weights != 'no' : model.load_state_dict(torch.load(mlp_weights))

elif modeltype in ['mlpLatent']:
    model = ParT_mlp.make_mlp(256,nodes_mlp,nlayer_mlp)
    model.to(device)
    if mlp_weights != 'no' : model.load_state_dict(torch.load(mlp_weights))

elif modeltype in ['LatentXbb','LatentXbb_Aux']:
    model = ParT_mlp.make_mlp(128,nodes_mlp,nlayer_mlp)
    model.to(device)
    if mlp_weights != 'no' : model.load_state_dict(torch.load(mlp_weights))

else:
    print('specify a model (ParTevent,ParTXbb,mlpXbb,mlpHlXbb,baseline)')    

experiment = Experiment(
  api_key = api_key,
  project_name = project_name,
  workspace=workspace,
  log_graph=True, # Can be True or False.
  auto_metric_logging=False # Can be True or False
)

hyper_params = {
   "learning_rate": learning_rate,
   "steps": epochs,
   "batch_size": batch_size,
   "alpha": alpha
}


experiment_name = f'{modeltype}_hl{nlayer_mlp}_nodes{nodes_mlp}_nj{njets_mlp}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_{message}'
if modeltype == 'Aux': experiment_name = f'{modeltype}_hl{nlayer_mlp}_nodes{nodes_mlp}_nj{njets_mlp}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_alpha{hyper_params["alpha"]}_{message}'
Experiment.set_name(experiment,experiment_name)

model_path = (f'models/{experiment_name}.pt' )
experiment.log_parameters(hyper_params)
if modeltype not in ['mlpLatent','LatentXbb','LatentXbb_Aux']:
    scaler_path = (f'models/{experiment_name}.pkl' )
else:
    scaler_path = 'no'
    
if modeltype in ['ParTevent','ParTXbb']:
    evals_part, model_part = ParT_mlp.train_loop(
        model,
        idxmap,
        device,
        experiment,
        model_path,
        config = dict(    
            LR = hyper_params['learning_rate'],
            batch_size = hyper_params['batch_size'],
            epochs = hyper_params['steps']
        )
    )

elif modeltype in ['mlpXbb','mlpHlXbb','mlpLatent','baseline','LatentXbb','LatentXbb_Aux']:
    evals_part, model_part = Mlp.train_loop(
        model,
        idxmap,
        device,
        experiment,
        model_path,
        scaler_path,
        config = dict(    
            LR = hyper_params['learning_rate'],
            batch_size = hyper_params['batch_size'],
            epochs = hyper_params['steps']
        )
    )

log_model(experiment, model, model_name = experiment_name )
