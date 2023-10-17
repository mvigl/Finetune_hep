from Finetune_hep.python import ParT_Xbb
from Finetune_hep.python import ParT_mlp
from Finetune_hep.python import ParT_latent
from Finetune_hep.python import Mlp
from Finetune_hep.python import definitions as df
from torch.utils.data import Dataset, DataLoader
import os
import sys
import torch
import yaml
import h5py
import argparse

def GetParser():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--config', dest='config_file', required=True, help='YAML configuration file')
    parser.add_argument('--size', dest='size', required=True, help='data size')
    return parser.parse_args()
args=GetParser()
with open(args.config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)

size = args.size
modeltype = config['modeltype']
Ntraining = config['Ntraining']
filelist_test = config['filelist-test']
config_path = config['config-path'] 
Xbb_scores_path = config['Xbb-scores-path']
sample = 'test'

with open(config_path) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)  

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]  
device = df.get_device()

if modeltype == 'mlpHlXbb':
    model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpHlXbb_subset_{Ntraining}/models/mlpHlXbb_hl4_nodes24_nj5_lr0.001_bs512_training_1subset_{size}.pt'
    scaler_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpHlXbb_subset_{Ntraining}/models/mlpHlXbb_hl4_nodes24_nj5_lr0.001_bs512_training_1subset_{size}.pkl'

    model = Mlp.InvariantModel( phi=Mlp.make_mlp(6,24,4,for_inference=False,binary=False),
                                rho=Mlp.make_mlp(24,24*2,4,for_inference=True,binary=True))
    
elif modeltype == 'mlpLatent':
    model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpLatent_subset_{Ntraining}/models/mlpLatent_hl3_nodes128_nj5_lr0.001_bs512_training_1subset_{size}.pt'
    scaler_path = 'no'

    model = Mlp.InvariantModel( phi=Mlp.make_mlp(128,128,3,for_inference=False,binary=False),
                                rho=Mlp.make_mlp(128,128,3,for_inference=True,binary=True)) 

elif modeltype == 'mlpLatentHl':
    model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpLatentHl_subset_{Ntraining}/models/mlpLatentHl_hl3_nodes128_nj5_lr0.001_bs512_training_1subset_{size}.pt'
    scaler_path = 'no'

    model = Mlp.InvariantModel( phi=Mlp.make_mlp(128+5,128,3,for_inference=False,binary=False),
                                rho=Mlp.make_mlp(128,128,3,for_inference=True,binary=True))     

elif modeltype == 'ParTevent':
    model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_subset_{Ntraining}/models/ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_{size}.pt'
    model = ParT_mlp.get_model(data_config,for_inference=True)  
    Xbb = False

elif modeltype == 'ParTevent_scratch':
    model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_scratch_subset_{Ntraining}/models/ParTevent_hl3_nodes128_nj5_lr0.001_bs256_training_1subset_{size}.pt'
    model = ParT_mlp.get_model(data_config,for_inference=True)  
    Xbb = False


model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

out_out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_{modeltype}_{Ntraining}'
if (not os.path.exists(out_out_dir)): os.system(f'mkdir {out_out_dir}')
out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_{modeltype}/{size}/'
if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')

if modeltype in ['mlpHlXbb','mlpLatent','mlpLatentHl']:
    y = Mlp.get_Mlp_preds(model,filelist_test,device,out_dir,Xbb_scores_path,scaler_path,modeltype)
else:
    y = ParT_mlp.get_Xbb_preds(model,filelist_test,device,out_dir)

