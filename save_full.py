from Finetune_hep.python import ParT_Xbb
from Finetune_hep.python import ParT_mlp
from Finetune_hep.python import ParT_mlp_Hl
from Finetune_hep.python import ParT_mlp_Xbb_Hl
from Finetune_hep.python import ParT_mlp_Xbb_Hl_sigmoid
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
    return parser.parse_args()
args=GetParser()
with open(args.config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)

modeltype = config['modeltype']
Ntraining = config['Ntraining']
filelist_test = config['filelist-test']
config_path = config['config-path'] 
model_path = config['model-path']
Xbb_scores_path = config['Xbb-scores-path']

with open(config_path) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)  

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]  
device = df.get_device()

if modeltype == 'mlpHlXbb':
    scaler_path = 'no'#model_path.replace(".pt", ".pkl")
    model = Mlp.InvariantModel( phi=Mlp.make_mlp(6,24,4,for_inference=False,binary=False),
                                rho=Mlp.make_mlp(24,24*2,4,for_inference=True,binary=True))
    
elif modeltype == 'mlpLatent':
    scaler_path = 'no'
    model = Mlp.InvariantModel( phi=Mlp.make_mlp(128,128,3,for_inference=False,binary=False),
                                rho=Mlp.make_mlp(128,128,3,for_inference=True,binary=True)) 

elif modeltype == 'mlpLatentHl':
    scaler_path = 'no'
    model = Mlp.InvariantModel( phi=Mlp.make_mlp(128+5,128,3,for_inference=False,binary=False),
                                rho=Mlp.make_mlp(128,128,3,for_inference=True,binary=True))     

elif modeltype == 'ParTevent':
    model = ParT_mlp.get_model(data_config,for_inference=True)  
    Xbb = False

elif modeltype == 'ParTevent_scratch':
    model = ParT_mlp.get_model(data_config,for_inference=True)  
    Xbb = False

elif modeltype == 'ParTevent_Hl':
    model = ParT_mlp_Hl.get_model(data_config,for_inference=True)  
    Xbb = False

elif modeltype == 'ParTevent_Hl_double':
    model = ParT_mlp_Hl.get_model(data_config,for_inference=True)  
    Xbb = False

elif modeltype == 'ParTevent_Hl_scratch':
    model = ParT_mlp_Hl.get_model(data_config,for_inference=True)  
    Xbb = False

elif modeltype == 'ParTevent_Xbb_Hl':
    model = ParT_mlp_Xbb_Hl_sigmoid.get_model(data_config,for_inference=True)  
    Xbb = False 

elif modeltype == 'ParTevent_Xbb_Hl_double':
    model = ParT_mlp_Xbb_Hl_sigmoid.get_model(data_config,for_inference=True)  
    Xbb = False     

elif modeltype == 'ParTevent_Xbb_Hl_scratch':
    model = ParT_mlp_Xbb_Hl.get_model(data_config,for_inference=True)  
    Xbb = False 

elif modeltype == 'ParTevent_paper':
    model = ParT_mlp.get_model(data_config,for_inference=True)  
    Xbb = False
    
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

out_out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_{modeltype}_{Ntraining}'
if (not os.path.exists(out_out_dir)): os.system(f'mkdir {out_out_dir}')
out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_{modeltype}_{Ntraining}/9800758/'
if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')

if modeltype in ['mlpHlXbb','mlpLatent','mlpLatentHl']:
    y = Mlp.get_Mlp_preds(model,filelist_test,device,out_dir,Xbb_scores_path,scaler_path,modeltype)
elif modeltype in ['ParTevent_Hl','ParTevent_Hl_double','ParTevent_Hl_scratch','ParTevent_Xbb_Hl','ParTevent_Xbb_Hl_scratch','ParTevent_Xbb_Hl_double']:
    y = ParT_mlp_Hl.get_Xbb_preds(model,filelist_test,device,out_dir)    
else:
    y = ParT_mlp.get_Xbb_preds(model,filelist_test,device,out_dir)
