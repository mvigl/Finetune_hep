from Finetune_hep.python import ParT_Xbb
from Finetune_hep.python import ParT_mlp
from Finetune_hep.python import definitions as df
from torch.utils.data import Dataset, DataLoader
import os
import sys
import torch
import yaml
import h5py
import argparse


print('start')

def GetParser():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', dest='config_file', required=True, help='YAML configuration file')
    parser.add_argument('--size', dest='size', required=True, help='data size')
    return parser.parse_args()

args=GetParser()
with open(args.config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)

size = args.size
subset= config['subset']
filelist_train= config['data-train']
filelist_test= config['data-test']
filelist_val= config['data-val']
config_path = config['config-path']
modeltype = config['modeltype']
name = str(size)

print('subset: ',subset)

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]  

device = df.get_device()
with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  
ParTXbb_model = ParT_Xbb.get_model(data_config,for_inference=True)  

print('get model')

fpr=[]
tpr=[]
threshold=[]

ParTXbb_model.to(device)
ParTXbb_model.eval()
if modeltype == 'ParTevent_Xbb_Hl':
    model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_Xbb_Hl_subset_1/models/ParTevent_Xbb_Hl_hl3_nodes128_nj5_lr0.001_bs256_WparT_Wmlp_training_1subset_{size}.pt'

elif modeltype == 'ParTevent_Xbb_Hl_double':
    model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_Xbb_Hl_subset_double_1/models/ParTevent_Xbb_Hl_hl3_nodes128_nj5_lr0.001_bs256_WparT_Wmlp_training_1subset_{size}.pt'

elif modeltype == 'ParTevent_Xbb_Hl_scratch':
    model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_Xbb_Hl_scratch_subset_1/models/ParTevent_Xbb_Hl_hl3_nodes128_nj5_lr0.001_bs256_training_1subset_{size}.pt'

ParTXbb_model = df.load_Xbb_backbone(ParTXbb_model,'Xbb',mlp_layers=1,ParT_params_path=model_path,mlp_params_path='no')
#ParTXbb_model.load_state_dict(torch.load(model_path))   

print('device: ', device)

#out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTXbb_scratch_etoe/{name}/'
#yi_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_train,device,out_dir,Xbb=True)

out_out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/{modeltype}'
if (not os.path.exists(out_out_dir)): os.system(f'mkdir {out_out_dir}')
out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/{modeltype}/{name}/'
if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
yi_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_test,device,out_dir,Xbb=True)

#out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTXbb_scratch_etoe/{name}/'
#yi_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_val,device,out_dir,Xbb=True)



      