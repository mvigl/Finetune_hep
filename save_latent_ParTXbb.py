from Finetune_hep.python import ParT_latent
from Finetune_hep.python import ParT_mlp
from Finetune_hep.python import definitions as df
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
    return parser.parse_args()

args=GetParser()
with open(args.config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)

subset= config['subset']
filelist_train= config['data-train']
filelist_test= config['data-test']
filelist_val= config['data-val']
config_path = config['config-path']
model_path = config['model-path']
name = config['out-name']

print('subset: ',subset)

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]  

device = df.get_device()
with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  
ParTXbb_model = ParT_latent.get_model(data_config,for_inference=True)  

print('get model')

fpr=[]
tpr=[]
threshold=[]

ParTXbb_model.to(device)
ParTXbb_model.eval()
ParTXbb_model = df.load_Xbb_backbone(ParTXbb_model,'',mlp_layers=1,ParT_params_path=model_path,mlp_params_path='no')
#ParTXbb_model.load_state_dict(torch.load(model_path))   

print('device: ', device)

#out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/Latent_ParTXbb_double/{name}/'
#yi_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_train,device,out_dir,Latent=True)

out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/Latent_ParTXbb_finetuned/{name}/'
yi_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_test,device,out_dir,Latent=True)

#out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/Latent_ParTXbb_double/{name}/'
#yi_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_val,device,out_dir,Latent=True)



      