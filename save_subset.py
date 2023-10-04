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


print('start')
parser = argparse.ArgumentParser(description='')
parser.add_argument('--size', type=int, help='size',default=100)
args = parser.parse_args()
size = args.size
name = str(size)
subset = False
filelist_test = '/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt'
config_path = '../../Finetune_hep/config/myJetClass_full.yaml'
Xbb_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTXbb/test_ParTXbb_score_training_1.h5'
sample = 'test'
nodes_mlp = 24
nlayer_mlp = 3

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]  
device = df.get_device()

sizes = [
1730,
19332,
195762,
1959955,
2704,
29145,
293774,
2940006,
4665,
48752,
489801,
4900263,
5880252,
6860297,
777,
7840400,
8820463,
9547,
97752,
979854]

sizes_latent = [
1960151,
196015,
19601,
2940227,
294022,
29402,
2940,
4900379,
490037,
49003,
4900,
5880454,
6860530,
7840606,
8820682,
980075,
98007,
9800,
980,
]

#(1730 19332 195762 1959955 2704 29145 293774 2940006 4665 48752 489801 4900263 5880252 6860297 777 7840400 8820463 9547 97752 979854)

#model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpHlXbb_subset_1/models/mlpHlXbb_hl3_nodes24_nj5_lr0.001_bs512_training_1subset_{size}.pt'
#scaler_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpHlXbb_subset_1/models/mlpHlXbb_hl3_nodes24_nj5_lr0.001_bs512_training_1subset_{size}.pkl'
#
#model = Mlp.InvariantModel( phi=Mlp.make_mlp(6,nodes_mlp,nlayer_mlp,binary=False),
#                                rho=Mlp.make_mlp(nodes_mlp,nodes_mlp*2,nlayer_mlp,for_inference=True))
#model.to(device)
#model.load_state_dict(torch.load(model_path))
#model.eval()
#
#out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/mlpHlXbb/{name}/'
#if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
#
#ymlpHlXbb = Mlp.get_Mlp_preds(model,filelist_test,device,subset,out_dir,Xbb_scores_path,scaler_path)
#
###==============
#
#out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/ParTevent/{name}/'
#if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
#
#model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_subset_1/models/ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_{size}.pt'
#with open(config_path) as file:
#    data_config = yaml.load(file, Loader=yaml.FullLoader)  
#
#model = ParT_mlp.get_model(data_config,for_inference=True)  
#model.to(device)
#Xbb = False
#model.load_state_dict(torch.load(model_path))
#model.eval()
#
#yParT = ParT_mlp.get_Xbb_preds(model,filelist_test,device,subset,out_dir)
#
###==============

#out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/ParTevent_scratch/{name}/'
#if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
#
#model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_subset_scratch_1/models/ParTevent_hl3_nodes128_nj5_lr0.001_bs256_training_1subset_{size}.pt'
#with open(config_path) as file:
#    data_config = yaml.load(file, Loader=yaml.FullLoader)  
#
#model = ParT_mlp.get_model(data_config,for_inference=True)  
#model.to(device)
#Xbb = False
#model.load_state_dict(torch.load(model_path))
#model.eval()
#
#yParT = ParT_mlp.get_Xbb_preds(model,filelist_test,device,subset,out_dir)
#
###==============

#out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTLatent/test/'
#if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
#
#model_path = '/raven/u/mvigl/Finetune_hep_dir/run/ParTevent/models/ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1_epoch_1_Val_loss_0.08823145180940628.pt'
#with open(config_path) as file:
#    data_config = yaml.load(file, Loader=yaml.FullLoader)  
#
#model = ParT_latent.get_model(data_config,for_inference=True)  
#model.to(device)
#Xbb = False
#model = df.load_weights_ParT_mlp(model,'ParTLatent',mlp_layers=1,ParT_params_path=model_path,mlp_params_path='no') 
#model.eval()
#
#yParT = ParT_mlp.get_Latent_preds(model,filelist_test,device,subset,out_dir)


####==============
size_latent = sizes_latent[sizes.index(size)]
name = str(size_latent) 
model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpLatent_subset_1/models/mlpLatent_hl3_nodes24_nj5_lr0.001_bs512_training_1subset_{size_latent}.pt'
scaler_path = 'no'

model = Mlp.InvariantModel_Latent(rho=Mlp.make_mlp(128,nodes_mlp,nlayer_mlp))
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/mlpLatent/{name}/'
if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
Xbb_scores_path = '/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTLatent/test/Latent_test.txt'
ymlpHlXbb = Mlp.get_MlpLatent_preds(model,filelist_test,device,subset,out_dir,Xbb_scores_path,scaler_path)
