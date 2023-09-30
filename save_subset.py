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

#(1730 19332 195762 1959955 2704 29145 293774 2940006 4665 48752 489801 4900263 5880252 6860297 777 7840400 8820463 9547 97752 979854)


#for size in sizes:
model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_subset_1/models/ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_{size}.pt'
with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  

model = ParT_mlp.get_model(data_config,for_inference=True)  
model.to(device)
Xbb = False
model.load_state_dict(torch.load(model_path))
model.eval()

out_dir = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/subsets/{name}/ParTevent'

yParT,targetParT = ParT_mlp.get_Xbb_preds(model,filelist_test,device,subset,out_dir)

#for size in sizes:
model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpHlXbb_subset_1/models/mlpHlXbb_hl3_nodes24_nj5_lr0.001_bs512_training_1subset_{size}.pt'
scaler_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpHlXbb_subset_1/models/mlpHlXbb_hl3_nodes24_nj5_lr0.001_bs512_training_1subset_{size}.pkl'

model = Mlp.InvariantModel( phi=Mlp.make_mlp(6,nodes_mlp,nlayer_mlp,binary=False),
                                rho=Mlp.make_mlp(nodes_mlp,nodes_mlp*2,nlayer_mlp,for_inference=True))
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()



