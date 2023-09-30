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

name = str(size)

#for size in sizes:
model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_subset_1/models/ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_{size}.pt'
with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  

model = ParT_mlp.get_model(data_config,for_inference=True)  
model.to(device)
Xbb = False
model.load_state_dict(torch.load(model_path))
model.eval()

idxmap = df.get_idxmap(filelist_test)
integer_file_map = df.create_integer_file_map(idxmap)
Dataset = df.CustomDataset(idxmap,integer_file_map)

train_loader = DataLoader(Dataset, batch_size=512, shuffle=True,num_workers=6)
build_features = df.build_features_and_labels
yParT,targetParT = ParT_mlp.get_Xbb_preds(model,train_loader,device,subset,build_features)


#for size in sizes:
model_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpHlXbb_subset_1/models/mlpHlXbb_hl3_nodes24_nj5_lr0.001_bs512_training_1subset_{size}.pt'
scaler_path = f'/raven/u/mvigl/Finetune_hep_dir/run/mlpHlXbb_subset_1/models/mlpHlXbb_hl3_nodes24_nj5_lr0.001_bs512_training_1subset_{size}.pkl'

model = Mlp.InvariantModel( phi=Mlp.make_mlp(6,nodes_mlp,nlayer_mlp,binary=False),
                                rho=Mlp.make_mlp(nodes_mlp,nodes_mlp*2,nlayer_mlp,for_inference=True))
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

Dataset_mlpHlXbb = Mlp.CustomDataset(filelist_test,
                        device,
                        scaler_path=scaler_path,
                        Xbb_scores_path=Xbb_scores_path,
                        test=True)

train_loader_mlpHlXbb = DataLoader(Dataset_mlpHlXbb, batch_size=512, shuffle=True)
ymlpHlXbb,targetmlpHlXbb = Mlp.get_preds(model,train_loader_mlpHlXbb,subset,device)  


Data = h5py.File(f'../../Finetune_hep/models/subsets/test_{name}.h5', 'w')
Data.create_dataset('ParTevent_evt_score', data=yParT.reshape(-1))
Data.create_dataset('ParTevent_evt_label', data=targetParT.reshape(-1),dtype='i4')
Data.create_dataset('mlpHlXbb_evt_score', data=ymlpHlXbb.reshape(-1))
Data.create_dataset('mlpHlXbb_evt_label', data=targetmlpHlXbb.reshape(-1),dtype='i4')
Data.close()     

