from Finetune_hep.python import ParT_Xbb
from Finetune_hep.python import ParT_mlp
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

def GetParser():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', dest='config_file', required=True, help='YAML configuration file')
    return parser.parse_args()

args=GetParser()
with open(args.config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)

subset= config['subset']
filelist_test= config['data-test']
modeltype = config['model']
config_path = config['config-path']
model_path = config['model-path']
name = config['out-name']
scaler_path = config['scaler-path']
Xbb_scores_path = config['Xbb-scores-path']

print('subset: ',subset)

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]  

device = df.get_device()

if modeltype == 'ParTevent':
    with open(config_path) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)  

    model = ParT_mlp.get_model(data_config,for_inference=False)  
    model.to(device)
    Xbb = False
    model.load_state_dict(torch.load(model_path))
    model.eval()

    idxmap = df.get_idxmap(filelist_test)
    integer_file_map = df.create_integer_file_map(idxmap)
    Dataset = df.CustomDataset(idxmap,integer_file_map)

    train_loader = DataLoader(Dataset, batch_size=512, shuffle=True,num_workers=6)
    build_features = df.build_features_and_labels
    yi,target = ParT_mlp.get_preds(model,train_loader,device,subset,build_features)



elif modeltype in ['mlpXbb','mlpHlXbb','baseline']:
    nodes_mlp = 24
    nlayer_mlp = 3
    model = Mlp.InvariantModel( phi=Mlp.make_mlp(6,nodes_mlp,nlayer_mlp,binary=False),
                                rho=Mlp.make_mlp(nodes_mlp,nodes_mlp*2,nlayer_mlp))
    if modeltype == 'mlpXbb': model = Mlp.make_mlp(2,nodes_mlp,nlayer_mlp)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if modeltype == 'baseline': 
        Dataset_baseline = Mlp.CustomDataset(filelist_test,
                                device,
                                scaler_path=scaler_path,
                                Xbb_scores_path='no',
                                test=True)

        train_loader_baseline = DataLoader(Dataset_baseline, batch_size=512, shuffle=True)

        yi,target = Mlp.get_preds(model,train_loader_baseline,subset,device)  

    if modeltype == 'mlpHlXbb': 
        Dataset_mlpHlXbb = Mlp.CustomDataset(filelist_test,
                            device,
                            scaler_path=scaler_path,
                            Xbb_scores_path=Xbb_scores_path,
                            test=True)

        train_loader_mlpHlXbb = DataLoader(Dataset_mlpHlXbb, batch_size=512, shuffle=True)

        yi,target = Mlp.get_preds(model,train_loader_mlpHlXbb,subset,device)      

elif modeltype in ['mlpLatent']:
    model = ParT_mlp.make_mlp(256,nodes_mlp,nlayer_mlp)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()


else:
    print('specify a model (ParTevent,mlpXbb,mlpHlXbb,baseline)')    

print('device: ', device)

Data = h5py.File(f'../../Finetune_hep/models/{modeltype}/test_{name}.h5', 'w')
Data.create_dataset('evt_score', data=yi.reshape(-1))
Data.create_dataset('evt_label', data=target.reshape(-1),dtype='i4')
Data.close()        

