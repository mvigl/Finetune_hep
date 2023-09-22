from Finetune_hep.python import ParT_Xbb
from Finetune_hep.python import ParT_mlp
from Finetune_hep.python import definitions as df
import torch
import yaml
import h5py
import argparse

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
config_path = config['config-path']
model_path = config['model-path']
name = config['out-name']

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]  

device = df.get_device()
with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  
ParTXbb_model = ParT_Xbb.get_model(data_config,for_inference=True)  

fpr=[]
tpr=[]
threshold=[]

ParTXbb_model.to(device)
ParTXbb_model.eval()
ParTXbb_model.load_state_dict(torch.load(model_path))

yi_ParTXbb,target_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_train,device,subset,Xbb=True)
Data_train = h5py.File(f'../../Finetune_hep/models/ParTXbb/train_{name}.h5', 'w')
Data_train.create_dataset('Xbb', data=yi_ParTXbb.reshape(-1,5))
Data_train.create_dataset('X_label', data=target_ParTXbb.reshape(-1,5),dtype='i4')
Data_train.close()        

yi_ParTXbb,target_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_test,device,subset,Xbb=True)
Data_test = h5py.File(f'../../Finetune_hep/models/ParTXbb/test_{name}.h5', 'w')
Data_test.create_dataset('Xbb', data=yi_ParTXbb.reshape(-1,5))
Data_test.create_dataset('X_label', data=target_ParTXbb.reshape(-1,5),dtype='i4')
Data_test.close()        

      