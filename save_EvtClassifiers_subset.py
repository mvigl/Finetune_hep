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

subset = True
filelist_test = '/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt'
config_path = '../../Finetune_hep/config/myJetClass_full.yaml'
Xbb_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTXbb/test_ParTXbb_score_training_1.h5'
sample = 'test'
nodes_mlp = 24
nlayer_mlp = 3

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]  
device = df.get_device()

models_ParTevent = ['ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1_epoch_9_Val_loss_1.2764809131622314.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_1730.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_19332.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_195762.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_1959955.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_2704.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_29145.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_293774.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_2940006.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_4665.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_48752.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_489801.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_4900263.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_5880252.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_6860297.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_777.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_7840400.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_8820463.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_9547.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_97752.pt',
'ParTevent_hl3_nodes128_nj5_lr0.001_bs256_WparT_training_1subset_979854.pt']

yParT = []
targetParT = []
for model_name in models_ParTevent:
    model_path = '/raven/u/mvigl/Finetune_hep_dir/run/ParTevent_subset_1/models/' + model_name
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
    yi,target = ParT_mlp.get_preds(model,train_loader,device,subset,build_features)
    yParT.append(yi)
    targetParT.append(target)

models_mlpHlXbb = []
ymlpHlXbb = []
targetmlpHlXbb = []
for model_name in models_mlpHlXbb:



for modeltype in modeltypes:
    direct = 
    model_path = config['model-path']
    name = config['out-name']
    scaler_path = config['scaler-path']


elif modeltype in ['mlpXbb','mlpHlXbb','baseline','mlpLatent','mlpLatentHl']:
    model = Mlp.InvariantModel( phi=Mlp.make_mlp(6,nodes_mlp,nlayer_mlp,binary=False),
                                rho=Mlp.make_mlp(nodes_mlp,nodes_mlp*2,nlayer_mlp,for_inference=True))
    if modeltype == 'mlpXbb': 
        model = Mlp.InvariantModel( phi=Mlp.make_mlp(1,nodes_mlp,nlayer_mlp,binary=False),
                                    rho=Mlp.make_mlp(nodes_mlp,nodes_mlp*2,nlayer_mlp,for_inference=True))

    if modeltype in ['mlpLatent']:
        model = Mlp.InvariantModel_Latent(rho=Mlp.make_mlp(128,nodes_mlp,nlayer_mlp,for_inference=True))

    if modeltype in ['mlpLatentHl']:
        model = Mlp.InvariantModel_Latent(rho=Mlp.make_mlp(128+6,nodes_mlp,nlayer_mlp,for_inference=True))

    
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

    if modeltype == 'mlpLatent': 
        Dataset_mlpHlXbb = Mlp.CustomDataset_Latent(filelist_test,
                            device,
                            scaler_path=scaler_path,
                            Xbb_scores_path=Xbb_scores_path,
                            test=True)

        train_loader_mlpHlXbb = DataLoader(Dataset_mlpHlXbb, batch_size=512, shuffle=True)

        yi,target = Mlp.get_preds(model,train_loader_mlpHlXbb,subset,device) 

    if modeltype == 'mlpLatentHl': 
        Dataset_mlpHlXbb = Mlp.CustomDataset_Latent_Hl(filelist_test,
                            device,
                            scaler_path=scaler_path,
                            Xbb_scores_path=Xbb_scores_path,
                            test=True)

        train_loader_mlpHlXbb = DataLoader(Dataset_mlpHlXbb, batch_size=512, shuffle=True)

        yi,target = Mlp.get_preds(model,train_loader_mlpHlXbb,subset,device)     

    if modeltype == 'mlpXbb': 
        Dataset_mlpHlXbb = Mlp.CustomDataset_XbbOnly(filelist_test,
                            device,
                            scaler_path=scaler_path,
                            Xbb_scores_path=Xbb_scores_path,
                            test=True)

        train_loader_mlpHlXbb = DataLoader(Dataset_mlpHlXbb, batch_size=512, shuffle=True)

        yi,target = Mlp.get_preds(model,train_loader_mlpHlXbb,subset,device) 

else:
    print('specify a model (ParTevent,mlpXbb,mlpHlXbb,baseline)')    

print('device: ', device)

if modeltype == 'ParTLatent':
    if sample == 'test':
        Data = h5py.File(f'../../Finetune_hep/models/{modeltype}/test_{name}.h5', 'w')
        Data.create_dataset('evt_score', data=yi_test.reshape(-1,5,128))
        Data.create_dataset('jet_mask', data=mask_test.reshape(-1,5))
        Data.create_dataset('evt_label', data=target_test.reshape(-1,1),dtype='i4')
        Data.close() 
    elif sample == 'train':
        Data_train = h5py.File(f'../../Finetune_hep/models/{modeltype}/train_{name}.h5', 'w')
        Data_train.create_dataset('evt_score', data=yi_train.reshape(-1,5,128))
        Data_train.create_dataset('jet_mask', data=mask_train.reshape(-1,5))
        Data_train.create_dataset('evt_label', data=target_train.reshape(-1,1),dtype='i4')
        Data_train.close() 
    elif sample == 'val':
        Data_val = h5py.File(f'../../Finetune_hep/models/{modeltype}/val_{name}.h5', 'w')
        Data_val.create_dataset('evt_score', data=yi_val.reshape(-1,5,128))
        Data_val.create_dataset('jet_mask', data=mask_val.reshape(-1,5))
        Data_val.create_dataset('evt_label', data=target_val.reshape(-1,1),dtype='i4')
        Data_val.close()        

else:
    if subset:
        Data = h5py.File(f'../../Finetune_hep/models/{modeltype}_subset/{sample}_{name}.h5', 'w')
        Data.create_dataset('evt_score', data=yi.reshape(-1))
        Data.create_dataset('evt_label', data=target.reshape(-1),dtype='i4')
        Data.close()  
    else:    
        Data = h5py.File(f'../../Finetune_hep/models/{modeltype}/{sample}_{name}.h5', 'w')
        Data.create_dataset('evt_score', data=yi.reshape(-1))
        Data.create_dataset('evt_label', data=target.reshape(-1),dtype='i4')
        Data.close()        

