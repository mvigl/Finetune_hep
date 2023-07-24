import sys, os
sys.path.append('/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/python')

import ParT_Xbb
import ParT_mlp
import Mlp
import definitions as df
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import cm
import numpy as np
import argparse
import pickle
import h5py
from torch.utils.data import Dataset, DataLoader

subset=False
filelist_train = '/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/config/train_list.txt'
filelist_test = '/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/config/test_list.txt'
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]  

config_path='../../Finetune_hep/config/myJetClass_full.yaml'
device = df.get_device()
with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  
ParTXbb_model = ParT_Xbb.get_model(data_config,for_inference=True)  

fpr=[]
tpr=[]
threshold=[]

ParTXbb_model.to(device)
ParTXbb_model.eval()
ParTXbb_model.load_state_dict(torch.load(f'/home/iwsatlas1/mavigl/Finetune_hep_dir/run/ParTXbb_full/models/ParTXbb_hl0_nodes128_nj1_lr0.001_bs512_WparT_training_0.pt'))

build_features = df.build_features_and_labels_Xbb
yi_ParTXbb,target_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_train,device,subset,Xbb=True)
Data_train = h5py.File(f'../../Finetune_hep/models/ParTXbb/ParTXbb_train_full.h5', 'w')
Data_train.create_dataset('Xbb', data=yi_ParTXbb.reshape(-1,2))
Data_train.create_dataset('X_label', data=target_ParTXbb.reshape(-1,2),dtype='i4')
Data_train.close()        

yi_ParTXbb,target_ParTXbb = ParT_mlp.get_Xbb_preds(ParTXbb_model,filelist_test,device,subset,Xbb=True)
Data_test = h5py.File(f'../../Finetune_hep/models/ParTXbb/ParTXbb_test_full.h5', 'w')
Data_test.create_dataset('Xbb', data=yi_ParTXbb.reshape(-1,2))
Data_test.create_dataset('X_label', data=target_ParTXbb.reshape(-1,2),dtype='i4')
Data_test.close()        

      