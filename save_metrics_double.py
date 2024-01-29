from Finetune_hep.python import ParT_mlp
from Finetune_hep.python import Mlp
from Finetune_hep.python import definitions as df
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
import h5py

filelist_test = '/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt'

cs =[7823.28,  648.174, 186.946, 32.2928]
sizes = [1730,  19332,  195762,  1959955,  2704,  29145,  293774,  2940006, 4665,  48752,  489801,  4900263,  777,  9547,  97752,  979854]#, 9800758]
sizes = np.sort(sizes)

def get_metrics(filelist_test,modeltype,size,Ntraining):
    yi=[]
    tagrgeti=[]
    tpr_common = np.linspace(0,1,10000)
    with open(filelist_test) as f:
        for line in f:
            filename = line.strip()
            data_index = filename.index("Data")
            sample_name = filename[data_index:]
            name = f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/final_{modeltype}_{Ntraining}/{size}/{sample_name}'
            with h5py.File(name, 'r') as Data:
                yi.append(Data['evt_score'][:].reshape(-1))
                tagrgeti.append(Data['evt_label'][:].reshape(-1)) 
    
    target = np.concatenate(tagrgeti).reshape(-1)
    y = np.concatenate(yi).reshape(-1)
    fpr, tpr, thresholds = roc_curve(target,y)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    acc = balanced_accuracy_score(target,(y>= optimal_threshold).astype(int))
    Auc = auc(fpr,tpr)
    return {'acc': acc,'auc': Auc,'y': y,'fpr': np.interp(tpr_common, tpr,fpr),'tpr': tpr_common} 


def get_mean_metrics(experiments):
    tpr_common = np.linspace(0,1,10000)
    Auc=[]
    acc=[]
    fpr=[]
    bkg_rej=[]
    for i in range(len(experiments)):
        Auc.append(experiments[i]['auc'])
        acc.append(experiments[i]['acc'])
        fpr.append(np.interp(tpr_common, experiments[i]['tpr'], experiments[i]['fpr']))
        bkg_rej.append(1/fpr[i])

    Auc = np.array(Auc).reshape(-1)
    acc = np.array(acc).reshape(-1)
    fpr = np.array(fpr)
    bkg_rej = np.array(bkg_rej)

    return {'acc_mean': np.mean(acc,axis=0),'acc_std': np.std(acc,axis=0),
            'auc_mean': np.mean(Auc,axis=0),'auc_std': np.std(Auc,axis=0),
            'fpr_mean': np.mean(fpr,axis=0),'fpr_std': np.std(fpr,axis=0),
            'bkg_rej_mean': np.mean(bkg_rej,axis=0),'bkg_rej_std': np.std(bkg_rej,axis=0),
            'tpr': tpr_common} 

def get_event_info(filelist):
    i=0
    with open(filelist) as f:
        for line in f:
            filename = line.strip()
            print('reading : ',filename)
            data_index = filename.index("Data")
            with h5py.File(filename, 'r') as Data:
                if i ==0:
                    data = Data['X_jet'][:]
                    target = Data['labels'][:] 
                    jet_mask = Data['jet_mask'][:]
                    sig_type,weights = get_sig_type_and_w(filename[data_index:],len(Data['X_jet'][:]))  
                else:
                    data = np.concatenate((data,Data['X_jet'][:]),axis=0)
                    target = np.concatenate((target,Data['labels'][:]),axis=0)
                    jet_mask = np.concatenate((jet_mask,Data['jet_mask'][:]),axis=0)
                    sig_type = np.concatenate((sig_type,get_sig_type_and_w(filename[data_index:],len(Data['X_jet'][:]))[0]),axis=0)
                    weights = np.concatenate((weights,get_sig_type_and_w(filename[data_index:],len(Data['X_jet'][:]))[1]),axis=0)
                i+=1         
    return data, target, sig_type, weights

def merge_dict(existing_dict):
    ds = [existing_dict[0],
          existing_dict[1],
          existing_dict[2],
          existing_dict[3],
          existing_dict[4],
          existing_dict[5],
          existing_dict[6],
          existing_dict[7],
          existing_dict[8],
          existing_dict[9],
          existing_dict[10],
          existing_dict[11],
          existing_dict[12],
          existing_dict[13],
          existing_dict[14],
          existing_dict[15],
          #existing_dict[16]
          ]
    d = {}
    for k in existing_dict[0].keys():
      d[k] = tuple(list(d[k] for d in ds))
    return d

import re

def extract_number_and_type(input_string):
    number_pattern = r'\d+'
    number_matches = re.findall(number_pattern, input_string)
    has_sig = 'sig' in input_string
    number = int(number_matches[0]) if number_matches else 0
    return number, has_sig


def get_sig_type_and_w(filename,length):
    samlpe,isSig = extract_number_and_type(filename)
    sig_type = np.zeros(length)
    weights = np.ones(length)
    if isSig:
        if samlpe < 18 : 
            sig_type=np.ones(length)*600
        elif ((samlpe>17) and (samlpe < 43)) : 
            sig_type=np.ones(length)*1000
        elif ((samlpe>42) and (samlpe < 45)) : 
            sig_type = np.ones(length)*1200  
        elif (samlpe==45) : 
            sig_type = np.ones(length)*1400  
        elif ((samlpe>45) and (samlpe < 48)) : 
            sig_type = np.ones(length)*1600  
        elif ((samlpe>47) and (samlpe < 51)) : 
            sig_type = np.ones(length)*1800 
        elif ((samlpe>50) and (samlpe < 92)) : 
            sig_type = np.ones(length)*2000 
        elif ((samlpe>91) and (samlpe < 94)) : 
            sig_type = np.ones(length)*2500   
        elif ((samlpe>93) and (samlpe < 125)) : 
            sig_type = np.ones(length)*3000  
        elif ((samlpe>124) and (samlpe < 127)) : 
            sig_type = np.ones(length)*4000       
        else:
            sig_type = np.ones(length)*4500    

                     
    else:
        if samlpe < 70 : 
            weights = np.ones(length)*cs[0]
        elif ((samlpe>69) and (samlpe < 125)) : 
            weights = np.ones(length)*cs[1]
        elif ((samlpe>124) and (samlpe < 201)) : 
            weights = np.ones(length)*cs[2]
        elif ((samlpe>200) and (samlpe < 261)) : 
            weights = np.ones(length)*cs[3]    
         
    return sig_type,weights

ParTevent_Xbb_Hl_double = [[],[],[],[]]
ParTevent_Hl_double = [[],[],[],[]]
ParTevent_double = [[],[],[],[]]

ParTevent_Xbb_Hl_double_mean = []
ParTevent_Hl_double_mean = []
ParTevent_double_mean = []

feats, target, sig_type, weights = get_event_info(filelist_test)

for i in range(len(sizes)):
    print('size : ', sizes[i])
    experiments_ParTevent_Xbb_Hl_double = []
    experiments_ParTevent_Hl_double = []
    experiments_ParTevent_double = []
    

    for Ntraining in range(4):
        experiments_ParTevent_Xbb_Hl_double.append(get_metrics(filelist_test,'ParTevent_Xbb_Hl_double',sizes[i],Ntraining+1))
        experiments_ParTevent_Hl_double.append(get_metrics(filelist_test,'ParTevent_Hl_double',sizes[i],Ntraining+1))
        experiments_ParTevent_double.append(get_metrics(filelist_test,'ParTevent_double',sizes[i],Ntraining+1))


    #print(i,experiments_ParTevent)
    ParTevent_Xbb_Hl_double_mean.append(get_mean_metrics(experiments_ParTevent_Xbb_Hl_double))
    ParTevent_Hl_double_mean.append(get_mean_metrics(experiments_ParTevent_Hl_double))
    ParTevent_double_mean.append(get_mean_metrics(experiments_ParTevent_double))

    for j in range(4):
        ParTevent_Xbb_Hl_double[j].append(experiments_ParTevent_Xbb_Hl_double[j])
        ParTevent_Hl_double[j].append(experiments_ParTevent_Hl_double[j])
        ParTevent_double[j].append(experiments_ParTevent_double[j])
   
ParTevent_Xbb_Hl_double_mean = merge_dict(ParTevent_Xbb_Hl_double_mean)
ParTevent_Hl_double_mean = merge_dict(ParTevent_Hl_double_mean)
ParTevent_double_mean = merge_dict(ParTevent_double_mean)

for j in range(4):
        ParTevent_Xbb_Hl_double[j] = merge_dict(ParTevent_Xbb_Hl_double[j])
        ParTevent_Hl_double[j] = merge_dict(ParTevent_Hl_double[j])
        ParTevent_double[j] = merge_dict(ParTevent_double[j])

out_out_dir = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/metrics/'
if (not os.path.exists(out_out_dir)): os.system(f'mkdir {out_out_dir}')
with h5py.File('/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/metrics/final_subsets_double.h5', 'w') as data:
        
        data.create_dataset('X_jet', data=feats)
        data.create_dataset('targets', data=target)
        data.create_dataset('sig_type', data=sig_type)
        data.create_dataset('weights', data=weights)
        
        model_group_ete_Xbb_Hl_double = data.create_group('Xbb_Hl_finetuned_double')
        model_group_ete_Hl_double = data.create_group('Latent_Hl_finetuned_double')
        model_group_ete_double = data.create_group('Latent_finetuned_double')


        subgroup = model_group_ete_Xbb_Hl_double.create_group(f'mean')
        subgroup.create_dataset('acc_mean', data=ParTevent_Xbb_Hl_double_mean['acc_mean'])
        subgroup.create_dataset('acc_std', data=ParTevent_Xbb_Hl_double_mean['acc_std'])
        subgroup.create_dataset('auc_mean', data=ParTevent_Xbb_Hl_double_mean['auc_mean'])
        subgroup.create_dataset('auc_std', data=ParTevent_Xbb_Hl_double_mean['auc_std'])
        subgroup.create_dataset('fpr_mean', data=ParTevent_Xbb_Hl_double_mean['fpr_mean'])
        subgroup.create_dataset('fpr_std', data=ParTevent_Xbb_Hl_double_mean['fpr_std'])
        subgroup.create_dataset('bkg_rej_mean', data=ParTevent_Xbb_Hl_double_mean['bkg_rej_mean'])
        subgroup.create_dataset('bkg_rej_std', data=ParTevent_Xbb_Hl_double_mean['bkg_rej_std'])
        subgroup.create_dataset('tpr', data=ParTevent_Xbb_Hl_double_mean['tpr'])

        subgroup = model_group_ete_Hl_double.create_group(f'mean')
        subgroup.create_dataset('acc_mean', data=ParTevent_Hl_double_mean['acc_mean'])
        subgroup.create_dataset('acc_std', data=ParTevent_Hl_double_mean['acc_std'])
        subgroup.create_dataset('auc_mean', data=ParTevent_Hl_double_mean['auc_mean'])
        subgroup.create_dataset('auc_std', data=ParTevent_Hl_double_mean['auc_std'])
        subgroup.create_dataset('fpr_mean', data=ParTevent_Hl_double_mean['fpr_mean'])
        subgroup.create_dataset('fpr_std', data=ParTevent_Hl_double_mean['fpr_std'])
        subgroup.create_dataset('bkg_rej_mean', data=ParTevent_Hl_double_mean['bkg_rej_mean'])
        subgroup.create_dataset('bkg_rej_std', data=ParTevent_Hl_double_mean['bkg_rej_std'])
        subgroup.create_dataset('tpr', data=ParTevent_Hl_double_mean['tpr'])

        subgroup = model_group_ete_double.create_group(f'mean')
        subgroup.create_dataset('acc_mean', data=ParTevent_double_mean['acc_mean'])
        subgroup.create_dataset('acc_std', data=ParTevent_double_mean['acc_std'])
        subgroup.create_dataset('auc_mean', data=ParTevent_double_mean['auc_mean'])
        subgroup.create_dataset('auc_std', data=ParTevent_double_mean['auc_std'])
        subgroup.create_dataset('fpr_mean', data=ParTevent_double_mean['fpr_mean'])
        subgroup.create_dataset('fpr_std', data=ParTevent_double_mean['fpr_std'])
        subgroup.create_dataset('bkg_rej_mean', data=ParTevent_double_mean['bkg_rej_mean'])
        subgroup.create_dataset('bkg_rej_std', data=ParTevent_double_mean['bkg_rej_std'])
        subgroup.create_dataset('tpr', data=ParTevent_double_mean['tpr'])


        
        for i in range(4):
            subgroup = model_group_ete_Xbb_Hl_double.create_group(f'training_{i+1}')
            subgroup.create_dataset('acc', data=ParTevent_Xbb_Hl_double[i]['acc'])
            subgroup.create_dataset('auc', data=ParTevent_Xbb_Hl_double[i]['auc'])
            subgroup.create_dataset('y', data=ParTevent_Xbb_Hl_double[i]['y'])
            subgroup.create_dataset('fpr', data=ParTevent_Xbb_Hl_double[i]['fpr'])
            subgroup.create_dataset('tpr', data=ParTevent_Xbb_Hl_double[i]['tpr'])

            subgroup = model_group_ete_Hl_double.create_group(f'training_{i+1}')
            subgroup.create_dataset('acc', data=ParTevent_Hl_double[i]['acc'])
            subgroup.create_dataset('auc', data=ParTevent_Hl_double[i]['auc'])
            subgroup.create_dataset('y', data=ParTevent_Hl_double[i]['y'])
            subgroup.create_dataset('fpr', data=ParTevent_Hl_double[i]['fpr'])
            subgroup.create_dataset('tpr', data=ParTevent_Hl_double[i]['tpr'])

            subgroup = model_group_ete_double.create_group(f'training_{i+1}')
            subgroup.create_dataset('acc', data=ParTevent_double[i]['acc'])
            subgroup.create_dataset('auc', data=ParTevent_double[i]['auc'])
            subgroup.create_dataset('y', data=ParTevent_double[i]['y'])
            subgroup.create_dataset('fpr', data=ParTevent_double[i]['fpr'])
            subgroup.create_dataset('tpr', data=ParTevent_double[i]['tpr'])


