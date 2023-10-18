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

sizes = [1730,  195762,  1959955,  2704,  29145,  293774,  2940006,  4665,  48752,  489801,  4900263,  9547,  97752,  979854]

sizes = np.sort(sizes)

def get_metrics(filelist_test,modeltype,size,Ntraining):
    yi=[]
    tagrgeti=[]
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
    return {'acc': acc,'auc': Auc,'y': y,'fpr': tpr,'tpr': tpr} 


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
            'bkg_rej_mean': np.mean(bkg_rej,axis=0),'bkg_rej_std': np.std(fpr,axis=0),
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

def merge_dict(existing_dict,append_dict):
    ds = [existing_dict,append_dict]
    d = {}
    for k in existing_dict.keys():
      d[k] = tuple(list(d[k] for d in ds))

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

ParTevent = []
ParTevent_scratch = []
mlpHlXbb = []
mlpLatent = []
mlpLatentHl = []

ParTevent_mean = []
ParTevent_scratch_mean = []
mlpHlXbb_mean = []
mlpLatent_mean = []
mlpLatentHl_mean = []

data, target, sig_type, weights = get_event_info(filelist_test)

for i in range(len(sizes)):

    experiments_ParTevent = []
    experiments_ParTevent_scratch = []
    experiments_mlpHlXbb = []
    experiments_mlpLatent = []
    experiments_mlpLatentHl = []

    for Ntraining in range(4):
        #experiments_ParTevent.append(get_metrics(filelist_test,'ParTevent',sizes[i],Ntraining+1))
        #experiments_ParTevent_scratch.append(get_metrics(filelist_test,'ParTevent_scratch',sizes[i],Ntraining+1))
        experiments_mlpHlXbb.append(get_metrics(filelist_test,'mlpHlXbb',sizes[i],Ntraining+1))
        experiments_mlpLatent.append(get_metrics(filelist_test,'mlpLatent',sizes[i],Ntraining+1))
        experiments_mlpLatentHl.append(get_metrics(filelist_test,'mlpLatentHl',sizes[i],Ntraining+1))   

    if i==0:
        print(i,experiments_mlpHlXbb)
        #ParTevent = experiments_ParTevent.copy() 
        #ParTevent_scratch = experiments_ParTevent_scratch.copy()    
        mlpHlXbb = experiments_mlpHlXbb.copy()    
        mlpLatent = experiments_mlpLatent.copy()   
        mlpLatentHl = experiments_mlpLatentHl.copy() 

        #ParTevent_mean = get_mean_metrics(experiments_ParTevent)
        #ParTevent_scratch_mean = get_mean_metrics(experiments_ParTevent_scratch)
        mlpHlXbb_mean = get_mean_metrics(experiments_mlpHlXbb)
        mlpLatent_mean = get_mean_metrics(experiments_mlpLatent)
        mlpLatentHl_mean = get_mean_metrics(experiments_mlpLatentHl) 
        print(i,mlpHlXbb_mean)

    else:
        print(i,experiments_mlpHlXbb)
        #ParTevent_mean = merge_dict(ParTevent_mean,get_mean_metrics(experiments_ParTevent))
        #ParTevent_scratch_mean = merge_dict(ParTevent_scratch_mean,get_mean_metrics(experiments_ParTevent_scratch))
        mlpHlXbb_mean = merge_dict(mlpHlXbb_mean,get_mean_metrics(experiments_mlpHlXbb))
        mlpLatent_mean = merge_dict(mlpLatent_mean,get_mean_metrics(experiments_mlpLatent))
        mlpLatentHl_mean = merge_dict(mlpLatentHl_mean,get_mean_metrics(experiments_mlpLatentHl))
        print(i,mlpHlXbb_mean)
        for j in range(4):
            #ParTevent[j] = merge_dict(ParTevent[j],experiments_ParTevent[j])
            #ParTevent_scratch[j] = merge_dict(ParTevent_scratch[j],experiments_ParTevent_scratch[j])
            mlpHlXbb[j] = merge_dict(mlpHlXbb[j],experiments_mlpHlXbb[j])
            mlpLatent[j] = merge_dict(mlpLatent[j],experiments_mlpLatent[j])
            mlpLatentHl[j] = merge_dict(mlpLatentHl[j],experiments_mlpLatentHl[j])   
   

out_out_dir = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/metrics/'
if (not os.path.exists(out_out_dir)): os.system(f'mkdir {out_out_dir}')
with h5py.File('/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/metrics/final_subsets.h5', 'w') as data:
        
        data.create_dataset('X_jet', data=data)
        data.create_dataset('targets', data=target)
        data.create_dataset('sig_type', data=sig_type)
        data.create_dataset('weights', data=weights)
        
        #model_group_ete = data.create_group('ete')
        #model_group_ete_scratch = data.create_group('ete_scratch')
        model_group_frozen = data.create_group('frozen')
        model_group_frozen_hl = data.create_group('frozen_hl')
        model_group_mlpHlXbb = data.create_group('mlpHlXbb')


        #subgroup = model_group_ete.create_group(f'mean')
        #subgroup.create_dataset('acc_mean', data=ParTevent_mean['acc_mean'])
        #subgroup.create_dataset('acc_std', data=ParTevent_mean['acc_std'])
        #subgroup.create_dataset('auc_mean', data=ParTevent_mean['auc_mean'])
        #subgroup.create_dataset('auc_std', data=ParTevent_mean['auc_std'])
        #subgroup.create_dataset('fpr_mean', data=ParTevent_mean['fpr_mean'])
        #subgroup.create_dataset('fpr_std', data=ParTevent_mean['fpr_std'])
        #subgroup.create_dataset('bkg_rej_mean', data=ParTevent_mean['bkg_rej_mean'])
        #subgroup.create_dataset('bkg_rej_std', data=ParTevent_mean['bkg_rej_std'])
        #subgroup.create_dataset('tpr', data=ParTevent_mean['tpr'])
#
        #subgroup = model_group_ete_scratch.create_group(f'mean')
        #subgroup.create_dataset('acc_mean', data=ParTevent_scratch_mean['acc_mean'])
        #subgroup.create_dataset('acc_std', data=ParTevent_scratch_mean['acc_std'])
        #subgroup.create_dataset('auc_mean', data=ParTevent_scratch_mean['auc_mean'])
        #subgroup.create_dataset('auc_std', data=ParTevent_scratch_mean['auc_std'])
        #subgroup.create_dataset('fpr_mean', data=ParTevent_scratch_mean['fpr_mean'])
        #subgroup.create_dataset('fpr_std', data=ParTevent_scratch_mean['fpr_std'])
        #subgroup.create_dataset('bkg_rej_mean', data=ParTevent_scratch_mean['bkg_rej_mean'])
        #subgroup.create_dataset('bkg_rej_std', data=ParTevent_scratch_mean['bkg_rej_std'])
        #subgroup.create_dataset('tpr', data=ParTevent_scratch_mean['tpr'])

        subgroup = model_group_frozen.create_group(f'mean')
        subgroup.create_dataset('acc_mean', data=mlpLatent_mean['acc_mean'])
        subgroup.create_dataset('acc_std', data=mlpLatent_mean['acc_std'])
        subgroup.create_dataset('auc_mean', data=mlpLatent_mean['auc_mean'])
        subgroup.create_dataset('auc_std', data=mlpLatent_mean['auc_std'])
        subgroup.create_dataset('fpr_mean', data=mlpLatent_mean['fpr_mean'])
        subgroup.create_dataset('fpr_std', data=mlpLatent_mean['fpr_std'])
        subgroup.create_dataset('bkg_rej_mean', data=mlpLatent_mean['bkg_rej_mean'])
        subgroup.create_dataset('bkg_rej_std', data=mlpLatent_mean['bkg_rej_std'])
        subgroup.create_dataset('tpr', data=mlpLatent_mean['tpr'])

        subgroup = model_group_frozen_hl.create_group(f'mean')
        subgroup.create_dataset('acc_mean', data=mlpLatentHl['acc_mean'])
        subgroup.create_dataset('acc_std', data=mlpLatentHl['acc_std'])
        subgroup.create_dataset('auc_mean', data=mlpLatentHl['auc_mean'])
        subgroup.create_dataset('auc_std', data=mlpLatentHl['auc_std'])
        subgroup.create_dataset('fpr_mean', data=mlpLatentHl['fpr_mean'])
        subgroup.create_dataset('fpr_std', data=mlpLatentHl['fpr_std'])
        subgroup.create_dataset('bkg_rej_mean', data=mlpLatentHl['bkg_rej_mean'])
        subgroup.create_dataset('bkg_rej_std', data=mlpLatentHl['bkg_rej_std'])
        subgroup.create_dataset('tpr', data=mlpLatentHl['tpr'])

        subgroup = model_group_mlpHlXbb.create_group(f'mean')
        subgroup.create_dataset('acc_mean', data=mlpHlXbb_mean['acc_mean'])
        subgroup.create_dataset('acc_std', data=mlpHlXbb_mean['acc_std'])
        subgroup.create_dataset('auc_mean', data=mlpHlXbb_mean['auc_mean'])
        subgroup.create_dataset('auc_std', data=mlpHlXbb_mean['auc_std'])
        subgroup.create_dataset('fpr_mean', data=mlpHlXbb_mean['fpr_mean'])
        subgroup.create_dataset('fpr_std', data=mlpHlXbb_mean['fpr_std'])
        subgroup.create_dataset('bkg_rej_mean', data=mlpHlXbb_mean['bkg_rej_mean'])
        subgroup.create_dataset('bkg_rej_std', data=mlpHlXbb_mean['bkg_rej_std'])
        subgroup.create_dataset('tpr', data=mlpHlXbb_mean['tpr'])
        
        for i in range(4):
            #subgroup = model_group_ete.create_group(f'training_{i+1}')
            #subgroup.create_dataset('acc', data=ParTevent[i]['acc'])
            #subgroup.create_dataset('auc', data=ParTevent[i]['auc'])
            #subgroup.create_dataset('y', data=ParTevent[i]['y'])
            #subgroup.create_dataset('fpr', data=ParTevent[i]['fpr'])
            #subgroup.create_dataset('tpr', data=ParTevent[i]['tpr'])
#
            #subgroup = model_group_ete_scratch.create_group(f'training_{i+1}')
            #subgroup.create_dataset('acc', data=ParTevent_scratch[i]['acc'])
            #subgroup.create_dataset('auc', data=ParTevent_scratch[i]['auc'])
            #subgroup.create_dataset('y', data=ParTevent_scratch[i]['y'])
            #subgroup.create_dataset('fpr', data=ParTevent_scratch[i]['fpr'])
            #subgroup.create_dataset('tpr', data=ParTevent_scratch[i]['tpr'])

            subgroup = model_group_frozen.create_group(f'training_{i+1}')
            subgroup.create_dataset('acc', data=mlpLatent[i]['acc'])
            subgroup.create_dataset('auc', data=mlpLatent[i]['auc'])
            subgroup.create_dataset('y', data=mlpLatent[i]['y'])
            subgroup.create_dataset('fpr', data=mlpLatent[i]['fpr'])
            subgroup.create_dataset('tpr', data=mlpLatent[i]['tpr'])

            subgroup = model_group_frozen_hl.create_group(f'training_{i+1}')
            subgroup.create_dataset('acc', data=mlpLatentHl[i]['acc'])
            subgroup.create_dataset('auc', data=mlpLatentHl[i]['auc'])
            subgroup.create_dataset('y', data=mlpLatentHl[i]['y'])
            subgroup.create_dataset('fpr', data=mlpLatentHl[i]['fpr'])
            subgroup.create_dataset('tpr', data=mlpLatentHl[i]['tpr'])

            subgroup = model_group_mlpHlXbb.create_group(f'training_{i+1}')
            subgroup.create_dataset('acc', data=mlpHlXbb[i]['acc'])
            subgroup.create_dataset('auc', data=mlpHlXbb[i]['auc'])
            subgroup.create_dataset('y', data=mlpHlXbb[i]['y'])
            subgroup.create_dataset('fpr', data=mlpHlXbb[i]['fpr'])
            subgroup.create_dataset('tpr', data=mlpHlXbb[i]['tpr'])


