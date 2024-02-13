from Finetune_hep.python import ParT,ParT_hlf,helpers,train,head
from Finetune_hep.python import ParT_hlf
from Finetune_hep.python import helpers,train
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import h5py
import yaml
import pickle

def full_model(config_file, **kwargs):

    with open(config_file) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader) 
    
    if data_config['inputs']['hlf']['concatenate']:
        model = ParT_hlf.get_model(data_config,**kwargs)  
    else:
        model = ParT.get_model(data_config,**kwargs)      
    
    return model

def head_model(config_file, **kwargs):

    with open(config_file) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader) 
    
    if data_config['inputs']['hlf']['concatenate']:
        model = ParT_hlf.get_model(data_config,**kwargs).head  
    else:
        model = ParT.get_model(data_config,**kwargs).head   
    
    return model

def save_rep(model,device,filelist,out_dir,repDim):

    with torch.no_grad():
        model.eval()
        with open(filelist) as f:
            i=-1
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                data_index = filename.index("Data")
                out_dir_i = out_dir + '/' + filename[data_index:]
                with h5py.File(filename, 'r') as Data:
                    if len(Data['X_label']) > 3000: size = 512
                    elif len(Data['X_label']) < 512: size = int(len(Data['X_label'])/2)
                    else:
                        size = len(Data['X_label'])/10
                        if len(Data['X_label']) == 0: 
                            print('no data')
                            continue
                    i+=1    
                    batches = np.array_split(np.arange(len(Data['X_label'])),int(len(Data['X_label'])/size))
                    for j in range(len(batches)):
                        data = {}
                        if model.Task == 'Xbb':
                            build_features = helpers.build_features_and_labels_Xbb
                            data['X_jet'] = Data['X_jet'][batches[j]].reshape(-1,len(helpers.jVars))
                            data['X_pfo'] = Data['X_pfo'][batches[j]].reshape(-1,100, len(helpers.pVars))
                            data['labels'] = Data['X_label'][batches[j]].reshape(-1,len(helpers.labelVars))
                            data = build_features(data)
                        else:   
                            build_features = helpers.build_features_and_labels
                            data['X_jet'] = Data['X_jet'][batches[j]]
                            data['X_pfo'] = Data['X_pfo'][batches[j]]
                            data['labels'] = Data['labels'][batches[j]]
                            data['jet_mask'] = Data['jet_mask'][batches[j]]
                            data = build_features(data) 
                        if (j==0):
                            preds = train.infer_val(model,data,device).detach().cpu().numpy()
                            target = data['label']
                        else:
                            preds = np.concatenate((preds,train.infer_val(model,data,device).detach().cpu().numpy()),axis=0)
                            target = np.concatenate((target,data['label']),axis=0)
                    if model.Task == 'Xbb':
                        Data = h5py.File(out_dir_i, 'w')
                        Data.create_dataset('Xbb_score', data=preds.reshape(-1,5,repDim))
                        Data.create_dataset('Xbb_label', data=target.reshape(-1,5),dtype='i4')
                        Data.close()   
                    else:
                        Data = h5py.File(out_dir_i, 'w')
                        Data.create_dataset('evt_score', data=preds.reshape(-1,repDim))
                        Data.create_dataset('evt_label', data=target.reshape(-1),dtype='i4')
                        Data.close()      
    return 0

def save_rep_head(model,device,filelist,out_dir,repDim,Xbb_scores_path,use_hlf,scaler_path):

    with torch.no_grad():
        model.to(device)
        model.eval()
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                with h5py.File(filename, 'r') as Data:
                    subset_offset = int(len(Data['X_jet']))
                    if i ==0:
                        hlf = Data['X_jet'][:subset_offset]
                        target = Data['labels'][:subset_offset] 
                        jet_mask = Data['jet_mask'][:subset_offset]
                    else:
                        hlf = np.concatenate((hlf,Data['X_jet'][:subset_offset]),axis=0)
                        target = np.concatenate((target,Data['labels'][:subset_offset]),axis=0)
                        jet_mask = np.concatenate((jet_mask,Data['jet_mask'][:subset_offset]),axis=0)
                    i+=1    
        scaler = StandardScaler()
        hlf[:,:,helpers.jVars.index('fj_pt')] = helpers.log(hlf[:,:,helpers.jVars.index('fj_pt')])
        hlf[:,:,helpers.jVars.index('fj_mass')] = helpers.log(hlf[:,:,helpers.jVars.index('fj_mass')])
        hlf[:,:,helpers.jVars.index('fj_sdmass')] = helpers.log(hlf[:,:,helpers.jVars.index('fj_sdmass')])
        if Xbb_scores_path != '': 
            subset_offset=0
            i=0
            with open(Xbb_scores_path) as f:
                for line in f:
                    filename = line.strip()
                    print('loading Xbb scores from : ',filename)
                    with h5py.File(filename, 'r') as Xbb_scores:
                        subset_offset = int(len(Xbb_scores['Xbb_score']))
                        if i ==0:
                            Xbb = Xbb_scores['Xbb_score'][:subset_offset]
                        else:
                            Xbb = np.concatenate((Xbb,Xbb_scores['Xbb_score'][:subset_offset]),axis=0)
                        i+=1    
            if repDim != 1: hlf[:,:,helpers.jVars.index('fj_doubleb')] = np.nan_to_num(Xbb.reshape(-1,5))  
                    
        if repDim != 1: Xbb = hlf[:,:,helpers.jVars.index('fj_doubleb')].reshape(-1,5,1)
        
        hlfeats = [helpers.jVars.index('fj_pt'),helpers.jVars.index('fj_eta'),helpers.jVars.index('fj_phi'),helpers.jVars.index('fj_mass'),helpers.jVars.index('fj_sdmass')]
        if use_hlf: data = np.concatenate((Xbb,hlf[:,:,hlfeats]),axis=-1)      
        else: data = Xbb

        if scaler_path !='' : 
            with open(scaler_path,'rb') as f:
                scaler = pickle.load(f)
                X_norm = head.transform_without_zeros(data,jet_mask,scaler)
                x = torch.from_numpy(X_norm).float().to(device)
        else:
            x = torch.from_numpy(data).float().to(device)    
        y = torch.from_numpy(target.reshape(-1,1)).float().to(device)
        jet_mask = torch.from_numpy(jet_mask.reshape(-1,5,1)).float().to(device) 
        preds = model(x) 

        Data = h5py.File(out_dir+'/scores.h5', 'w')
        Data.create_dataset('evt_score', data=preds.reshape(-1,repDim))
        Data.create_dataset('evt_label', data=y.reshape(-1),dtype='i4')
        Data.close()   
        
