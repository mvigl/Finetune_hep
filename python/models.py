from Finetune_hep.python import ParT
from Finetune_hep.python import ParT_hlf
from Finetune_hep.python import helpers,train
import torch
import numpy as np
import h5py
import yaml

def full_model(config_file, **kwargs):

    with open(config_file) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader) 
    
    if data_config['inputs']['hlf']['concatenate']:
        model = ParT_hlf.get_model(data_config,**kwargs)  
    else:
        model = ParT.get_model(data_config,**kwargs)      
    
    return model

def head(config_file, **kwargs):

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
                            data['pf_mask'][:,:,:,:2] += np.abs(data['jet_mask'][:,:,np.newaxis]-1)
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