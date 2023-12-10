import h5py
import matplotlib.pyplot as plt
import numpy as np
from openTSNE import TSNE


labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]       
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
pVars = [f'pfcand_{v}' for v in ['ptrel','erel','etarel','phirel','dxy','dxysig','dz','dzsig','deltaR','charge','isChargedHad','isNeutralHad','isGamma','isEl','isMu']]        


filelist = '/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt'
Xbb_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/Latent_ParTXbb_scratch//ParTXbb_score_training_1'
Xbb_scratch_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/Latent_ParTXbb_scratch_etoe/ParTXbb_score_training_1'
Xbb_finetuned_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/Latent_ParTXbb_finetuned/ParTXbb_score_training_1'
Xbb_double_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/Latent_ParTXbb_double/ParTXbb_score_training_1'
subset_batches = 0.1

subset_offset=0
i=0
with open(filelist) as f:
    for line in f:
        filename = line.strip()
        print('reading : ',filename)
        with h5py.File(filename, 'r') as Data:
            subset_offset = int(len(Data['X_jet'])*subset_batches)
            if i ==0:
                #data = Data['X_jet'][:subset_offset]
                target_evt = Data['labels'][:subset_offset] 
                jet_mask = Data['jet_mask'][:subset_offset]
                #print(Data['X_label'].shape)
                target = Data['X_label'][:subset_offset,:,labelVars.index('label_H_bb')] 
            else:
                #data = np.concatenate((data,Data['X_jet'][:subset_offset]),axis=0)
                target_evt = np.concatenate((target_evt,Data['labels'][:subset_offset]),axis=0)
                jet_mask = np.concatenate((jet_mask,Data['jet_mask'][:subset_offset]),axis=0)
                target = np.concatenate((target,Data['X_label'][:subset_offset,:,labelVars.index('label_H_bb')] ),axis=0)
            i+=1    


target = target.reshape(-1)[(jet_mask==1).reshape(-1)]
target_evt = target_evt.reshape(-1)

subset_offset=0
i=0
with open(filelist) as f:
    for line in f:
        filename = line.strip()
        data_index = filename.index("Data")
        sample_name = filename[data_index:]
        
        name = f'{Xbb_scores_path}/{sample_name}'
        print('loading Xbb scores from : ',name)
        with h5py.File(name, 'r') as Xbb_scores:
            subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
            if i ==0:
                Xbb = Xbb_scores['evt_score'][:subset_offset]
            else:
                Xbb = np.concatenate((Xbb,Xbb_scores['evt_score'][:subset_offset]),axis=0)

        name = f'{Xbb_scratch_scores_path}/{sample_name}'
        print('loading scratch scalar from : ',name)
        with h5py.File(name, 'r') as Xbb_scores:
            subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
            if i ==0:
                Xbb_scratch = Xbb_scores['evt_score'][:subset_offset]
            else:
                Xbb_scratch = np.concatenate((Xbb_scratch,Xbb_scores['evt_score'][:subset_offset]),axis=0)

        print('loading finetuned scalar from : ',name)
        name = f'{Xbb_finetuned_scores_path}/{sample_name}'
        with h5py.File(name, 'r') as Xbb_scores:
            subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
            if i ==0:
                Xbb_finetuned = Xbb_scores['evt_score'][:subset_offset]
            else:
                Xbb_finetuned = np.concatenate((Xbb_finetuned,Xbb_scores['evt_score'][:subset_offset]),axis=0)

        name = f'{Xbb_double_scores_path}/{sample_name}'
        print('loading double finetuned scalar from : ',name)
        with h5py.File(name, 'r') as Xbb_scores:
            subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
            if i ==0:
                Xbb_double = Xbb_scores['evt_score'][:subset_offset]
            else:
                Xbb_double = np.concatenate((Xbb_double,Xbb_scores['evt_score'][:subset_offset]),axis=0)
            
        i+=1            

Xbb = (np.nan_to_num(Xbb)[jet_mask==1]).reshape(-1,128)
Xbb_scratch = (np.nan_to_num(Xbb_scratch)[jet_mask==1]).reshape(-1,128)
Xbb_finetuned = (np.nan_to_num(Xbb_finetuned)[jet_mask==1]).reshape(-1,128)
Xbb_double = (np.nan_to_num(Xbb_double)[jet_mask==1]).reshape(-1,128)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def plot_grid(vecs,target,label,title):
    features = vecs[target == label]
    mean_features = np.mean(features, axis=0)
    mean_features_2d = mean_features.reshape(16, 8)
    sign = np.sign(np.mean(vecs[target == 1], axis=0) - np.mean(vecs[target == 0], axis=0))
    print(sign.shape)
    auc_values = [roc_auc_score(target.reshape(-1,1), sign[i]*vecs[:, i]) for i in range(128)]
    auc_values_2d = np.array(auc_values).reshape(16, 8)

    plt.figure(figsize=(8, 8))
    heatmap = plt.imshow(auc_values_2d, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0, vmax=1)

    #for i in range(16):
    #    for j in range(8):
    #        auc = auc_values_2d[i, j]
    #        plt.text(j, i, f'{auc:.3f}', color='white', ha='center', va='center', fontsize=8)

    plt.colorbar(label='Mean Value')
    plt.title(title)
    plt.xlabel('Dimension')
    plt.ylabel('Dimension')
    plt.savefig(f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/plots/{title}.png')
    plt.show()

plot_grid(Xbb,target,1,title='Frozen vector')
plot_grid(Xbb_scratch,target,1,title='From-scratch vector')
plot_grid(Xbb_finetuned,target,1,title='Finetuned vector')
plot_grid(Xbb_double,target,1,title='Double-Finetuned vector')



