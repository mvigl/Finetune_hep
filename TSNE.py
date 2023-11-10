import h5py
import matplotlib.pyplot as plt
import numpy as np
from openTSNE import TSNE


labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]       
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
pVars = [f'pfcand_{v}' for v in ['ptrel','erel','etarel','phirel','dxy','dxysig','dz','dzsig','deltaR','charge','isChargedHad','isNeutralHad','isGamma','isEl','isMu']]        


filelist = '/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt'
Xbb_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/config/Latent_ParTXbb_scratch/Xbb_scratch_test_list_1.txt'
Xbb_finetuned_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/config/Latent_ParTXbb_finetuned/Xbb_scratch_test_list_1.txt'
subset_batches = 0.0001

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
                #target = Data['labels'][:subset_offset] 
                jet_mask = Data['jet_mask'][:subset_offset]
                target = Data['X_label'][:subset_offset,:,labelVars.index('label_H_bb')] 
            else:
                #data = np.concatenate((data,Data['X_jet'][:subset_offset]),axis=0)
                #target = np.concatenate((target,Data['labels'][:subset_offset]),axis=0)
                jet_mask = np.concatenate((jet_mask,Data['jet_mask'][:subset_offset]),axis=0)
                target = np.concatenate((target,Data['X_label'][:subset_offset,:,labelVars.index('label_H_bb')] ),axis=0)
            i+=1    

target = target.reshape(-1)
subset_offset=0
i=0
with open(Xbb_scores_path) as f:
    for line in f:
        filename = line.strip()
        print('loading Xbb scores from : ',filename)
        with h5py.File(filename, 'r') as Xbb_scores:
            subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
            if i ==0:
                Xbb = Xbb_scores['evt_score'][:subset_offset]
            else:
                Xbb = np.concatenate((Xbb,Xbb_scores['evt_score'][:subset_offset]),axis=0)
            i+=1    
Xbb = (np.nan_to_num(Xbb)[jet_mask==1]).reshape(-1,128)

subset_offset=0
i=0
with open(Xbb_finetuned_scores_path) as f:
    for line in f:
        filename = line.strip()
        print('loading Xbb scores from : ',filename)
        with h5py.File(filename, 'r') as Xbb_scores:
            subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
            if i ==0:
                Xbb_finetuned = Xbb_scores['evt_score'][:subset_offset]
            else:
                Xbb_finetuned = np.concatenate((Xbb_finetuned,Xbb_scores['evt_score'][:subset_offset]),axis=0)
            i+=1    
Xbb_finetuned = (np.nan_to_num(Xbb_finetuned)[jet_mask==1]).reshape(-1,128)


tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)

embed = tsne.fit(Xbb)
embed_finetuned = tsne.fit(Xbb_finetuned)

fig = plt.figure()
plt.scatter(embed[:,0],embed[:,1],c = target[:len(embed)], cmap = 'viridis', alpha = 0.2)
plt.colorbar()
fig.savefig(f"../../Finetune_hep/plots/TSNE.pdf")

fig = plt.figure()
plt.scatter(embed_finetuned[:,0],embed_finetuned[:,1],c = target[:len(embed_finetuned)], cmap = 'viridis', alpha = 0.2)
plt.colorbar()
fig.savefig(f"../../Finetune_hep/plots/TSNE_finetuned.pdf")