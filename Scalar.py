import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]       
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
pVars = [f'pfcand_{v}' for v in ['ptrel','erel','etarel','phirel','dxy','dxysig','dz','dzsig','deltaR','charge','isChargedHad','isNeutralHad','isGamma','isEl','isMu']]        


filelist = '/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt'
Xbb_scratch_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTevent_Xbb_Hl_scratch'
Xbb_finetuned_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTevent_Xbb_Hl'
Xbb_double_scores_path = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTevent_Xbb_Hl_double'
subset_batches = 1

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

auc_scratch = []
auc_scratch_rev = []
auc_finetuned = []
auc_double = []
sizes = [1730,  19332,  195762,  1959955,  2704,  29145,  293774,  2940006, 4665,  48752,  489801,  4900263,  777,  9547,  97752,  979854, 9800758]
sizes = np.sort(sizes)

sizes_low = [41, 89, 138, 242]
sizes_low = np.sort(sizes_low)

for size in sizes_low:
        subset_offset=0
        i=0
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                data_index = filename.index("Data")
                sample_name = filename[data_index:]

                name = f'{Xbb_finetuned_scores_path}/{size}/{sample_name}'
                print('loading finetuned scalar from : ',name)
                with h5py.File(name, 'r') as Xbb_scores:
                    subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
                    if i ==0:
                        Xbb_finetuned = Xbb_scores['evt_score'][:subset_offset]
                    else:
                        Xbb_finetuned = np.concatenate((Xbb_finetuned,Xbb_scores['evt_score'][:subset_offset]),axis=0)

                name = f'{Xbb_double_scores_path}/{size}/{sample_name}'
                if name == '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTevent_Xbb_Hl/41/Data_test_sig_73.h5':
                    name = '/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTevent_Xbb_Hl_double/41/Data_test_sig_73.h5'
                print('loading double finetuned scalar from : ',name)
                with h5py.File(name, 'r') as Xbb_scores:
                    subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
                    if i ==0:
                        Xbb_double = Xbb_scores['evt_score'][:subset_offset]
                    else:
                        Xbb_double = np.concatenate((Xbb_double,Xbb_scores['evt_score'][:subset_offset]),axis=0)

                i+=1            

        Xbb_finetuned = (np.nan_to_num(Xbb_finetuned)[jet_mask==1]).reshape(-1)
        auc_finetuned.append(roc_auc_score(target.reshape(-1,1),Xbb_finetuned))
        Xbb_double = (np.nan_to_num(Xbb_double)[jet_mask==1]).reshape(-1)
        auc_double.append(roc_auc_score(target.reshape(-1,1),Xbb_double))

        auc_scratch.append(0.)
        auc_scratch_rev.append(0.)

for size in sizes:
        subset_offset=0
        i=0
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                data_index = filename.index("Data")
                sample_name = filename[data_index:]

                name = f'{Xbb_scratch_scores_path}/{size}/{sample_name}'
                print('loading scratch scalar from : ',name)
                with h5py.File(name, 'r') as Xbb_scores:
                    subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
                    if i ==0:
                        Xbb_scratch = Xbb_scores['evt_score'][:subset_offset]
                    else:
                        Xbb_scratch = np.concatenate((Xbb_scratch,Xbb_scores['evt_score'][:subset_offset]),axis=0)

                name = f'{Xbb_finetuned_scores_path}/{size}/{sample_name}'
                print('loading finetuned scalar from : ',name)
                with h5py.File(name, 'r') as Xbb_scores:
                    subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
                    if i ==0:
                        Xbb_finetuned = Xbb_scores['evt_score'][:subset_offset]
                    else:
                        Xbb_finetuned = np.concatenate((Xbb_finetuned,Xbb_scores['evt_score'][:subset_offset]),axis=0)

                name = f'{Xbb_double_scores_path}/{size}/{sample_name}'
                print('loading double finetuned scalar from : ',name)
                with h5py.File(name, 'r') as Xbb_scores:
                    subset_offset = int(len(Xbb_scores['evt_score'])*subset_batches)
                    if i ==0:
                        Xbb_double = Xbb_scores['evt_score'][:subset_offset]
                    else:
                        Xbb_double = np.concatenate((Xbb_double,Xbb_scores['evt_score'][:subset_offset]),axis=0)

                i+=1            

        Xbb_scratch = (np.nan_to_num(Xbb_scratch)[jet_mask==1]).reshape(-1)
        auc_scratch.append(roc_auc_score(target.reshape(-1,1),Xbb_scratch))
        auc_scratch_rev.append(roc_auc_score(target.reshape(-1,1),1-Xbb_scratch))
        Xbb_finetuned = (np.nan_to_num(Xbb_finetuned)[jet_mask==1]).reshape(-1)
        auc_finetuned.append(roc_auc_score(target.reshape(-1,1),Xbb_finetuned))
        Xbb_double = (np.nan_to_num(Xbb_double)[jet_mask==1]).reshape(-1)
        auc_double.append(roc_auc_score(target.reshape(-1,1),Xbb_double))

        if size == 9800758:
            fpr_scratch, tpr_scratch, thresholds_scratch = roc_curve(target,Xbb_scratch)
            fpr_scratch_rev, tpr_scratch_rev, thresholds_scratch_rev = roc_curve(target,1-Xbb_scratch)
            fpr_finetuned, tpr_finetuned, thresholds_finetuned = roc_curve(target,Xbb_finetuned)
            fpr_double, tpr_double, thresholds_double = roc_curve(target,Xbb_double)


with h5py.File('/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/metrics/scalars_auc.h5', 'w') as data:
        
        data.create_dataset('auc_scratch', data=auc_scratch)
        data.create_dataset('tpr_scratch', data=tpr_scratch)
        data.create_dataset('fpr_scratch', data=fpr_scratch)
        data.create_dataset('auc_scratch_rev', data=auc_scratch_rev)
        data.create_dataset('tpr_scratch_rev', data=tpr_scratch_rev)
        data.create_dataset('fpr_scratch_rev', data=fpr_scratch_rev)
        data.create_dataset('auc_finetuned', data=auc_finetuned)
        data.create_dataset('tpr_finetuned', data=tpr_finetuned)
        data.create_dataset('fpr_finetuned', data=fpr_finetuned)
        data.create_dataset('auc_double', data=auc_double)
        data.create_dataset('tpr_double', data=tpr_double)
        data.create_dataset('fpr_double', data=fpr_double)
        


