import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import cm
import numpy as np
import h5py

filelist = '/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt'
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]      
def get_vars(filelist):
    i=0
    with open(filelist) as f:
        for line in f:
            filename = line.strip()
            print('reading : ',filename)
            with h5py.File(filename, 'r') as Data:
                if i ==0:
                    data = Data['X_jet'][:]
                    target = Data['labels'][:]
                    target_jet = Data['X_label'][:]
                    jet_mask = Data['jet_mask'][:]
                else:
                    data = np.concatenate((data,Data['X_jet'][:]),axis=0)
                    target = np.concatenate((target,Data['labels'][:]),axis=0) 
                    target_jet = np.concatenate((target_jet,Data['X_label'][:]),axis=0) 
                    jet_mask = np.concatenate((jet_mask,Data['jet_mask'][:]),axis=0) 
            i+=1
    return data,target,target_jet,jet_mask

data,target,target_jet,jet_mask = get_vars(filelist)

fpr_ParTXbb=[]
auc_ParTXbb=[]
fpr_CMSXbb=[]
auc_CMSXbb=[]
jet_mask = (jet_mask.reshape(-1)).astype(bool)

for i in range(5):
    with h5py.File(f'/raven/u/mvigl/Finetune_hep_dir/Finetune_hep/models/ParTXbb/test_ParTXbb_score_training_{i+1}.h5', 'r') as test_data:
        yi_ParTXbb,target_ParTXbb = test_data['Xbb'][:].reshape(-1)[jet_mask],test_data['X_label'][:].reshape(-1)[jet_mask]
    fpr_i, tpr_i, threshold_i = roc_curve(target_ParTXbb.reshape(-1,1), yi_ParTXbb.reshape(-1,1),drop_intermediate=False)

    if i==0: 
        tpr_common = tpr_i
    fpr_ParTXbb.append(np.interp(tpr_common, tpr_i, fpr_i))
    auc_ParTXbb.append(auc(fpr_i,tpr_i))

yi_doubleb = (data[:,:,jVars.index('fj_doubleb')].reshape(-1)[jet_mask]+1)/2
target_doubleb = target_jet[:,:,labelVars.index('label_H_bb')].reshape(-1)[jet_mask]
fpr_i, tpr_i, threshold_i = roc_curve(target_doubleb.reshape(-1,1), yi_doubleb.reshape(-1,1),drop_intermediate=False)
fpr_CMSXbb.append(np.interp(tpr_common, tpr_i, fpr_i))
auc_CMSXbb.append(auc(fpr_i,tpr_i))

fpr_CMSXbb_mean = fpr_CMSXbb[0]
auc_CMSXbb_mean = auc_CMSXbb[0]
fpr_ParTXbb_mean = np.mean(fpr_ParTXbb,axis=0)
fpr_ParTXbb_std = np.std(fpr_ParTXbb,axis=0)
auc_ParTXbb_mean = np.mean(auc_ParTXbb,axis=0)
auc_ParTXbb_std = np.std(auc_ParTXbb,axis=0)

#b=np.linspace(0,1,101)
#fig, ax = plt.subplots()
#ax.hist(yi_ParTXbb, lw=0.8,bins=b, weights=target_ParTXbb, label=('ParTXbb sig'),alpha=0.8,density=True)
#ax.hist(yi_ParTXbb, lw=0.8,bins=b, weights=(target_ParTXbb==0)*1, label=('ParTXbb bkg'),alpha=0.8,density=True)
#ax.hist(yi_doubleb, lw=0.8,bins=b, weights=target_doubleb, label=('fj_doubleb sig'),alpha=0.8,density=True)
#ax.hist(yi_doubleb, lw=0.8,bins=b, weights=(target_doubleb==0)*1, label=('fj_doubleb bkg'),alpha=0.8,density=True)
#ax.set_xlabel(r'Signal Efficiency')
#ax.set_ylabel(r'')
#ax.semilogy()
#ax.set_xlim(0,1)
#ax.grid(True)
#ax.legend(loc='lower left')
#ax.set_title(f"ROC")
#fig.savefig(f"../../Finetune_hep/plots/Full_preds_Xbb.pdf")

# plot ROC curve
fig = plt.figure()
ax = fig.add_subplot(4,1,(1,3))
ax.plot(tpr_common, 1/fpr_ParTXbb_mean, lw=0.8, label=f'ParTXbb AUC = {auc_ParTXbb_mean:.4f}',color='b')
ax.fill_between(tpr_common, (1/(fpr_ParTXbb_mean-fpr_ParTXbb_std)), (1/(fpr_ParTXbb_mean+fpr_ParTXbb_std)),color='b',alpha=0.2)
ax.plot(tpr_common, 1/fpr_CMSXbb_mean, lw=0.8, label=f'CMSXbb AUC = {auc_CMSXbb_mean:.4f}', color='r')
ax.set_ylabel(r'Background rejection')
ax.semilogy()
ax.set_ylim(0.01,10000)
ax.set_xlim(0.4,1)
ax.grid(True)
ax.legend(loc='lower left')
plt.setp(ax.get_xticklabels(), visible=False)
ax = fig.add_subplot(4,1,4)
ax.plot(tpr_common,(1/fpr_ParTXbb_mean)/(1/fpr_CMSXbb_mean),lw=0.8,color='b')
ax.fill_between( tpr_common, (1/(fpr_ParTXbb_mean-fpr_ParTXbb_std))/(1/fpr_CMSXbb_mean),
                 (1/(fpr_ParTXbb_mean+fpr_ParTXbb_std))/(1/fpr_CMSXbb_mean),color='b',alpha=0.2)
ax.plot(tpr_common,(1/fpr_CMSXbb_mean)/(1/fpr_CMSXbb_mean),lw=0.8,color='r')      
ax.set_xlim(0.4,1)
ax.set_ylabel(r'Ratio')
ax.grid(True)
ax.set_xlabel(r'Signal efficiency',loc="right")
fig.savefig(f"../../Finetune_hep/plots/Full_ROC_Xbb.pdf")


