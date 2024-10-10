import uproot 
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import pandas as pd
import h5py
import argparse
import sys
from Finetune_hep.python import helpers

parser = argparse.ArgumentParser(description='')
parser.add_argument('--bkg',  action='store_true', help='is bkg', default=False)
parser.add_argument('--mass', help='mass',default=0)
parser.add_argument('--fname', help='fname',default='')
parser.add_argument('--mess', help='mess',default='')
parser.add_argument('--out', help='outputDir',default='/raven/ptmp/mvigl/H5_samples_full')
parser.add_argument('--testonly',  action='store_true', help='save only test set', default=False)
args = parser.parse_args()

fname = args.fname
mass = float(args.mass)
bkg = args.bkg
mess = args.mess
out = args.out
testonly = args.testonly

tree = uproot.open(fname)['deepntuplizer/tree']     
print(tree)

helpers.jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
helpers.labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]
helpers.pVars = [f'pfcand_{v}' for v in ['ptrel','erel','etarel','phirel','dxy','dxysig','dz','dzsig','deltaR','charge','isChargedHad','isNeutralHad','isGamma','isEl','isMu']]

larr = tree.arrays(helpers.labelVars)
jarr = tree.arrays(helpers.jVars)
parr = tree.arrays(helpers.pVars)

if bkg: labels_tot=np.zeros(len(jarr))
else: labels_tot=np.ones(len(jarr))
mass_labels_tot=np.ones(len(jarr))*mass

njets = ak.count(jarr['fj_pt'],axis=-1)
minjets = np.where(njets>1)
print(len(njets[minjets])/len(njets))
larr = larr[minjets]
jarr = jarr[minjets]
parr = parr[minjets]
labels = labels_tot[minjets]
mass_labels = mass_labels_tot[minjets]
njets = njets[minjets]

if len(njets) < 2: 
    print('no events')
    sys.exit()

helpers.labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]
helpers.jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
helpers.pVars = [f'pfcand_{v}' for v in ['ptrel','erel','etarel','phirel','dxy','dxysig','dz','dzsig','deltaR','charge','isChargedHad','isNeutralHad','isGamma','isEl','isMu']]


maxJets = 5
maxPFOs = 100

labelFeat = len(helpers.labelVars)
jetFeat = len(helpers.jVars)
pfoFeat = len(helpers.pVars) 

size = 10000
if len(njets) < 10000: 
    size = len(njets)

njets_batches = np.array_split(np.arange(len(njets)),int(len(njets)/size))

for i in range(len(njets_batches)):

    njets_batch = njets[njets_batches[i]]
    jarr_batch = jarr[njets_batches[i]]
    larr_batch = larr[njets_batches[i]]
    parr_batch = parr[njets_batches[i]]

    print(f'batch number {i+1} out of {len(njets_batches)}')
    print(f'{len(njets_batch)} events')

    X_jet_batch = np.zeros((len(njets_batch),maxJets,jetFeat))
    X_pfo_batch = np.zeros((len(njets_batch),maxJets,maxPFOs,pfoFeat))
    X_label_batch = np.zeros((len(njets_batch),maxJets,labelFeat))


    ak_idx = ak.argsort(jarr_batch['fj_pt'],ascending=False)
    #ak_idx = ak.argsort(jarr_batch['fj_doubleb'],ascending=False)

    X_jet_batch = np.stack([np.asarray(ak.fill_none(ak.pad_none(jarr_batch[ak_idx][v], maxJets, clip=True), 0))
                      for v in helpers.jVars],axis=-1)


    X_label_batch = np.stack([np.asarray(ak.fill_none(ak.pad_none(larr_batch[v], maxJets, clip=True), 0))
                      for v in helpers.labelVars],axis=-1)        


    for jet_mult in range(2,maxJets+1):
        print(jet_mult)    
        mask = (njets_batch == jet_mult)
        print(jet_mult)
        vecs = [ak.fill_none(
                ak.pad_none(parr_batch[ak_idx][v][mask],maxPFOs,clip=True,axis=-1),
                0).to_numpy() 
            for v in helpers.pVars]

        print(np.shape(vecs))
        X_tmp = np.stack(vecs,axis=-1)
        if len(X_tmp)==0: continue
        if jet_mult < maxJets:
            X_pfo_batch[mask,:jet_mult] = X_tmp
        else:    
            X_pfo_batch[mask] = X_tmp[:,:maxJets]     


    if i == 0:
        X_jet = np.copy(X_jet_batch)
        X_label = np.copy(X_label_batch)
        X_pfo = np.copy(X_pfo_batch)
    else:
        X_jet = np.concatenate((X_jet,X_jet_batch),axis=0)
        X_label = np.concatenate((X_label,X_label_batch),axis=0)
        X_pfo = np.concatenate((X_pfo,X_pfo_batch),axis=0)


np.random.seed(1)
train_ix = np.sort(np.random.permutation(len(njets))[:int(len(njets)*0.8)])
test_ix = np.delete(np.arange(len(njets)),np.isin(np.arange(len(njets)),train_ix))  
val_ix = test_ix[:int(len(test_ix)/2)]
test_ix = test_ix[int(len(test_ix)/2):]
jet_mask = np.where(X_jet[:,:,0] == 0, 0, 1)

if (not testonly):
    evts_train = np.arange(len(X_label[train_ix]))[np.where(X_pfo[train_ix][:,0,0,0] != 0)[0]]
    print(len(evts_train)/len(train_ix))

    Data_train = h5py.File(f'{out}/Data_train_{mass}_{mess}.h5', 'w')
    Data_train.create_dataset('mass_param', data=mass_labels[train_ix][evts_train],dtype='i4')
    Data_train.create_dataset('labels', data=labels[train_ix][evts_train],dtype='i4')
    Data_train.create_dataset('jet_mask', data=jet_mask[train_ix][evts_train],dtype='i4')
    Data_train.create_dataset('X_jet', data=X_jet[train_ix][evts_train])
    Data_train.create_dataset('X_label', data=X_label[train_ix][evts_train],dtype='i4')
    Data_train.create_dataset('X_pfo', data=X_pfo[train_ix][evts_train])
    Data_train.create_dataset('X_jet_singlejet', data=(X_jet[train_ix][evts_train].reshape((-1,6))[jet_mask[train_ix][evts_train].reshape(-1).astype(bool)]))
    Data_train.create_dataset('X_pfo_singlejet', data=(X_pfo[train_ix][evts_train].reshape((-1,100,15))[jet_mask[train_ix][evts_train].reshape(-1).astype(bool)]))
    Data_train.create_dataset('X_label_singlejet', data=(X_label[train_ix][evts_train].reshape((-1,6))[jet_mask[train_ix][evts_train].reshape(-1).astype(bool)]),dtype='i4')
    Data_train.close()   

evts_test = np.arange(len(X_label[test_ix]))[np.where(X_pfo[test_ix][:,0,0,0] != 0)[0]]
print(len(evts_test)/len(test_ix))

Data_test = h5py.File(f'{out}/Data_test_{mass}_{mess}.h5', 'w')
Data_test.create_dataset('mass_param', data=mass_labels[test_ix][evts_test],dtype='i4')
Data_test.create_dataset('labels', data=labels[test_ix][evts_test],dtype='i4')
Data_test.create_dataset('jet_mask', data=jet_mask[test_ix][evts_test],dtype='i4')
Data_test.create_dataset('X_jet', data=X_jet[test_ix][evts_test])
Data_test.create_dataset('X_label', data=X_label[test_ix][evts_test],dtype='i4')
Data_test.create_dataset('X_pfo', data=X_pfo[test_ix][evts_test])
Data_test.create_dataset('X_jet_singlejet', data=(X_jet[test_ix][evts_test].reshape((-1,6))[jet_mask[test_ix][evts_test].reshape(-1).astype(bool)]))
Data_test.create_dataset('X_pfo_singlejet', data=(X_pfo[test_ix][evts_test].reshape((-1,100,15))[jet_mask[test_ix][evts_test].reshape(-1).astype(bool)]))
Data_test.create_dataset('X_label_singlejet', data=(X_label[test_ix][evts_test].reshape((-1,6))[jet_mask[test_ix][evts_test].reshape(-1).astype(bool)]),dtype='i4')
Data_test.close()        

if (not testonly):
    evts_val = np.arange(len(X_label[val_ix]))[np.where(X_pfo[val_ix][:,0,0,0] != 0)[0]]
    print(len(evts_val)/len(val_ix))

    Data_val = h5py.File(f'{out}/Data_val_{mass}_{mess}.h5', 'w')
    Data_val.create_dataset('mass_param', data=mass_labels[val_ix][evts_val],dtype='i4')
    Data_val.create_dataset('labels', data=labels[val_ix][evts_val],dtype='i4')
    Data_val.create_dataset('jet_mask', data=jet_mask[val_ix][evts_val],dtype='i4')
    Data_val.create_dataset('X_jet', data=X_jet[val_ix][evts_val])
    Data_val.create_dataset('X_label', data=X_label[val_ix][evts_val],dtype='i4')
    Data_val.create_dataset('X_pfo', data=X_pfo[val_ix][evts_val])
    Data_val.create_dataset('X_jet_singlejet', data=(X_jet[val_ix][evts_val].reshape((-1,6))[jet_mask[val_ix][evts_val].reshape(-1).astype(bool)]))
    Data_val.create_dataset('X_pfo_singlejet', data=(X_pfo[val_ix][evts_val].reshape((-1,100,15))[jet_mask[val_ix][evts_val].reshape(-1).astype(bool)]))
    Data_val.create_dataset('X_label_singlejet', data=(X_label[val_ix][evts_val].reshape((-1,6))[jet_mask[val_ix][evts_val].reshape(-1).astype(bool)]),dtype='i4')
    Data_val.close()    