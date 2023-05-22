from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd
import math
import vector
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.preprocessing import StandardScaler
from tempfile import TemporaryDirectory
import torch.optim as optim
vector.register_awkward()

labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]       
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
pVars = [f'pfcand_{v}' for v in ['ptrel','erel','etarel','phirel','dxy','dxysig','dz','dzsig','deltaR','charge','isChargedHad','isNeutralHad','isGamma','isEl','isMu']]        

def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device

def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x

def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))


def InverseDeltaPhi(phi2, DeltaPhi):
    k = ak.zeros_like(DeltaPhi)
    while ak.any(phi2 + DeltaPhi - k*math.pi > math.pi):
        awkward_array = ak.Array((phi2 + DeltaPhi - k*math.pi > math.pi))
        k = k+2*awkward_array
    while ak.any(phi2 + DeltaPhi - k*math.pi <= -math.pi):
        awkward_array = ak.Array((phi2 + DeltaPhi - k*math.pi <= -math.pi))
        k = k-2*awkward_array
    phi1 = phi2 + DeltaPhi - k*math.pi
    return phi1

def standard(data,mean,std):
    ma_data = np.ma.masked_equal(data,0)
    result = ( ( ma_data - mean ) * std )
    return result.filled(fill_value=0)

def log(data):
    ma_data = np.ma.masked_equal(data,0)
    result = np.log(ma_data)
    return result.filled(fill_value=0)  

def divide(data,data_2):
    ma_data = np.ma.masked_equal(data,0)
    ma_data_2 = np.ma.masked_equal(data_2,0)
    result = ma_data/ma_data_2
    return result.filled(fill_value=0)              

def build_features_and_labels(X_pfo,X_jet,X_label,njets, transform_features=True):

    # compute new features
    L = vector.array(
        {
            "pt": X_jet[:,:,jVars.index('fj_pt')],
            "phi": X_jet[:,:,jVars.index('fj_phi')],
            "eta": X_jet[:,:,jVars.index('fj_eta')],
            "M": X_jet[:,:,jVars.index('fj_sdmass')],
        }
    )
    fj_energy = L.energy
    a={}
    for var in pVars:
        a[var] = X_pfo[:,:,:,pVars.index(var)]
    a['label_H_bb'] =  X_label[:,:,labelVars.index('label_H_bb')] 

    etasign = np.sign(X_jet[:,:,jVars.index('fj_eta')])

    a['pfcand_energy'] =  a['pfcand_erel']*np.reshape(fj_energy,(len(X_jet),njets,1))
    a['pfcand_pt'] = a['pfcand_ptrel']*np.reshape(X_jet[:,:,jVars.index('fj_pt')],(len(X_jet),njets,1))
    a['pfcand_eta'] = (np.ma.masked_equal(np.reshape(X_jet[:,:,jVars.index('fj_eta')],(len(X_jet),njets,1)),0) + np.ma.masked_equal(a['pfcand_etarel']*np.reshape(etasign,(len(X_jet),njets,1)),0) ).filled(fill_value=0)
    a['pfcand_phi'] = InverseDeltaPhi(np.reshape(X_jet[:,:,jVars.index('fj_phi')],(len(X_jet),njets,1)) , a['pfcand_phirel'] )
    a['pfcand_px'] = a['pfcand_pt']*np.cos(np.ma.masked_equal(a['pfcand_phi'],0)).filled(fill_value=0)
    a['pfcand_py'] = a['pfcand_pt']*np.sin(np.ma.masked_equal(a['pfcand_phi'],0)).filled(fill_value=0)
    a['pfcand_pz'] = a['pfcand_pt']*np.sinh(np.ma.masked_equal(a['pfcand_eta'],0)).filled(fill_value=0)
    a['pfcand_dphi'] = np.copy(a['pfcand_phirel'])
    a['pfcand_deta'] = np.copy(a['pfcand_etarel'])

    arr = np.copy(a['pfcand_ptrel'])
    a['pfcand_mask'] = np.where(arr == 0, 0, 1)
    a['pfcand_pt_log'] = log(a['pfcand_pt'])
    a['pfcand_e_log'] = log(a['pfcand_energy'])
    a['pfcand_logptrel'] = log(X_pfo[:,:,:,pVars.index('pfcand_ptrel')])
    a['pfcand_logerel'] = log(X_pfo[:,:,:,pVars.index('pfcand_erel')])
    a['pfcand_d0err'] = divide(X_pfo[:,:,:,pVars.index('pfcand_dxy')] , X_pfo[:,:,:,pVars.index('pfcand_dxysig')])
    a['pfcand_dzerr'] = divide(X_pfo[:,:,:,pVars.index('pfcand_dz')] , X_pfo[:,:,:,pVars.index('pfcand_dzsig')])
    a['pfcand_d0'] = np.tanh(np.ma.masked_equal(X_pfo[:,:,:,pVars.index('pfcand_dxy')],0)).filled(fill_value=0)
    a['pfcand_dz'] = np.tanh(np.ma.masked_equal(X_pfo[:,:,:,pVars.index('pfcand_dz')],0)).filled(fill_value=0)



    # apply standardization
    if transform_features:
        a['pfcand_pt_log'] = standard(a['pfcand_pt_log'],1.7,0.7) 
        a['pfcand_e_log'] = standard(a['pfcand_e_log'],2.0,0.7) 
        a['pfcand_logptrel'] = standard(a['pfcand_logptrel'],-4.7,0.7) 
        a['pfcand_logerel'] = standard(a['pfcand_logerel'],-4.7,0.7)
        a['pfcand_deltaR'] = standard(X_pfo[:,:,:,pVars.index('pfcand_deltaR')],0.2,4.0) 
        a['pfcand_d0err'] = _clip(a['pfcand_d0err'] , 0, 1)
        a['pfcand_dzerr'] = _clip(a['pfcand_dzerr'], 0, 1)

    feature_list = {
        'pf_points': ['pfcand_deta', 'pfcand_dphi'], # not used in ParT
        'pf_features': ['pfcand_pt_log',
                        'pfcand_e_log',
                        'pfcand_logptrel',
                        'pfcand_logerel',
                        'pfcand_deltaR',
                        'pfcand_charge', 
                        'pfcand_isChargedHad',
                        'pfcand_isNeutralHad', 
                        'pfcand_isGamma',
                        'pfcand_isEl', 
                        'pfcand_isMu', 
                        'pfcand_d0', 
                        'pfcand_d0err',
                        'pfcand_dz', 
                        'pfcand_dzerr', 
                        'pfcand_deta',
                        'pfcand_dphi',
                        ],
        'pf_vectors': [
            'pfcand_px',
            'pfcand_py',
            'pfcand_pz',
            'pfcand_energy',
        ],
        'pf_mask': ['pfcand_mask']
    }
    
    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([a[n] for n in names], axis=2)
        
    label_list = ['label_H_bb']                  
    out['label'] = np.stack([a[n].astype('int') for n in label_list], axis=2)

    return out


def build_features_and_labels_single_jet(X_pfo,X_jet,X_label, transform_features=True):

    # compute new features
    L = vector.array(
        {
            "pt": X_jet[:,jVars.index('fj_pt')],
            "phi": X_jet[:,jVars.index('fj_phi')],
            "eta": X_jet[:,jVars.index('fj_eta')],
            "M": X_jet[:,jVars.index('fj_sdmass')],
        }
    )
    fj_energy = L.energy
    a={}
    for var in pVars:
        a[var] = X_pfo[:,:,pVars.index(var)]
    a['label_H_bb'] =  X_label[:,labelVars.index('label_H_bb')] 


    etasign = np.sign(X_jet[:,jVars.index('fj_eta')])

    a['pfcand_energy'] =  a['pfcand_erel']*np.reshape(fj_energy,(len(X_jet),1))
    a['pfcand_pt'] = a['pfcand_ptrel']*np.reshape(X_jet[:,jVars.index('fj_pt')],(len(X_jet),1))
    a['pfcand_eta'] = (np.ma.masked_equal(np.reshape(X_jet[:,jVars.index('fj_eta')],(len(X_jet),1)),0) + np.ma.masked_equal(a['pfcand_etarel']*np.reshape(etasign,(len(X_jet),1)),0) ).filled(fill_value=0)
    a['pfcand_phi'] = InverseDeltaPhi(np.reshape(X_jet[:,jVars.index('fj_phi')],(len(X_jet),1)) , a['pfcand_phirel'] )
    a['pfcand_px'] = a['pfcand_pt']*np.cos(np.ma.masked_equal(a['pfcand_phi'],0)).filled(fill_value=0)
    a['pfcand_py'] = a['pfcand_pt']*np.sin(np.ma.masked_equal(a['pfcand_phi'],0)).filled(fill_value=0)
    a['pfcand_pz'] = a['pfcand_pt']*np.sinh(np.ma.masked_equal(a['pfcand_eta'],0)).filled(fill_value=0)
    a['pfcand_dphi'] = np.copy(a['pfcand_phirel'])
    a['pfcand_deta'] = np.copy(a['pfcand_etarel'])

 

    arr = np.copy(a['pfcand_ptrel'])
    a['pfcand_mask'] = np.where(arr == 0, 0, 1)



    a['pfcand_pt_log'] = log(a['pfcand_pt'])
    a['pfcand_e_log'] = log(a['pfcand_energy'])
    a['pfcand_logptrel'] = log(X_pfo[:,:,pVars.index('pfcand_ptrel')])
    a['pfcand_logerel'] = log(X_pfo[:,:,pVars.index('pfcand_erel')])
    a['pfcand_d0err'] = divide(X_pfo[:,:,pVars.index('pfcand_dxy')] , X_pfo[:,:,pVars.index('pfcand_dxysig')])
    a['pfcand_dzerr'] = divide(X_pfo[:,:,pVars.index('pfcand_dz')] , X_pfo[:,:,pVars.index('pfcand_dzsig')])
    a['pfcand_d0'] = np.tanh(np.ma.masked_equal(X_pfo[:,:,pVars.index('pfcand_dxy')],0)).filled(fill_value=0)
    a['pfcand_dz'] = np.tanh(np.ma.masked_equal(X_pfo[:,:,pVars.index('pfcand_dz')],0)).filled(fill_value=0)



    # apply standardization
    if transform_features:

        a['pfcand_pt_log'] = standard(a['pfcand_pt_log'],1.7,0.7) 
        a['pfcand_e_log'] = standard(a['pfcand_e_log'],2.0,0.7) 
        a['pfcand_logptrel'] = standard(a['pfcand_logptrel'],-4.7,0.7) 
        a['pfcand_logerel'] = standard(a['pfcand_logerel'],-4.7,0.7)
        a['pfcand_deltaR'] = standard(X_pfo[:,:,pVars.index('pfcand_deltaR')],0.2,4.0) 
        a['pfcand_d0err'] = _clip(a['pfcand_d0err'] , 0, 1)
        a['pfcand_dzerr'] = _clip(a['pfcand_dzerr'], 0, 1)

    feature_list = {
        'pf_points': ['pfcand_deta', 'pfcand_dphi'], # not used in ParT
        'pf_features': ['pfcand_pt_log',
                        'pfcand_e_log',
                        'pfcand_logptrel',
                        'pfcand_logerel',
                        'pfcand_deltaR',
                        'pfcand_charge', 
                        'pfcand_isChargedHad',
                        'pfcand_isNeutralHad', 
                        'pfcand_isGamma',
                        'pfcand_isEl', 
                        'pfcand_isMu', 
                        'pfcand_d0', 
                        'pfcand_d0err',
                        'pfcand_dz', 
                        'pfcand_dzerr', 
                        'pfcand_deta',
                        'pfcand_dphi',
                        ],
        'pf_vectors': [
            'pfcand_px',
            'pfcand_py',
            'pfcand_pz',
            'pfcand_energy',
        ],
        'pf_mask': ['pfcand_mask']
    }

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([a[n] for n in names], axis=1)


    label_list = ['label_H_bb']                  
    out['label'] = np.stack([a[n].astype('int') for n in label_list], axis=1)

    
    return out


def plot_evals(evals,label):
    l = plt.plot(pd.DataFrame([e['train_loss'] for e in evals]), linestyle='dashed')
    plt.plot(pd.DataFrame([e['test_loss'] for e in evals]), c = l[0].get_color(), label = label)



def get_train_data(path,subset=False):

    if subset == True:
        with open(path+'/X_pfo_train.npy', 'rb') as f:
            X_pfo_train=np.load(f)[:4096]
        with open(path+'/X_jet_train.npy', 'rb') as f:
            X_jet_train=np.load(f)[:4096]  
        with open(path+'/njets_train.npy', 'rb') as f:
            njets_train=np.load(f)[:4096]        
        with open(path+'/labels_train.npy', 'rb') as f:
            labels_train=np.load(f)[:4096]
        with open(path+'/X_label_train.npy', 'rb') as f:
            X_label_train=np.load(f)[:4096] 
    else:
        with open(path+'/X_pfo_train.npy', 'rb') as f:
            X_pfo_train=np.load(f)
        with open(path+'/X_jet_train.npy', 'rb') as f:
            X_jet_train=np.load(f)   
        with open(path+'/njets_train.npy', 'rb') as f:
            njets_train=np.load(f)        
        with open(path+'/labels_train.npy', 'rb') as f:
            labels_train=np.load(f)
        with open(path+'/X_label_train.npy', 'rb') as f:
            X_label_train=np.load(f)             

    evts_train = np.arange(len(X_label_train))[np.where(X_pfo_train[:,0,0,0] != 0)[0]]
    labels_train = labels_train[evts_train]

    return X_pfo_train, X_jet_train, njets_train, labels_train, X_label_train, evts_train

def get_test_data(path):

    with open(path+'/X_pfo_test.npy', 'rb') as f:
        X_pfo_test=np.load(f)
    with open(path+'/X_jet_test.npy', 'rb') as f:
        X_jet_test=np.load(f)   
    with open(path+'/njets_test.npy', 'rb') as f:
        njets_test=np.load(f)  
    with open(path+'/labels_test.npy', 'rb') as f:
        labels_test=np.load(f)
    with open(path+'/X_label_test.npy', 'rb') as f:
        X_label_test=np.load(f) 

    evts_test = np.arange(len(X_label_test))[np.where(X_pfo_test[:,0,0,0] != 0)[0]]
    labels_test = labels_test[evts_test]    

    return X_pfo_test, X_jet_test, njets_test, labels_test, X_label_test, evts_test

def load_weights_ParT_mlp(model,mlp_layers=0,ParT_params_path='no',mlp_params_path='no'):    

    if (ParT_params_path != 'no'):
        for i, layer in enumerate(torch.load(ParT_params_path).keys()):
            if i > (mlp_layers*2-1):
                model.state_dict()[layer].copy_(torch.load(ParT_params_path)[layer])
    if (mlp_params_path != 'no'):
        for i, layer in enumerate(torch.load(mlp_params_path).keys()):
            if i <= (mlp_layers*2-1):
                model.state_dict()[layer].copy_(torch.load(mlp_params_path)[layer])

    return model    

def getXbb_scores(Xbb_scores_path,evts):
    with open(Xbb_scores_path, 'rb') as f:
        Xbb_scores=np.load(f)[evts]
    return Xbb_scores

def get_mlp_feat(X_jet,njets,modeltype,evts,Xbb_scores_path='no',subset=False):
    if (Xbb_scores_path != 'no'): 
        if subset == True:
            X_jet[:,:,jVars.index('fj_doubleb')] = getXbb_scores(Xbb_scores_path,evts)[:4096] 
        else:
            X_jet[:,:,jVars.index('fj_doubleb')] = getXbb_scores(Xbb_scores_path,evts)
    if modeltype == 'mlpXbb':
        data = np.reshape(X_jet[:,:njets,jVars.index('fj_doubleb')],(-1,2))
    elif modeltype == 'mlpHlXbb':
        X_jet[:,:,jVars.index('fj_mass')] = log(X_jet[:,:,jVars.index('fj_mass')]) 
        X_jet[:,:,jVars.index('fj_sdmass')] = log(X_jet[:,:,jVars.index('fj_sdmass')])  
        X_jet[:,:,jVars.index('fj_pt')] = log(X_jet[:,:,jVars.index('fj_pt')]) 
        data = np.reshape(X_jet[:,:njets],(-1,12))
    return data
        