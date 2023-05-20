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



def infer(model,data,train_batch,device):
    N = train_batch
    pf_points = torch.tensor(data['pf_points'][N]).float().to(device)
    pf_features = torch.tensor(data['pf_features'][N]).float().to(device)
    pf_vectors = torch.tensor(data['pf_vectors'][N]).float().to(device)
    pf_mask = torch.tensor(data['pf_mask'][N]).float().to(device)
    preds = model(pf_points,pf_features,pf_vectors,pf_mask)
    pf_points.to("cpu")
    pf_features.to("cpu")
    pf_vectors.to("cpu")
    pf_mask.to("cpu")
    return preds.reshape(len(train_batch))

def infer_val(model,data,train_batch,device):
    with torch.no_grad():
        return infer(model,data,train_batch,device)
    

def train_step(model,data,labels,opt,loss_fn,train_batch,device):
    model.train()
    preds = infer(model,data,train_batch)
    target = torch.tensor(labels[train_batch]).float().to(device)
    loss = loss_fn(preds,target)
    target.to("cpu")
    preds.to("cpu")
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model,labels, loss_fn,data,train_set,validtion_set,device):
    with torch.no_grad():
        model.eval()

        ix_train = np.array_split(train_set,int(len(train_set)/512))

        for i in range(len(ix_train)):
            preds_train_i = infer_val(model,data,ix_train[i]).reshape(len(ix_train[i]))
            if i==0:
                preds_train = preds_train_i.detach().cpu().numpy()
            else:    
                preds_train = np.concatenate((preds_train,preds_train_i.detach().cpu().numpy()),axis=0)
        preds_train = torch.tensor(preds_train).float().to(device)

        ix_val = np.array_split(validtion_set,int(len(validtion_set)/512))

        for i in range(len(ix_val)):
            preds_val_i = infer_val(model,data,ix_val[i]).reshape(len(ix_val[i]))
            if i==0:
                preds_val = preds_val_i.detach().cpu().numpy()
            else:    
                preds_val = np.concatenate((preds_val,preds_val_i.detach().cpu().numpy()),axis=0)       
        preds_val = torch.tensor(preds_val).float().to(device)
        
        target_train = torch.tensor(labels[train_set]).float().to(device)
        train_loss = loss_fn(preds_train,target_train)
        target_train.to("cpu")
        preds_train.to("cpu")
        target_val = torch.tensor(labels[validtion_set]).float().to(device)
        test_loss = loss_fn(preds_val,target_val)
        target_val.to("cpu")
        preds_val.to("cpu")
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    
    
def train_loop(model, data, labels, config, device):
    opt = optim.Adam(model.parameters(), config['LR'])
    loss_fn = nn.BCELoss()
    evals = []
    best_val_loss = float('inf')
    N = len(data['pf_features'])
    ix = np.arange(N)
    ix2 = np.arange(N)
    np.random.seed(1)
    np.random.shuffle(ix)
    N_tr = np.floor(0.8 * N).astype(int)
    train_set = ix2[ix[:N_tr]]
    validtion_set = ix2[ix[N_tr:]]
    with TemporaryDirectory() as tempdir:
        best_model_params_path = (f'/home/iwsatlas1/mavigl/Hbb/ParT/Trained_ParT/models/COMBINED_TRAINING_{config["LR"]}_{config["batch_size"]}.pt' )
    for epoch in range (0,config['epochs']):
        train_batches = DataLoader(
            ix2,  batch_size=config['batch_size'], sampler=SubsetRandomSampler(ix[:N_tr])
        )
        print(f'epoch: {epoch+1}') 
        for i, train_batch in enumerate( train_batches ):
            train_batch = train_batch.numpy()
            report = train_step(model, data, labels, opt, loss_fn,train_batch )
        evals.append(eval_fn(model,labels, loss_fn,data,train_set,validtion_set) )    
        val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    

    return evals, model


def plot_evals(evals,label):
    l = plt.plot(pd.DataFrame([e['train_loss'] for e in evals]), linestyle='dashed')
    plt.plot(pd.DataFrame([e['test_loss'] for e in evals]), c = l[0].get_color(), label = label)


def get_preds(model,data,evts,device):

    ix = np.array_split(np.arange(len(evts)),int(len(evts)/512))
    for i in range(len(ix)):
        preds_i = infer_val(model,data,ix[i],device).reshape(len(ix[i]))
        if i==0:
            yi_model = preds_i.detach().cpu().numpy()
        else:    
            yi_model = np.concatenate((yi_model,preds_i.detach().cpu().numpy()),axis=0)
    return yi_model


def get_train_data(path):
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