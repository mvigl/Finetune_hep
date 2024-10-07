import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import math
import vector
import torch
from torch.utils.data import Dataset
import h5py
vector.register_awkward()

labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]       
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
pVars = [f'pfcand_{v}' for v in ['ptrel','erel','etarel','phirel','dxy','dxysig','dz','dzsig','deltaR','charge','isChargedHad','isNeutralHad','isGamma','isEl','isMu']]        

def make_mlp(in_features,out_features,nlayer,for_inference=False,binary=True):
    layers = []
    for i in range(nlayer):
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    if binary: layers.append(torch.nn.Linear(in_features, 1))
    if for_inference: layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers)
    return model

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


def build_features_and_labels_Xbb(Data, transform_features=True):

    # compute new features
    L = vector.array(
        {
            "pt": Data['X_jet'][:,jVars.index('fj_pt')],
            "phi": Data['X_jet'][:,jVars.index('fj_phi')],
            "eta": Data['X_jet'][:,jVars.index('fj_eta')],
            "M": Data['X_jet'][:,jVars.index('fj_sdmass')],
        }
    )
    fj_energy = L.energy
    a={}
    for var in pVars:
        a[var] = Data['X_pfo'][:,:,pVars.index(var)]
    a['label_H_bb'] =  Data['labels'][:,labelVars.index('label_H_bb')] 

    etasign = np.sign(Data['X_jet'][:,jVars.index('fj_eta')])

    a['pfcand_energy'] =  a['pfcand_erel']*(fj_energy[:,np.newaxis])
    a['pfcand_pt'] = a['pfcand_ptrel']*(Data['X_jet'][:,jVars.index('fj_pt')][:,np.newaxis])
    a['pfcand_eta'] = (np.ma.masked_equal((Data['X_jet'][:,jVars.index('fj_eta')][:,np.newaxis]),0) + np.ma.masked_equal(a['pfcand_etarel']*(etasign[:,np.newaxis]),0)).filled(fill_value=0)
    a['pfcand_phi'] = InverseDeltaPhi((Data['X_jet'][:,jVars.index('fj_phi')][:,np.newaxis]) , a['pfcand_phirel'] )
    a['pfcand_px'] = a['pfcand_pt']*np.cos(np.ma.masked_equal(a['pfcand_phi'],0)).filled(fill_value=0)
    a['pfcand_py'] = a['pfcand_pt']*np.sin(np.ma.masked_equal(a['pfcand_phi'],0)).filled(fill_value=0)
    a['pfcand_pz'] = a['pfcand_pt']*np.sinh(np.ma.masked_equal(a['pfcand_eta'],0)).filled(fill_value=0)
    a['pfcand_dphi'] = np.copy(a['pfcand_phirel'])
    a['pfcand_deta'] = np.copy(a['pfcand_etarel'])

    arr = np.copy(a['pfcand_ptrel'])
    a['pfcand_mask'] = np.where(arr == 0, 0, 1)
    a['pfcand_pt_log'] = log(a['pfcand_pt'])
    a['pfcand_e_log'] = log(a['pfcand_energy'])
    a['pfcand_logptrel'] = log(Data['X_pfo'][:,:,pVars.index('pfcand_ptrel')])
    a['pfcand_logerel'] = log(Data['X_pfo'][:,:,pVars.index('pfcand_erel')])
    a['pfcand_d0err'] = divide(Data['X_pfo'][:,:,pVars.index('pfcand_dxy')] , Data['X_pfo'][:,:,pVars.index('pfcand_dxysig')])
    a['pfcand_dzerr'] = divide(Data['X_pfo'][:,:,pVars.index('pfcand_dz')] , Data['X_pfo'][:,:,pVars.index('pfcand_dzsig')])
    a['pfcand_d0'] = np.tanh(np.ma.masked_equal(Data['X_pfo'][:,:,pVars.index('pfcand_dxy')],0)).filled(fill_value=0)
    a['pfcand_dz'] = np.tanh(np.ma.masked_equal(Data['X_pfo'][:,:,pVars.index('pfcand_dz')],0)).filled(fill_value=0)



    # apply standardization
    if transform_features:
        a['pfcand_pt_log'] = standard(a['pfcand_pt_log'],1.7,0.7) 
        a['pfcand_e_log'] = standard(a['pfcand_e_log'],2.0,0.7) 
        a['pfcand_logptrel'] = standard(a['pfcand_logptrel'],-4.7,0.7) 
        a['pfcand_logerel'] = standard(a['pfcand_logerel'],-4.7,0.7)
        a['pfcand_deltaR'] = standard(Data['X_pfo'][:,:,pVars.index('pfcand_deltaR')],0.2,4.0) 
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




def build_features_and_labels(Data, transform_features=True):

    # compute new features
    L = vector.array(
        {
            "pt": Data['X_jet'][:,:,jVars.index('fj_pt')],
            "phi": Data['X_jet'][:,:,jVars.index('fj_phi')],
            "eta": Data['X_jet'][:,:,jVars.index('fj_eta')],
            "M": Data['X_jet'][:,:,jVars.index('fj_sdmass')],
        }
    )
    fj_energy = L.energy
    a={}
    for var in pVars:
        a[var] = Data['X_pfo'][:,:,:,pVars.index(var)]
    a['label_sig'] = Data['labels']
    a['jet_mask'] = Data['jet_mask']

    etasign = np.sign(Data['X_jet'][:,:,jVars.index('fj_eta')])

    a['pfcand_energy'] =  a['pfcand_erel']*(fj_energy[:,:,np.newaxis])
    a['pfcand_pt'] = a['pfcand_ptrel']*(Data['X_jet'][:,:,jVars.index('fj_pt')][:,:,np.newaxis])
    a['pfcand_eta'] = (np.ma.masked_equal((Data['X_jet'][:,:,jVars.index('fj_eta')][:,:,np.newaxis]),0) + np.ma.masked_equal(a['pfcand_etarel']*(etasign[:,:,np.newaxis]),0)).filled(fill_value=0)
    a['pfcand_phi'] = InverseDeltaPhi((Data['X_jet'][:,:,jVars.index('fj_phi')][:,:,np.newaxis]) , a['pfcand_phirel'] )
    a['pfcand_px'] = a['pfcand_pt']*np.cos(np.ma.masked_equal(a['pfcand_phi'],0)).filled(fill_value=0)
    a['pfcand_py'] = a['pfcand_pt']*np.sin(np.ma.masked_equal(a['pfcand_phi'],0)).filled(fill_value=0)
    a['pfcand_pz'] = a['pfcand_pt']*np.sinh(np.ma.masked_equal(a['pfcand_eta'],0)).filled(fill_value=0)
    a['pfcand_dphi'] = np.copy(a['pfcand_phirel'])
    a['pfcand_deta'] = np.copy(a['pfcand_etarel'])

    arr = np.copy(a['pfcand_ptrel'])
    a['pfcand_mask'] = np.where(arr == 0, 0, 1)
    a['pfcand_pt_log'] = log(a['pfcand_pt'])
    a['pfcand_e_log'] = log(a['pfcand_energy'])
    a['pfcand_logptrel'] = log(Data['X_pfo'][:,:,:,pVars.index('pfcand_ptrel')])
    a['pfcand_logerel'] = log(Data['X_pfo'][:,:,:,pVars.index('pfcand_erel')])
    a['pfcand_d0err'] = divide(Data['X_pfo'][:,:,:,pVars.index('pfcand_dxy')] , Data['X_pfo'][:,:,:,pVars.index('pfcand_dxysig')])
    a['pfcand_dzerr'] = divide(Data['X_pfo'][:,:,:,pVars.index('pfcand_dz')] , Data['X_pfo'][:,:,:,pVars.index('pfcand_dzsig')])
    a['pfcand_d0'] = np.tanh(np.ma.masked_equal(Data['X_pfo'][:,:,:,pVars.index('pfcand_dxy')],0)).filled(fill_value=0)
    a['pfcand_dz'] = np.tanh(np.ma.masked_equal(Data['X_pfo'][:,:,:,pVars.index('pfcand_dz')],0)).filled(fill_value=0)



    # apply standardization
    if transform_features:
        a['pfcand_pt_log'] = standard(a['pfcand_pt_log'],1.7,0.7) 
        a['pfcand_e_log'] = standard(a['pfcand_e_log'],2.0,0.7) 
        a['pfcand_logptrel'] = standard(a['pfcand_logptrel'],-4.7,0.7) 
        a['pfcand_logerel'] = standard(a['pfcand_logerel'],-4.7,0.7)
        a['pfcand_deltaR'] = standard(Data['X_pfo'][:,:,:,pVars.index('pfcand_deltaR')],0.2,4.0) 
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

    jet_mask_list = ['jet_mask']
    out['jet_mask'] = np.stack([a[n].astype('int') for n in jet_mask_list], axis=2)

    evt_label_list = ['label_sig']
    out['label'] = np.stack([a[n].astype('int') for n in evt_label_list], axis=1)

    Data['X_jet'][:,:,jVars.index('fj_pt')] = log(Data['X_jet'][:,:,jVars.index('fj_pt')])
    Data['X_jet'][:,:,jVars.index('fj_mass')] = log(Data['X_jet'][:,:,jVars.index('fj_mass')])
    Data['X_jet'][:,:,jVars.index('fj_sdmass')] = log(Data['X_jet'][:,:,jVars.index('fj_sdmass')])
    hlfeats_ix = [jVars.index('fj_pt'),jVars.index('fj_eta'),jVars.index('fj_phi'),jVars.index('fj_mass'),jVars.index('fj_sdmass')]            
    out['hl_feats'] = Data['X_jet'][:,:,hlfeats_ix]
    out['pf_mask'][:,:,:,:2] += np.abs(out['jet_mask'][:,:,np.newaxis]-1)
    return out

def get_idxmap(filelist,subset=1):
    idxmap = {}
    offset = 0 
    with open(filelist) as f:
        for line in f:
            filename = line.strip()
            with h5py.File(filename, 'r') as Data:
                idxmap[filename] = np.arange(offset,offset+int(len(Data['labels'][:])*subset))
                offset += int(len(Data['labels'][:])*subset)
    return idxmap            

def get_idxmap_Xbb(filelist,subset=1):
    idxmap = {}
    offset = 0 
    with open(filelist) as f:
        for line in f:
            filename = line.strip()
            with h5py.File(filename, 'r') as Data:
                idxmap[filename] = np.arange(offset,offset+int(len(Data['X_label_singlejet'][:]))*subset)
                offset += int(len(Data['X_label_singlejet'][:])*subset)
    return idxmap              
         

def create_integer_file_map(idxmap):
    integer_file_map = {}
    file_names = list(idxmap.keys())
    file_vectors = list(idxmap.values())
    for i, file in enumerate(file_names):
        vector = file_vectors[i]
        for integer in vector:
            if integer in integer_file_map:
                integer_file_map[integer].append(file)
            else:
                integer_file_map[integer] = [file]

    return integer_file_map     
    
class CustomDataset(Dataset):
    def __init__(self, idxmap,integer_file_map):
        self.integer_file_map = integer_file_map
        self.length = len(integer_file_map)
        self.idxmap = idxmap
        print("N data : ", self.length)
        
    def __getitem__(self, index):
        file_path = self.integer_file_map[index][0]
        offset = np.min(self.idxmap[file_path])
        data = {}
        with h5py.File(file_path, 'r') as f:
            data['X_jet'] = f['X_jet'][index-offset]
            data['X_pfo'] = f['X_pfo'][index-offset]
            data['X_label'] = f['X_label'][index-offset]
            data['labels'] = f['labels'][index-offset]
            data['jet_mask'] = f['jet_mask'][index-offset]
        return data
    
    def __len__(self):
        return self.length    


class Xbb_CustomDataset(Dataset):
    def __init__(self, idxmap,integer_file_map):
        self.integer_file_map = integer_file_map
        self.length = len(integer_file_map)
        self.idxmap = idxmap
        print("N data : ", self.length)
        
    def __getitem__(self, index):
        file_path = self.integer_file_map[index][0]
        offset = np.min(self.idxmap[file_path])
        data = {}
        with h5py.File(file_path, 'r') as f:
            data['X_jet'] = f['X_jet_singlejet'][int(index-offset)]
            data['X_pfo'] = f['X_pfo_singlejet'][int(index-offset)]
            data['labels'] = f['X_label_singlejet'][int(index-offset)]
        return data
    
    def __len__(self):
        return self.length 


def load_weights(model,weights,device):
    pretrained_dict = torch.load(weights,map_location=torch.device(device))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print('loading weights :')
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    return model