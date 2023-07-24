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
import h5py
import pickle
vector.register_awkward()
from ParticleTransformer import ParticleTransformer


labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]       
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
pVars = [f'pfcand_{v}' for v in ['ptrel','erel','etarel','phirel','dxy','dxysig','dz','dzsig','deltaR','charge','isChargedHad','isNeutralHad','isGamma','isEl','isMu']]        

def log(data):
    ma_data = np.ma.masked_equal(data,0)
    result = np.log(ma_data)
    return result.filled(fill_value=0)  

def make_mlp(in_features,out_features,nlayer,for_inference=False):
    layers = []
    for i in range(nlayer):
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, 1))
    if for_inference: layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers)
    return model

class CustomDataset(Dataset):
    def __init__(self, filelist,device,scaler_path,Xbb_scores_path,test=False):
        self.device = device
        self.x=[]
        self.y=[]
        i=0
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                with h5py.File(filename, 'r') as Data:
                    if i ==0:
                        data = Data['X_jet'][:]
                        target = Data['labels'][:] 
                    else:
                        data = np.concatenate((data,Data['X_jet'][:]),axis=0)
                        target = np.concatenate((target,Data['labels'][:]),axis=0)
                    i+=1    
        self.scaler = StandardScaler() # this is super useful a scikit learn function
        data[:,:,jVars.index('fj_pt')] = log(data[:,:,jVars.index('fj_pt')])
        data[:,:,jVars.index('fj_mass')] = log(data[:,:,jVars.index('fj_mass')])
        data[:,:,jVars.index('fj_sdmass')] = log(data[:,:,jVars.index('fj_sdmass')])
        if Xbb_scores_path != 'no': 
            print('loading Xbb scores from : ',Xbb_scores_path)
            with h5py.File(Xbb_scores_path, 'r') as Xbb_scores:
                data[:,:,jVars.index('fj_doubleb')] = Xbb_scores['Xbb'][:]
        if scaler_path !='no' : 
            if (test == False): 
                X_norm = self.scaler.fit_transform(data.reshape(-1,12))
                self.x = torch.from_numpy(X_norm).float().to(device)
                with open(scaler_path,'wb') as f:
                    pickle.dump(self.scaler, f)
            else:         
                with open(scaler_path,'rb') as f:
                    self.scaler = pickle.load(f)
                X_norm = self.scaler.transform(data.reshape(-1,12))
                self.x = torch.from_numpy(X_norm).float().to(device)
        else:
            self.x = torch.from_numpy(data.reshape(-1,12)).float().to(device)    
        self.y = torch.from_numpy(target.reshape(-1,1)).float().to(device)
        self.length = len(target)
        print('N data : ',self.length)
        
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
             
    
def train_step(model,data,target,opt,loss_fn):
    model.train()
    preds = model(data)
    loss = loss_fn(preds,target)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model, loss_fn,train_loader,val_loader,subset,device):
    with torch.no_grad():
        model.eval()
        for i, train_batch in enumerate( train_loader ): 
            if i==0:
                data, target = train_batch
                data = data.cpu().numpy()
                target = target.cpu().numpy()
            else: 
                data = np.concatenate((data,train_batch[0].cpu().numpy()),axis=0)
                target = np.concatenate((target,train_batch[1].cpu().numpy()),axis=0)
            if (i > 100): break 
        for i, val_batch in enumerate( val_loader ):
            if i==0:
                data_val, target_val = val_batch
                data_val = data_val.cpu().numpy()
                target_val = target_val.cpu().numpy()
            else: 
                data_val = np.concatenate((data_val,val_batch[0].cpu().numpy()),axis=0)
                target_val = np.concatenate((target_val,val_batch[1].cpu().numpy()),axis=0)            
            if (subset and i > 10): break 

        train_loss = loss_fn(model( torch.from_numpy(data).float().to(device) ).reshape(len(data)),torch.from_numpy(target.reshape(-1)).float().to(device))
        test_loss = loss_fn(model( torch.from_numpy(data_val).float().to(device) ).reshape(len(data_val)),torch.from_numpy(target_val.reshape(-1)).float().to(device))
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    

def train_loop(model,filelist, device, experiment, path, scaler_path,Xbb_scores_path,subset,config):
    opt = optim.Adam(model.parameters(), config['LR'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.2]).to(device))
    evals = []
    best_val_loss = float('inf')
    with TemporaryDirectory() as tempdir:
        best_model_params_path = path
    Dataset = CustomDataset(filelist,device,scaler_path,Xbb_scores_path)
    num_samples = Dataset.length
    num_train = int(0.80 * num_samples)
    num_val = num_samples - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(Dataset, [num_train, num_val])    
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    for epoch in range (0,config['epochs']):
        print(f'epoch: {epoch+1}') 
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        for i, train_batch in enumerate( train_loader ):
            data, target = train_batch
            report = train_step(model, data, target, opt, loss_fn )
            if (subset and i > 10): break
        evals.append(eval_fn(model, loss_fn,train_loader,val_loader,subset,device) )    
        val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
            
        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    

    return evals, model


def get_preds(model,loader,subset,device):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate( loader ):
            if i==0:
                data, target = batch
                yi = model(data).detach().cpu().numpy()
                target = target.cpu().numpy()
            else: 
                yi = np.concatenate((yi,model(batch[0]).detach().cpu().numpy()),axis=0)
                target = np.concatenate((target,batch[1].cpu().numpy()),axis=0)
            if (subset and i > 10): break    
    return yi,target