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

def make_mlp(in_features,out_features,nlayer):
    layers = []
    for i in range(nlayer):
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, 1))
    layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers)
    return model

class CustomDataset(Dataset):
    def __init__(self, idxmap,integer_file_map,device,scaler_path):
        self.scaler_path = scaler_path
        self.device = device
        self.scaler = StandardScaler()
        self.integer_file_map = integer_file_map
        self.length = len(integer_file_map)
        self.idxmap = idxmap
        print("N data : ", self.length)      
        
    def __getitem__(self, index):
        file_path = self.integer_file_map[index][0]
        offset = np.min(self.idxmap[file_path])
        with h5py.File(file_path, 'r') as f:
            x = f['X_jet'][index-offset].reshape(12)
            y = f['labels'][index-offset].reshape(-1)
        self.x = torch.from_numpy(x).float().to(self.device)    
        self.y = torch.from_numpy(y).float().to(self.device)    
        return self.x,self.y
    def __len__(self):
        return self.length            
    
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
        for i, train_batch in enumerate( train_loader ):
            if i < 100:    
                if i==0:
                    data, target = train_batch
                    data = data.cpu().numpy()
                    target = target.cpu().numpy()
                else: 
                    data = np.concatenate((data,train_batch[0].cpu().numpy()),axis=0)
                    target = np.concatenate((target,train_batch[1].cpu().numpy()),axis=0)
            if (subset and i > 10): break 
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
    

def train_loop(model, idxmap,integer_file_map, device, experiment, path, scaler_path,subset,config):
    opt = optim.Adam(model.parameters(), config['LR'])
    loss_fn = nn.BCELoss()
    evals = []
    best_val_loss = float('inf')
    with TemporaryDirectory() as tempdir:
        best_model_params_path = path
    Dataset = CustomDataset(idxmap,integer_file_map,device,scaler_path)
    num_samples = Dataset.length
    num_train = int(0.80 * num_samples)
    num_val = num_samples - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(Dataset, [num_train, num_val])    
    for epoch in range (0,config['epochs']):
        print(f'epoch: {epoch+1}') 
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
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
                data = data.cpu().numpy()
                target = target.cpu().numpy()
            else: 
                data = np.concatenate((data,batch[0].cpu().numpy()),axis=0)
                target = np.concatenate((target,batch[1].cpu().numpy()),axis=0)
            if (subset and i > 10): break    
    print('oi')            
    print(data.shape)
    return model(torch.from_numpy(data).float().to(device)).detach().cpu().numpy(),target