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


import pickle
class myDataset(Dataset):
    def __init__( self, X, y,device):
        super(myDataset, self).__init__()
        # Normalize the inputs
        self.scaler = StandardScaler() # this is super useful a scikit learn function
        X_norm = self.scaler.fit_transform(X)
        self.x = torch.from_numpy(X_norm).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)
        with open('/home/iwsatlas1/mavigl/Hbb/ParT/Trained_ParT/models/scaler_latent.pkl','wb') as f:
            pickle.dump(self.scaler, f)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def train_step(model,data,target,opt,loss_fn):
    model.train()
    preds = model(data).reshape(len(data))
    loss = loss_fn(preds,target)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model, loss_fn,train_set,val_set):
    with torch.no_grad():
        data, target = train_set
        data_val, target_val = val_set
        model.eval()
        train_loss = loss_fn(model(data).reshape(len(data)),target.float())
        test_loss = loss_fn(model(data_val).reshape(len(data_val)),target_val.float())
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    

def sample(dataset, batch_size):
    N = len(dataset)
    ix = np.arange(N)
    np.random.seed(1)
    np.random.shuffle(ix)
    #Samplers for the train / val split
    N_tr = np.floor(0.8 * N).astype(int)
    train_batches = DataLoader(
        dataset, batch_size=batch_size, sampler=SubsetRandomSampler(ix[:N_tr])
    )
    train_set = DataLoader(
        dataset, batch_size=int(N_tr), sampler=SubsetRandomSampler(ix[:N_tr])
    )
    validtion_set = DataLoader(
        dataset, batch_size=int(N-N_tr), sampler=SubsetRandomSampler(ix[N_tr:])
    ) # It doesn't matter that the validation set is randomly sampled
    return train_batches, train_set, validtion_set
    
def train_loop(model, X,y, device, experiment, path, config):
    dataset = myDataset(X,y,device)
    opt = optim.Adam(model.parameters(), config['LR'])
    loss_fn = nn.BCELoss()
    evals = []
    best_val_loss = float('inf')
    with TemporaryDirectory() as tempdir:
        best_model_params_path = path
    for epoch in range (0,config['epochs']):
        print(f'epoch: {epoch+1}') 
        train_batches, train_set, validtion_set = sample(dataset, config['batch_size'])
        for i, train_batch in enumerate( train_batches ):
            data, target = train_batch
            report = train_step(model, data, target.float(), opt, loss_fn )
        for train, val in zip(train_set, validtion_set):    
            evals.append(eval_fn(model, loss_fn,train,val) )    
            val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
            
        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    

    return evals, model