import sys, os
sys.path.append('/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/python')

import definitions as df
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

class ParticleTransformerWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        in_dim = kwargs['embed_dims'][-1]
        fc_params = kwargs.pop('fc_params')
        num_classes = kwargs.pop('num_classes')
        self.for_inference = kwargs['for_inference']

        fcs = []
        self.fc = make_mlp(in_dim*2,out_features=128,nlayer = 3)

        kwargs['num_classes'] = None
        kwargs['fc_params'] = None
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        features = torch.reshape(features,(-1,17,110))
        lorentz_vectors = torch.reshape(lorentz_vectors,(-1,4,110))
        mask = torch.reshape(mask,(-1,1,110))
        x_cls = self.mod(features, v=lorentz_vectors, mask=mask) 
        output_parT = torch.reshape(x_cls,(-1,2*128))
        output = self.fc(output_parT)
        return output

def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config['inputs']['pf_features']['vars']),
        num_classes=len(data_config['labels']['value']),
        # network configurations
        pair_input_dim=4,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],       
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
    )
    cfg.update(**kwargs)

    model = ParticleTransformerWrapper(**cfg)
    return model

def get_loss(data_config, **kwargs):
    return torch.nn.BCELoss()


def infer(model,batch,device):
    pf_points = torch.tensor(batch['pf_points']).float().to(device)
    pf_features = torch.tensor(batch['pf_features']).float().to(device)
    pf_vectors = torch.tensor(batch['pf_vectors']).float().to(device)
    pf_mask = torch.tensor(batch['pf_mask']).float().to(device)
    preds = model(pf_points,pf_features,pf_vectors,pf_mask)
    return preds.reshape(-1)

def infer_val(model,batch,device):
    with torch.no_grad():
        return infer(model,batch,device)
    

def train_step(model,opt,loss_fn,train_batch,device):
    model.train()
    preds = infer(model,train_batch,device)
    target = torch.tensor(train_batch['evt_label']).float().to(device)
    loss = loss_fn(preds,target)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model,loss_fn,train_loader,val_loader,device):
    with torch.no_grad():

        for i, train_batch in enumerate( train_loader ):
            if i < 100:    
                if i==0:
                    preds_train = infer_val(model,train_batch,device).detach().cpu().numpy()
                    target_train = train_batch['evt_label']
                else:    
                    preds_train = np.concatenate((preds_train,infer_val(model,train_batch,device).detach().cpu().numpy()),axis=0)
                    target_train = np.concatenate((target_train,train_batch['evt_label']),axis=0)
        preds_train = torch.tensor(preds_train).float().to(device)
        target_train = torch.tensor(target_train).float().to(device)

        for i, val_batch in enumerate( val_loader ):
            if i==0:
                preds_val = infer_val(model,val_batch,device).detach().cpu().numpy()
                target_val = val_batch['evt_label']
            else:    
                preds_val = np.concatenate((preds_val,infer_val(model,val_batch,device).detach().cpu().numpy()),axis=0)  
                target_val = np.concatenate((target_val,val_batch['evt_label']),axis=0)     
        preds_val = torch.tensor(preds_val).float().to(device)
        target_val = torch.tensor(target_val).float().to(device)
        
        train_loss = loss_fn(preds_train,target_train)
        val_loss = loss_fn(preds_val,target_val)
        print(f'train_loss: {float(train_loss)} | validation_loss: {float(val_loss)}')
        return {'train_loss': float(train_loss),'validation_loss': float(val_loss)}
    
    
def train_loop(model, idxmap, device,experiment, path, config):
    opt = optim.Adam(model.parameters(), config['LR'])
    loss_fn = nn.BCELoss()
    evals = []
    best_val_loss = float('inf')
    Dataset = df.CustomDataset(idxmap)
    num_samples = Dataset.length
    num_train = int(0.80 * num_samples)
    num_val = num_samples - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(Dataset, [num_train, num_val])

    with TemporaryDirectory() as tempdir:
        best_model_params_path = path
    for epoch in range (0,config['epochs']):
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
        print(f'epoch: {epoch+1}') 
        for i, train_batch in enumerate( train_loader ):
            report = train_step(model, opt, loss_fn,train_batch ,device)
        evals.append(eval_fn(model, loss_fn,train_loader,val_loader,device) )    
        val_loss = evals[epoch]['validation_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    

    return evals, model


def get_preds(model,data_loader,device):

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate( data_loader ):
                if i==0:
                    preds = infer_val(model,batch,device).detach().cpu().numpy()
                    target = batch['evt_label']
                else:    
                    preds = np.concatenate((preds,infer_val(model,batch,device).detach().cpu().numpy()),axis=0)
                    target = np.concatenate((target,batch['evt_label']),axis=0)

    return preds,target