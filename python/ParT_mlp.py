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


def get_preds(model,data,evts,device):

    ix = np.array_split(np.arange(len(evts)),int(len(evts)/512))
    for i in range(len(ix)):
        preds_i = infer_val(model,data,ix[i],device).reshape(len(ix[i]))
        if i==0:
            yi_model = preds_i.detach().cpu().numpy()
        else:    
            yi_model = np.concatenate((yi_model,preds_i.detach().cpu().numpy()),axis=0)
    return yi_model

def infer(model,data,train_batch,device):
    N = train_batch
    pf_points = torch.tensor(data['pf_points'][N]).float().to(device)
    pf_features = torch.tensor(data['pf_features'][N]).float().to(device)
    pf_vectors = torch.tensor(data['pf_vectors'][N]).float().to(device)
    pf_mask = torch.tensor(data['pf_mask'][N]).float().to(device)
    preds = model(pf_points,pf_features,pf_vectors,pf_mask)
    return preds.reshape(len(train_batch))

def infer_val(model,data,train_batch,device):
    with torch.no_grad():
        return infer(model,data,train_batch,device)
    

def train_step(model,data,labels,opt,loss_fn,train_batch,device):
    model.train()
    preds = infer(model,data,train_batch,device)
    target = torch.tensor(labels[train_batch]).float().to(device)
    loss = loss_fn(preds,target)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model,labels, loss_fn,data,train_set,validtion_set,device):
    with torch.no_grad():
        model.eval()

        ix_train = np.array_split(train_set,int(len(train_set)/512))

        for i in range(len(ix_train)):
            preds_train_i = infer_val(model,data,ix_train[i],device).reshape(len(ix_train[i]))
            if i==0:
                preds_train = preds_train_i.detach().cpu().numpy()
            else:    
                preds_train = np.concatenate((preds_train,preds_train_i.detach().cpu().numpy()),axis=0)
        preds_train = torch.tensor(preds_train).float().to(device)

        ix_val = np.array_split(validtion_set,int(len(validtion_set)/512))

        for i in range(len(ix_val)):
            preds_val_i = infer_val(model,data,ix_val[i],device).reshape(len(ix_val[i]))
            if i==0:
                preds_val = preds_val_i.detach().cpu().numpy()
            else:    
                preds_val = np.concatenate((preds_val,preds_val_i.detach().cpu().numpy()),axis=0)       
        preds_val = torch.tensor(preds_val).float().to(device)
        
        target_train = torch.tensor(labels[train_set]).float().to(device)
        train_loss = loss_fn(preds_train,target_train)
        target_val = torch.tensor(labels[validtion_set]).float().to(device)
        test_loss = loss_fn(preds_val,target_val)
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    
    
def train_loop(model, data, labels, device,experiment, path, config):
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
        best_model_params_path = path
    for epoch in range (0,config['epochs']):
        train_batches = DataLoader(
            ix2,  batch_size=config['batch_size'], sampler=SubsetRandomSampler(ix[:N_tr])
        )
        print(f'epoch: {epoch+1}') 
        for i, train_batch in enumerate( train_batches ):
            train_batch = train_batch.numpy()
            report = train_step(model, data, labels, opt, loss_fn,train_batch ,device)
        evals.append(eval_fn(model,labels, loss_fn,data,train_set,validtion_set,device) )    
        val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    

    return evals, model