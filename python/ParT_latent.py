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

class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


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
        fc_params=None,
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
    )
    cfg.update(**kwargs)

    model = ParticleTransformerWrapper(**cfg)
    return model

def infer(model,data,train_batch,device):
    N = train_batch
    pf_points = torch.tensor(data['pf_points'][N]).float().to(device)
    pf_features = torch.tensor(data['pf_features'][N]).float().to(device)
    pf_vectors = torch.tensor(data['pf_vectors'][N]).float().to(device)
    pf_mask = torch.tensor(data['pf_mask'][N]).float().to(device)
    preds = model(pf_points,pf_features,pf_vectors,pf_mask)
    return preds.reshape((-1,128))

def infer_val(model,data,train_batch,device):
    with torch.no_grad():
        return infer(model,data,train_batch,device)

def get_preds(model,data,evts,device):

    ix = np.array_split(np.arange(len(evts)),int(len(evts)/512))
    for i in range(len(ix)):
        preds_i = infer_val(model,data,ix[i],device).reshape((-1,128))
        if i==0:
            yi_model = preds_i.detach().cpu().numpy()
        else:    
            yi_model = np.concatenate((yi_model,preds_i.detach().cpu().numpy()),axis=0)
    return yi_model        