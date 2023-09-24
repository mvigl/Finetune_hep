from Finetune_hep.python import definitions as df
import numpy as np
import math
import vector
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
vector.register_awkward()
from Finetune_hep.python.ParticleTransformer import ParticleTransformer
import h5py
from torch_optimizer import Lookahead


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

class ParticleTransformerWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        in_dim = kwargs['embed_dims'][-1]
        fc_params = kwargs.pop('fc_params')
        num_classes = kwargs.pop('num_classes')
        self.for_inference = kwargs['for_inference']

        fcs = []
        self.fc = make_mlp(in_dim,out_features=128,nlayer = 3,for_inference=self.for_inference)

        kwargs['num_classes'] = None
        kwargs['fc_params'] = None
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask,jet_mask):
        features = torch.reshape(features,(-1,17,100))
        lorentz_vectors = torch.reshape(lorentz_vectors,(-1,4,100))
        mask = torch.reshape(mask,(-1,1,100))
        x_cls = self.mod(features, v=lorentz_vectors, mask=mask) 
        output_parT = torch.sum(torch.reshape(x_cls,(-1,5,128))*jet_mask,dim=1)
        #output = self.fc(output_parT)
        return output_parT

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

    