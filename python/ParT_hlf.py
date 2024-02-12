from Finetune_hep.python import helpers
import numpy as np
import vector
import torch
import torch.nn as nn
vector.register_awkward()
from Finetune_hep.python.ParticleTransformer import ParticleTransformer
    


class ParticleTransformerWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.embed_dims = kwargs['embed_dims'][-1]
        self.for_inference = kwargs['for_inference']
        self.head_width = kwargs['head_width']
        self.head_nlayers = kwargs['head_nlayers']
        self.head_latent = kwargs['head_latent']
        self.Task = kwargs['Task']
        self.hlf_dim = kwargs['hlf_dim']

        if self.head_latent:
            self.head = InvariantModel( phi = helpers.make_mlp(
                                            in_features = self.embed_dims+self.hlf_dim,
                                            out_features = self.head_width,
                                            nlayer = self.head_nlayers,
                                            binary = False,
                                            for_inference=False
                                            ),
                                        rho = helpers.make_mlp(
                                            in_features = self.head_width,
                                            out_features= self.head_width,
                                            nlayer = self.head_nlayers,
                                            binary = True,
                                            for_inference=self.for_inference
                                            )    
            )    
        else:
            self.Xbb = helpers.make_mlp(
                                in_features = self.embed_dims,
                                out_features = self.head_width,
                                nlayer = 0,
                                binary = True,
                                for_inference=False
            )
            self.head = InvariantModel( phi= helpers.make_mlp(
                                            in_features = 1+self.hlf_dim,
                                            out_features = self.head_width,
                                            nlayer = self.head_nlayers,
                                            binary = False,
                                            for_inference=False
                                            ),
                                        rho= helpers.make_mlp(
                                            in_features = self.head_width,
                                            out_features= self.head_width,
                                            nlayer = self.head_nlayers,
                                            binary = True,
                                            for_inference=self.for_inference
                                            )    
                        )   
        
        kwargs['num_classes'] = None
        kwargs['fc_params'] = None
        self.input_dim = kwargs['input_dim']
        self.pair_input_dim = kwargs['pair_input_dim']
        self.Nconst_max = kwargs['Nconst_max']
        self.head_Njets_max = kwargs['head_Njets_max']
        self.save_representaions = kwargs['save_representaions']
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask,jet_mask,hl_feats):
        features = torch.reshape(features,(-1,self.input_dim,self.Nconst_max))
        lorentz_vectors = torch.reshape(lorentz_vectors,(-1,self.pair_input_dim,self.Nconst_max))
        mask = torch.reshape(mask,(-1,1,self.Nconst_max))
        x_cls = self.mod(features, v=lorentz_vectors, mask=mask) 
        if self.head_latent:
            output_parT = torch.cat( ( torch.reshape(x_cls,(-1,self.head_Njets_max,self.embed_dims)) , hl_feats ) ,axis=-1 )
        else:
            output_Xbb = self.Xbb(torch.reshape(x_cls,(-1,self.head_Njets_max,self.embed_dims)))
            output_parT = torch.cat( ( output_Xbb, hl_feats ) ,axis=-1 )
        if self.save_representaions: return output_parT    
        output_head = self.head(output_parT,jet_mask)
        return output_head

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
        Nconst_max=data_config['inputs']['pf_features']['length'],
        head_nlayers=data_config['head']['nlayers'],
        head_width=data_config['head']['width'],
        head_latent=data_config['head']['latent'],
        Task=data_config['Task'],
        head_Njets_max=data_config['head']['Njets_max'],
        hlf_dim=len(data_config['inputs']['hlf']['vars']),
        save_representaions=data_config['save_representaions'],
    )
    cfg.update(**kwargs)

    model = ParticleTransformerWrapper(**cfg)
    return model


class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x,jet_mask):
        x = self.phi(x)*jet_mask
        x = torch.sum(x, dim=1)
        out = self.rho(x)

        return out