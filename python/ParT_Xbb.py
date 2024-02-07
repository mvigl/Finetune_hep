from comet_ml.integration.pytorch import log_model
import vector
import torch
import torch.nn as nn
vector.register_awkward()
from Finetune_hep.python.ParticleTransformer import ParticleTransformer
from Finetune_hep.python import definitions as df


class ParticleTransformerWrapper(nn.Module):
    def __init__(self,**kwargs) -> None:
        super().__init__()

        self.embed_dims = kwargs['embed_dims'][-1]
        self.for_inference = kwargs['for_inference']
        self.head_width = kwargs['head_width']
        self.head_nlayers = kwargs['head_nlayers']
        self.head_latent = kwargs['head_latent']
        self.Task = kwargs['Task']

        self.head = df.make_mlp(
                                in_features = self.embed_dims,
                                out_features=self.head_width,
                                nlayer = self.head_nlayers,
                                binary = not self.head_latent,
                                for_inference=self.for_inference
        )
        
        if self.hl_feat: 
            if self.head_latent:
                self.deepsets = df.InvariantModel(  phi=df.make_mlp(128+5,128,3,for_inference=False,binary=False),
                                                    rho=df.make_mlp(128,128,3,for_inference=self.for_inference))
            else:
                self.deepsets = df.InvariantModel(  phi=df.make_mlp(1+5,24,4,for_inference=False,binary=False),
                                                    rho=df.make_mlp(24,48,4,for_inference=self.for_inference))    
        
        kwargs['num_classes'] = None
        kwargs['fc_params'] = None
        self.input_dim = kwargs['input_dim']
        self.pair_input_dim = kwargs['pair_input_dim']
        self.head_Njets_max = kwargs['head_Njets_max']
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask,jet_mask=None):
        if self.Task == 'Xbb':    
            x_cls = self.mod(features, v=lorentz_vectors, mask=mask)
            output = self.head(x_cls)
            return output
        elif self.Task == 'Event':
            features = torch.reshape(features,(-1,self.input_dim,100))
            lorentz_vectors = torch.reshape(lorentz_vectors,(-1,self.pair_input_dim,100))
            mask = torch.reshape(mask,(-1,1,100))
            x_cls = self.mod(features, v=lorentz_vectors, mask=mask) 
            output_parT = torch.sum(torch.reshape(x_cls,(-1,self.head_Njets_max,self.embed_dims))*jet_mask,dim=1)
            output_head = self.fc(output_parT)
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
        head_nlayers=0,
        head_width=128,
        head_latent=False,
        Task='Xbb',
        head_Njets_max=5
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
        # compute the representation for each data point
        x = self.phi(x)*jet_mask

        # sum up the representations
        x = torch.sum(x, dim=1)

        # compute the output
        out = self.rho(x)

        return out
