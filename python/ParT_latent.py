import numpy as np
import vector
import torch

vector.register_awkward()
from Finetune_hep.python.ParticleTransformer import ParticleTransformer

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
    
def infer(model,batch,device):
    pf_points = torch.tensor(batch['pf_points']).float().to(device)
    pf_features = torch.tensor(batch['pf_features']).float().to(device)
    pf_vectors = torch.tensor(batch['pf_vectors']).float().to(device)
    pf_mask = torch.tensor(batch['pf_mask']).float().to(device)
    preds = model(pf_points,pf_features,pf_vectors,pf_mask)
    return preds.reshape((-1,128))

def infer_val(model,batch,device):
    with torch.no_grad():
        return infer(model,batch,device)    


def get_preds(model,data_loader,device):
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate( data_loader ):
                if i==0:
                    preds = infer_val(model,batch,device).detach().cpu().numpy()
                    target = batch['label']
                else:    
                    preds = np.concatenate((preds,infer_val(model,batch,device).detach().cpu().numpy()),axis=0)
                    target = np.concatenate((target,batch['label']),axis=0)

    return preds,target