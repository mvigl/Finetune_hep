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
# from torch.optim.lr_scheduler import ExponentialLR
# from ignite.handlers import create_lr_scheduler_with_warmup

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


def infer(model,batch,device,isXbb):
    pf_points = torch.tensor(batch['pf_points']).float().to(device)
    pf_features = torch.tensor(batch['pf_features']).float().to(device)
    pf_vectors = torch.tensor(batch['pf_vectors']).float().to(device)
    pf_mask = torch.tensor(batch['pf_mask']).float().to(device)
    if isXbb: preds = model(pf_points,pf_features,pf_vectors,pf_mask)
    else: 
        jet_mask = torch.tensor(batch['jet_mask']).float().to(device)
        preds = model(pf_points,pf_features,pf_vectors,pf_mask,jet_mask)
    return preds

def infer_val(model,batch,device,isXbb=False):
    with torch.no_grad():
        return infer(model,batch,device,isXbb)
    

def train_step(model,opt,loss_fn,train_batch,device,scheduler,isXbb=False):
    model.train()
    opt.zero_grad()
    preds = infer(model,train_batch,device,isXbb)
    target = torch.tensor(train_batch['label']).float().to(device)
    loss = loss_fn(preds,target)
    loss.backward()
    opt.step()
    scheduler.step()
    return {'loss': float(loss)}

def eval_fn(model,loss_fn,train_loader,val_loader,device,subset,build_features,isXbb=False):
    with torch.no_grad():
        model.eval()
        for i, train_batch in enumerate( train_loader ): 
            if (subset and i > 1): break    
            if (i > 100): break
            train_batch['X_jet']=train_batch['X_jet'].numpy()
            train_batch['X_pfo']=train_batch['X_pfo'].numpy()
            train_batch['X_label']=train_batch['X_label'].numpy()
            train_batch['labels']=train_batch['labels'].numpy()
            if not isXbb: train_batch['jet_mask']=train_batch['jet_mask'].numpy()
            train_batch = build_features(train_batch)
            if not isXbb: train_batch['pf_mask'][:,:,:,:2] += np.abs(train_batch['jet_mask'][:,:,np.newaxis]-1)
            if i==0:
                preds_train = infer_val(model,train_batch,device,isXbb).detach().cpu().numpy()
                target_train = train_batch['label']
            else:    
                preds_train = np.concatenate((preds_train,infer_val(model,train_batch,device,isXbb).detach().cpu().numpy()),axis=0)
                target_train = np.concatenate((target_train,train_batch['label']),axis=0)        
        preds_train = torch.tensor(preds_train).float().to(device)
        target_train = torch.tensor(target_train).float().to(device)

        for i, val_batch in enumerate( val_loader ):
            if (subset and i > 1): break 
            val_batch['X_jet']=val_batch['X_jet'].numpy()
            val_batch['X_pfo']=val_batch['X_pfo'].numpy()
            val_batch['X_label']=val_batch['X_label'].numpy()
            val_batch['labels']=val_batch['labels'].numpy()
            if not isXbb: val_batch['jet_mask']=val_batch['jet_mask'].numpy() 
            val_batch = build_features(val_batch)
            if not isXbb: val_batch['pf_mask'][:,:,:,:2] += np.abs(val_batch['jet_mask'][:,:,np.newaxis]-1)
            if i==0:
                preds_val = infer_val(model,val_batch,device,isXbb).detach().cpu().numpy()
                target_val = val_batch['label']
            else:    
                preds_val = np.concatenate((preds_val,infer_val(model,val_batch,device,isXbb).detach().cpu().numpy()),axis=0)  
                target_val = np.concatenate((target_val,val_batch['label']),axis=0)        
        preds_val = torch.tensor(preds_val).float().to(device)
        target_val = torch.tensor(target_val).float().to(device)
        
        train_loss = loss_fn(preds_train,target_train)
        val_loss = loss_fn(preds_val,target_val)
        print(f'train_loss: {float(train_loss)} | validation_loss: {float(val_loss)}')
        return {'train_loss': float(train_loss),'validation_loss': float(val_loss)}
    
def get_scheduler(epochs,njets_train,batch_size,warmup_steps,opt):
    total_steps = epochs * int(math.ceil(njets_train/batch_size))
    warmup_steps = warmup_steps
    flat_steps = total_steps * 0.7 - 1
    min_factor = 0.00
    def lr_fn(step_num):
        if step_num > total_steps:
            raise ValueError(
                "Tried to step {} times. The specified number of total steps is {}".format(
                    step_num + 1, total_steps))
        if step_num < warmup_steps:
            return 1. * step_num / warmup_steps
        if step_num <= flat_steps:
            return 1.0
        pct = (step_num - flat_steps) / (total_steps - flat_steps)
        return max(min_factor, 1 - pct)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn, last_epoch=-1)
    scheduler._update_per_step = True  # mark it to update the lr every step, instead of every epoch
    return scheduler      
    
def train_loop(model, idxmap,integer_file_map,idxmap_val,integer_file_map_val, device,experiment, path,subset, config):
    evals = []
    best_val_loss = float('inf')
    if config['Xbb']:
        print('Xbb task')
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([21.39]).to(device))
        Dataset = df.Xbb_CustomDataset(idxmap,integer_file_map)
        Dataset_val = df.Xbb_CustomDataset(idxmap_val,integer_file_map_val)
        build_features = df.build_features_and_labels_Xbb
    else:    
        print('Evt task')
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([13.76]).to(device))
        Dataset = df.CustomDataset(idxmap,integer_file_map)
        Dataset_val = df.CustomDataset(idxmap_val,integer_file_map_val)
        build_features = df.build_features_and_labels
    num_samples = Dataset.length
    # num_train = int(0.80 * num_samples)
    # num_val = num_samples - num_train
    # train_dataset, val_dataset = torch.utils.data.random_split(Dataset, [num_train, num_val])
    val_loader = DataLoader(Dataset_val, batch_size=config['batch_size'], shuffle=True,num_workers=12)

    base_opt = torch.optim.RAdam(model.parameters(), lr=config['LR'], betas=(0.95, 0.999),eps=1e-05) # Any optimizer
    opt = Lookahead(base_opt, k=6, alpha=0.5)
    scheduler = get_scheduler(config['epochs'],num_samples,config['batch_size'],50,opt)

    best_model_params_path = path
    for epoch in range (0,config['epochs']):
        train_loader = DataLoader(Dataset, batch_size=config['batch_size'], shuffle=True,num_workers=12)
        print('Epoch:', epoch,'LR:',opt.param_groups[0]["lr"])
        for i, train_batch in enumerate( train_loader ):
            if (subset and i > 1): break
            train_batch['X_jet']=train_batch['X_jet'].numpy()
            train_batch['X_pfo']=train_batch['X_pfo'].numpy()
            train_batch['X_label']=train_batch['X_label'].numpy()
            train_batch['labels']=train_batch['labels'].numpy()
            if not config['Xbb']: train_batch['jet_mask']=train_batch['jet_mask'].numpy()
            train_batch = build_features(train_batch)
            if not config['Xbb']: train_batch['pf_mask'][:,:,:,:2] += np.abs(train_batch['jet_mask'][:,:,np.newaxis]-1)
            report = train_step(model, opt, loss_fn,train_batch ,device,scheduler,config['Xbb'])
        evals.append(eval_fn(model, loss_fn,train_loader,val_loader,device,subset,build_features,config['Xbb']) )    
        val_loss = evals[epoch]['validation_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    

    return evals, model


def get_preds(model,data_loader,device,subset,build_features,isXbb=False):

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate( data_loader ):  
                if (i % 500) == 0: print('batch : ', i)
                batch['X_jet']=batch['X_jet'].numpy()
                batch['X_pfo']=batch['X_pfo'].numpy()
                batch['X_label']=batch['X_label'].numpy()
                batch['labels']=batch['labels'].numpy()
                if not isXbb: batch['jet_mask']=batch['jet_mask'].numpy()
                batch = build_features(batch)  
                if not isXbb: batch['pf_mask'][:,:,:,:2] += np.abs(batch['jet_mask'][:,:,np.newaxis]-1)
                if i==0:
                    preds = infer_val(model,batch,device,isXbb).detach().cpu().numpy()
                    target = batch['label']
                else:    
                    preds = np.concatenate((preds,infer_val(model,batch,device,isXbb).detach().cpu().numpy()),axis=0)
                    target = np.concatenate((target,batch['label']),axis=0)
                if (subset and i>5): break    

    return preds,target

def get_Xbb_preds(model,filelist,device,subset,Xbb=False):

    with torch.no_grad():
        model.eval()
        with open(filelist) as f:
            i=-1
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                with h5py.File(filename, 'r') as Data:
                    if len(Data['X_label']) > 3000: size = 512
                    else : 
                        size = len(Data['X_label'])/10
                        if len(Data['X_label']) == 0: 
                            print('no data')
                            continue
                    i+=1    
                    batches = np.array_split(np.arange(len(Data['X_label'])),int(len(Data['X_label'])/size))
                    for j in range(len(batches)):
                        data = {}
                        if Xbb:
                            build_features = df.build_features_and_labels_Xbb
                            data['X_jet'] = Data['X_jet'][batches[j]].reshape(-1,6)
                            data['X_pfo'] = Data['X_pfo'][batches[j]].reshape(-1,100, 15)
                            data['X_label'] = Data['X_label'][batches[j]].reshape(-1,6)
                            data = build_features(data)
                        else:   
                            build_features = df.build_features_and_labels
                            data['X_jet'] = Data['X_jet'][batches[j]]
                            data['X_pfo'] = Data['X_pfo'][batches[j]]
                            data['labels'] = Data['labels'][batches[j]]
                            data = build_features(data) 
                        if (i ==0 and j==0):
                            preds = infer_val(model,data,device).detach().cpu().numpy()
                            target = data['label']
                        else:
                            preds = np.concatenate((preds,infer_val(model,data,device).detach().cpu().numpy()),axis=0)
                            target = np.concatenate((target,data['label']),axis=0)
                if (subset and i>5): break
    return preds,target
