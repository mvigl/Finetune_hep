import numpy as np
import vector
import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Finetune_hep.python import helpers
from torch_optimizer import Lookahead
vector.register_awkward()


def infer(model,batch,device):
    pf_points = torch.tensor(batch['pf_points']).float().to(device)
    pf_features = torch.tensor(batch['pf_features']).float().to(device)
    pf_vectors = torch.tensor(batch['pf_vectors']).float().to(device)
    pf_mask = torch.tensor(batch['pf_mask']).float().to(device)
    if model.Task == 'Xbb': preds = model(pf_points,pf_features,pf_vectors,pf_mask)
    else: 
        hl_feats = torch.tensor(batch['hl_feats']).float().to(device)
        jet_mask = torch.tensor(batch['jet_mask']).float().to(device)
        preds = model(pf_points,pf_features,pf_vectors,pf_mask,jet_mask,hl_feats)
    return preds

def infer_val(model,batch,device):
    with torch.no_grad():
        return infer(model,batch,device)
    

def train_step(model,opt,loss_fn,train_batch,device,scheduler):
    model.train()
    opt.zero_grad()
    preds = infer(model,train_batch,device)
    target = torch.tensor(train_batch['label']).float().to(device)
    loss = loss_fn(preds,target)
    loss.backward()
    opt.step()
    if scheduler!=False: scheduler.step()
    return {'loss': float(loss)}

def eval_fn(model,loss_fn,train_loader,val_loader,device,build_features):
    with torch.no_grad():
        model.eval()
        for i, train_batch in enumerate( train_loader ):  
            if (i > 20): break
            train_batch['X_jet']=train_batch['X_jet'].numpy()
            train_batch['X_pfo']=train_batch['X_pfo'].numpy()
            train_batch['labels']=train_batch['labels'].numpy()
            if model.Task == 'Event': train_batch['jet_mask']=train_batch['jet_mask'].numpy()
            train_batch = build_features(train_batch)
            if i==0:
                preds_train = infer_val(model,train_batch,device).detach().cpu().numpy()
                target_train = train_batch['label']
            else:    
                preds_train = np.concatenate((preds_train,infer_val(model,train_batch,device).detach().cpu().numpy()),axis=0)
                target_train = np.concatenate((target_train,train_batch['label']),axis=0)        
        preds_train = torch.tensor(preds_train).float().to(device)
        target_train = torch.tensor(target_train).float().to(device)

        for i, val_batch in enumerate( val_loader ):
            val_batch['X_jet']=val_batch['X_jet'].numpy()
            val_batch['X_pfo']=val_batch['X_pfo'].numpy()
            val_batch['labels']=val_batch['labels'].numpy()
            if model.Task == 'Event': val_batch['jet_mask']=val_batch['jet_mask'].numpy() 
            val_batch = build_features(val_batch)
            if i==0:
                preds_val = infer_val(model,val_batch,device).detach().cpu().numpy()
                target_val = val_batch['label']
            else:    
                preds_val = np.concatenate((preds_val,infer_val(model,val_batch,device).detach().cpu().numpy()),axis=0)  
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
    
def train_loop(model, config):
    evals = []
    best_val_loss = float('inf')
    if model.Task == 'Xbb':
        print('Xbb task')
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([21.39]).to(config['device']))
        Dataset = helpers.Xbb_CustomDataset(config['filelist'],config['subset'],train=True,device="cpu")
        Dataset_val = helpers.Xbb_CustomDataset(config['filelist_val'],config['subset'],train=False,device="cpu")
        build_features = helpers.build_features_and_labels_Xbb
    else:    
        print('Evt task')
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([13.76]).to(config['device']))
        Dataset = helpers.CustomDataset(config['filelist'],config['subset'],train=True,device="cpu")
        Dataset_val = helpers.CustomDataset(config['filelist'],config['subset'],train=False,device="cpu")
        build_features = helpers.build_features_and_labels
    num_samples = Dataset.length
    val_loader = DataLoader(Dataset_val, batch_size=config['batch_size'], shuffle=True,num_workers=config['num_workers'])

    base_opt = torch.optim.RAdam(model.parameters(), lr=config['LR'], betas=(0.95, 0.999),eps=1e-05) # Any optimizer
    opt = Lookahead(base_opt, k=6, alpha=0.5)
    scheduler = get_scheduler(config['epochs'],num_samples,config['batch_size'],5,opt)

    best_model_params_path = config['out_model_path']

    for epoch in range (0,config['epochs']):
        train_loader = DataLoader(Dataset, batch_size=config['batch_size'], shuffle=True,num_workers=config['num_workers'])
        print('Epoch:', epoch+config["start_epoch"],'LR:',opt.param_groups[0]["lr"])
        for i, train_batch in enumerate( train_loader ):
            train_batch['X_jet']=train_batch['X_jet'].numpy()
            train_batch['X_pfo']=train_batch['X_pfo'].numpy()
            train_batch['labels']=train_batch['labels'].numpy()
            if model.Task == 'Event': train_batch['jet_mask']=train_batch['jet_mask'].numpy()
            train_batch = build_features(train_batch)
            report = train_step(model, opt, loss_fn,train_batch ,config['device'],scheduler)
        evals.append(eval_fn(model, loss_fn,train_loader,val_loader,config['device'],build_features))    
        val_loss = evals[epoch]['validation_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
        torch.save(model.state_dict(), f'{config["out_model_path"].replace(".pt", "")}_epoch_{epoch+config["start_epoch"]}_Val_loss_{val_loss}.pt')
        config['experiment'].log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, step=(epoch+config["start_epoch"]),epoch=(epoch+config["start_epoch"]))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    

    return evals, model