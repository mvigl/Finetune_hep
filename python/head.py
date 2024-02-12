import numpy as np
import vector
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Finetune_hep.python import helpers
from sklearn.preprocessing import StandardScaler
import h5py
import pickle
import torch.optim as optim
vector.register_awkward()

def fit_transform_without_zeros(data,jet_mask,scaler):
    non_zero_data = data.reshape(-1,len(helpers.jVars))[jet_mask.reshape(-1).astype(bool)]
    scaled_non_zero_data = scaler.fit_transform(non_zero_data)
    scaled_data = np.zeros_like(data.reshape(-1,len(helpers.jVars)), dtype=float)
    scaled_data[jet_mask.reshape(-1).astype(bool)] = scaled_non_zero_data
    return scaled_data.reshape(-1,5,len(helpers.jVars)),scaler

def transform_without_zeros(data,jet_mask,scaler):
    non_zero_data = data.reshape(-1,len(helpers.jVars))[jet_mask.reshape(-1).astype(bool)]
    scaled_non_zero_data = scaler.transform(non_zero_data)
    scaled_data = np.zeros_like(data.reshape(-1,len(helpers.jVars)), dtype=float)
    scaled_data[jet_mask.reshape(-1).astype(bool)] = scaled_non_zero_data
    return scaled_data.reshape(-1,5,len(helpers.jVars))

class CustomDataset(Dataset):
    def __init__(self, filelist,device,scaler_path,Xbb_scores_path,use_hlf=True,test=False,subset_batches=1,latent=False):
        self.device = device
        self.x=[]
        self.y=[]
        self.jet_mask=[]
        i=0
        subset_offset=0
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                with h5py.File(filename, 'r') as Data:
                    subset_offset = int(len(Data['X_jet'])*subset_batches)
                    if i ==0:
                        hlf = Data['X_jet'][:subset_offset]
                        target = Data['labels'][:subset_offset] 
                        jet_mask = Data['jet_mask'][:subset_offset]
                    else:
                        hlf = np.concatenate((hlf,Data['X_jet'][:subset_offset]),axis=0)
                        target = np.concatenate((target,Data['labels'][:subset_offset]),axis=0)
                        jet_mask = np.concatenate((jet_mask,Data['jet_mask'][:subset_offset]),axis=0)
                    i+=1    
        self.scaler = StandardScaler()
        hlf[:,:,helpers.jVars.index('fj_pt')] = helpers.log(hlf[:,:,helpers.jVars.index('fj_pt')])
        hlf[:,:,helpers.jVars.index('fj_mass')] = helpers.log(hlf[:,:,helpers.jVars.index('fj_mass')])
        hlf[:,:,helpers.jVars.index('fj_sdmass')] = helpers.log(hlf[:,:,helpers.jVars.index('fj_sdmass')])
        if Xbb_scores_path != '': 
            subset_offset=0
            i=0
            with open(Xbb_scores_path) as f:
                for line in f:
                    filename = line.strip()
                    print('loading Xbb scores from : ',filename)
                    with h5py.File(filename, 'r') as Xbb_scores:
                        subset_offset = int(len(Xbb_scores['Xbb_score'])*subset_batches)
                        if i ==0:
                            Xbb = Xbb_scores['Xbb_score'][:subset_offset]
                        else:
                            Xbb = np.concatenate((Xbb,Xbb_scores['Xbb_score'][:subset_offset]),axis=0)
                        i+=1    
            if latent == False: hlf[:,:,helpers.jVars.index('fj_doubleb')] = np.nan_to_num(Xbb.reshape(-1,5))  
                    
        if latent == False: Xbb = hlf[:,:,helpers.jVars.index('fj_doubleb')].reshape(-1,5,1)
        
        hlfeats = [helpers.jVars.index('fj_pt'),helpers.jVars.index('fj_eta'),helpers.jVars.index('fj_phi'),helpers.jVars.index('fj_mass'),helpers.jVars.index('fj_sdmass')]
        if use_hlf: data = np.concatenate((Xbb,hlf[:,:,hlfeats]),axis=-1)      
        else: data = Xbb

        if scaler_path !='' : 
            if (test == False): 
                X_norm,self.scaler = fit_transform_without_zeros(data,jet_mask,self.scaler)
                self.x = torch.from_numpy(X_norm).float().to(device)
                with open(scaler_path,'wb') as f:
                    pickle.dump(self.scaler, f)
            else:         
                with open(scaler_path,'rb') as f:
                    self.scaler = pickle.load(f)
                X_norm = transform_without_zeros(data,jet_mask,self.scaler)
                self.x = torch.from_numpy(X_norm).float().to(device)
        else:
            self.x = torch.from_numpy(data).float().to(device)    
        self.y = torch.from_numpy(target.reshape(-1,1)).float().to(device)
        self.jet_mask = torch.from_numpy(jet_mask.reshape(-1,5,1)).float().to(device)    
        self.length = len(target)
        print('N data : ',self.length)
        
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx],self.jet_mask[idx]
    

def train_step(model,data,target,jet_mask,opt,loss_fn):
    model.train()
    print(jet_mask.shape)
    print(data.shape)
    if 'phi.0.weight' in model.state_dict().keys(): preds = model(data,jet_mask)    
    else: preds =  model(data)
    loss = loss_fn(preds,target)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model, loss_fn,train_loader,val_loader,device):
    with torch.no_grad():
        model.eval()
        for i, train_batch in enumerate( train_loader ): 
            if i==0:
                data, target, jet_mask = train_batch
                data = data.cpu().numpy()
                target = target.cpu().numpy()
                jet_mask = jet_mask.cpu().numpy()
            else: 
                data = np.concatenate((data,train_batch[0].cpu().numpy()),axis=0)
                target = np.concatenate((target,train_batch[1].cpu().numpy()),axis=0)
                jet_mask = np.concatenate((jet_mask,train_batch[2].cpu().numpy()),axis=0)
            if (i > 100): break 
        for i, val_batch in enumerate( val_loader ):
            if i==0:
                data_val, target_val, jet_mask_val = val_batch
                data_val = data_val.cpu().numpy()
                target_val = target_val.cpu().numpy()
                jet_mask_val = jet_mask_val.cpu().numpy()
            else: 
                data_val = np.concatenate((data_val,val_batch[0].cpu().numpy()),axis=0)
                target_val = np.concatenate((target_val,val_batch[1].cpu().numpy()),axis=0)   
                jet_mask_val = np.concatenate((jet_mask_val,val_batch[2].cpu().numpy()),axis=0)          

        if 'phi.0.weight' in model.state_dict().keys(): 
            train_loss = loss_fn(model( torch.from_numpy(data).float().to(device),torch.from_numpy(jet_mask).float().to(device)).reshape(len(data)),torch.from_numpy(target.reshape(-1)).float().to(device))
            test_loss = loss_fn(model( torch.from_numpy(data_val).float().to(device),torch.from_numpy(jet_mask_val).float().to(device)).reshape(len(data_val)),torch.from_numpy(target_val.reshape(-1)).float().to(device))
        else:
            train_loss = loss_fn(model( torch.from_numpy(data).float().to(device) ).reshape(len(data)),torch.from_numpy(target.reshape(-1)).float().to(device))
            test_loss = loss_fn(model( torch.from_numpy(data_val).float().to(device) ).reshape(len(data_val)),torch.from_numpy(target_val.reshape(-1)).float().to(device))    
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    

def train_loop(model,config):
    opt = optim.Adam(model.parameters(), config['LR'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([13.76]).to(config['device']))
    evals = []
    best_val_loss = float('inf')
    Dataset = CustomDataset(config['filelist'],config['device'],config['scaler_path'],config['Xbb_scores_path'],use_hlf=config['use_hlf'],test=False,subset_batches=config['subset'])
    Dataset_val = CustomDataset(config['filelist_val'],config['device'],config['scaler_path'],config['Xbb_scores_path_val'],test=True,subset_batches=config['subset_val'])

    best_model_params_path = config['out_model_path']   
    val_loader = DataLoader(Dataset_val, batch_size=config['batch_size'], shuffle=True)
    for epoch in range (0,config['epochs']):
        print(f'epoch: {epoch+1}') 
        train_loader = DataLoader(Dataset, batch_size=config['batch_size'], shuffle=True)
        for i, train_batch in enumerate( train_loader ):
            data, target, jet_mask = train_batch
            report = train_step(model, data, target,jet_mask, opt, loss_fn )
        evals.append(eval_fn(model, loss_fn,train_loader,val_loader,config['device']) )         
        val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
        config['experiment'].log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    

    return evals, model