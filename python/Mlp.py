import numpy as np
import vector
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tempfile import TemporaryDirectory
import torch.optim as optim
import h5py
import pickle
vector.register_awkward()


labelVars = [f'label_{v}' for v in ['QCD_b','QCD_bb','QCD_c','QCD_cc','QCD_others','H_bb']]       
jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]
pVars = [f'pfcand_{v}' for v in ['ptrel','erel','etarel','phirel','dxy','dxysig','dz','dzsig','deltaR','charge','isChargedHad','isNeutralHad','isGamma','isEl','isMu']]        

def log(data):
    ma_data = np.ma.masked_equal(data,0)
    result = np.log(ma_data)
    return result.filled(fill_value=0)  

def make_mlp(in_features,out_features,nlayer,for_inference=False,binary=True):
    layers = []
    for i in range(nlayer):
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    if binary: layers.append(torch.nn.Linear(in_features, 1))
    if for_inference: layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers)
    return model

def fit_transform_without_zeros(data,jet_mask,scaler):
    non_zero_data = data.reshape(-1,6)[jet_mask.reshape(-1).astype(bool)]
    scaled_non_zero_data = scaler.fit_transform(non_zero_data)
    scaled_data = np.zeros_like(data.reshape(-1,6), dtype=float)
    scaled_data[jet_mask.reshape(-1).astype(bool)] = scaled_non_zero_data
    return scaled_data.reshape(-1,5,6),scaler

def transform_without_zeros(data,jet_mask,scaler):
    non_zero_data = data.reshape(-1,6)[jet_mask.reshape(-1).astype(bool)]
    scaled_non_zero_data = scaler.transform(non_zero_data)
    scaled_data = np.zeros_like(data.reshape(-1,6), dtype=float)
    scaled_data[jet_mask.reshape(-1).astype(bool)] = scaled_non_zero_data
    return scaled_data.reshape(-1,5,6)

class CustomDataset(Dataset):
    def __init__(self, filelist,device,scaler_path,Xbb_scores_path,test=False):
        self.device = device
        self.x=[]
        self.y=[]
        self.jet_mask=[]
        i=0
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                with h5py.File(filename, 'r') as Data:
                    if i ==0:
                        data = Data['X_jet'][:]
                        target = Data['labels'][:] 
                        jet_mask = Data['jet_mask'][:]
                    else:
                        data = np.concatenate((data,Data['X_jet'][:]),axis=0)
                        target = np.concatenate((target,Data['labels'][:]),axis=0)
                        jet_mask = np.concatenate((jet_mask,Data['jet_mask'][:]),axis=0)
                    i+=1    
        self.scaler = StandardScaler() # this is super useful a scikit learn function
        data[:,:,jVars.index('fj_pt')] = log(data[:,:,jVars.index('fj_pt')])
        data[:,:,jVars.index('fj_mass')] = log(data[:,:,jVars.index('fj_mass')])
        data[:,:,jVars.index('fj_sdmass')] = log(data[:,:,jVars.index('fj_sdmass')])
        if Xbb_scores_path != 'no': 
            print('loading Xbb scores from : ',Xbb_scores_path)
            with h5py.File(Xbb_scores_path, 'r') as Xbb_scores:
                data[:,:,jVars.index('fj_doubleb')] = Xbb_scores['Xbb'][:]
        if scaler_path !='no' : 
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
        self.jet_mask = torch.from_numpy(jet_mask).float().to(device)    
        self.length = len(target)
        print('N data : ',self.length)
        
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx],self.jet_mask[idx]
             
class CustomDataset_XbbOnly(Dataset):
    def __init__(self, filelist,device,scaler_path,Xbb_scores_path,test=False):
        self.device = device
        self.x=[]
        self.y=[]
        self.jet_mask=[]
        i=0
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                with h5py.File(filename, 'r') as Data:
                    if i ==0:
                        target = Data['labels'][:] 
                        jet_mask = Data['jet_mask'][:]
                    else:
                        target = np.concatenate((target,Data['labels'][:]),axis=0)
                        jet_mask = np.concatenate((jet_mask,Data['jet_mask'][:]),axis=0)
                    i+=1    
        print('loading Xbb scores from : ',Xbb_scores_path)
        with h5py.File(Xbb_scores_path, 'r') as Xbb_scores:
            data = np.nan_to_num(Xbb_scores['Xbb'][:].reshape((-1,5,1)))
        self.x = torch.from_numpy(data).float().to(device)    
        self.y = torch.from_numpy(target.reshape(-1,1)).float().to(device)
        self.jet_mask = torch.from_numpy(jet_mask).float().to(device)    
        self.length = len(target)
        print('N data : ',self.length)
        
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx],self.jet_mask[idx]

class CustomDataset_Latent(Dataset):
    def __init__(self, filelist,device,scaler_path,Xbb_scores_path,test=False):
        self.device = device
        self.x=[]
        self.y=[]
        self.jet_mask=[]
        i=0
        print('loading Xbb scores from : ',Xbb_scores_path)
        with h5py.File(Xbb_scores_path, 'r') as latent:
            target = latent['evt_label'][:]
            jet_mask = latent['jet_mask'][:]
            data = np.sum(np.nan_to_num(latent['evt_score'][:])*jet_mask[:,:,np.newaxis],axis=1)
        self.x = torch.from_numpy(data).float().to(device)    
        self.y = torch.from_numpy(target.reshape(-1,1)).float().to(device)
        self.jet_mask = torch.from_numpy(jet_mask).float().to(device)    
        self.length = len(target)
        print('N data : ',self.length)
        
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx],self.jet_mask[idx]        

class CustomDataset_Latent_Hl(Dataset):
    def __init__(self, filelist,device,scaler_path,Xbb_scores_path,test=False):
        self.device = device
        self.x=[]
        self.y=[]
        self.jet_mask=[]
        i=0
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                with h5py.File(filename, 'r') as Data:
                    if i ==0:
                        data = Data['X_jet'][:]
                        target = Data['labels'][:] 
                        jet_mask = Data['jet_mask'][:]
                    else:
                        data = np.concatenate((data,Data['X_jet'][:]),axis=0)
                        target = np.concatenate((target,Data['labels'][:]),axis=0)
                        jet_mask = np.concatenate((jet_mask,Data['jet_mask'][:]),axis=0)
                    i+=1    
        data[:,:,jVars.index('fj_pt')] = log(data[:,:,jVars.index('fj_pt')])
        data[:,:,jVars.index('fj_mass')] = log(data[:,:,jVars.index('fj_mass')])
        data[:,:,jVars.index('fj_sdmass')] = log(data[:,:,jVars.index('fj_sdmass')])
        print('loading Xbb scores from : ',Xbb_scores_path)
        with h5py.File(Xbb_scores_path, 'r') as latent:
            target = latent['evt_label'][:]
            jet_mask = latent['jet_mask'][:]
            data = np.sum(np.concatenate( (np.nan_to_num(data),np.nan_to_num(latent['evt_score'][:])) ,axis=-1)*jet_mask[:,:,np.newaxis],axis=1)
        self.x = torch.from_numpy(data).float().to(device)    
        self.y = torch.from_numpy(target.reshape(-1,1)).float().to(device)
        self.jet_mask = torch.from_numpy(jet_mask).float().to(device)    
        self.length = len(target)
        print('N data : ',self.length)
        
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx],self.jet_mask[idx]        

            
def train_step(model,data,target,jet_mask,opt,loss_fn):
    model.train()
    preds = model(data,jet_mask)
    loss = loss_fn(preds,target)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model, loss_fn,train_loader,val_loader,subset,device,subset_batches=1):
    with torch.no_grad():
        model.eval()
        for i, train_batch in enumerate( train_loader ): 
            if (subset and i >= subset_batches ): break 
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
            if (subset and i >= subset_batches ): break 
            if i==0:
                data_val, target_val, jet_mask_val = val_batch
                data_val = data_val.cpu().numpy()
                target_val = target_val.cpu().numpy()
                jet_mask_val = jet_mask_val.cpu().numpy()
            else: 
                data_val = np.concatenate((data_val,val_batch[0].cpu().numpy()),axis=0)
                target_val = np.concatenate((target_val,val_batch[1].cpu().numpy()),axis=0)   
                jet_mask_val = np.concatenate((jet_mask_val,val_batch[2].cpu().numpy()),axis=0)          

        train_loss = loss_fn(model( torch.from_numpy(data).float().to(device),torch.from_numpy(jet_mask).float().to(device)).reshape(len(data)),torch.from_numpy(target.reshape(-1)).float().to(device))
        test_loss = loss_fn(model( torch.from_numpy(data_val).float().to(device),torch.from_numpy(jet_mask_val).float().to(device)).reshape(len(data_val)),torch.from_numpy(target_val.reshape(-1)).float().to(device))
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    

def train_loop(model,filelist,filelist_val, device, experiment, path, scaler_path,Xbb_scores_path,Xbb_scores_path_val,subset,modeltype,config):
    opt = optim.Adam(model.parameters(), config['LR'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([13.76]).to(device))
    evals = []
    best_val_loss = float('inf')
    with TemporaryDirectory() as tempdir:
        best_model_params_path = path
    if modeltype == 'mlpXbb':
        Dataset = CustomDataset_XbbOnly(filelist,device,scaler_path,Xbb_scores_path)
        Dataset_val = CustomDataset_XbbOnly(filelist_val,device,scaler_path,Xbb_scores_path_val,test=True)
    elif modeltype == 'mlpLatent':
        Dataset = CustomDataset_Latent(filelist,device,scaler_path,Xbb_scores_path)
        Dataset_val = CustomDataset_Latent(filelist_val,device,scaler_path,Xbb_scores_path_val,test=True)    
    elif modeltype == 'mlpLatentHl':
        Dataset = CustomDataset_Latent_Hl(filelist,device,scaler_path,Xbb_scores_path)
        Dataset_val = CustomDataset_Latent_Hl(filelist_val,device,scaler_path,Xbb_scores_path_val,test=True)    
    else:    
        Dataset = CustomDataset(filelist,device,scaler_path,Xbb_scores_path)
        Dataset_val = CustomDataset(filelist_val,device,scaler_path,Xbb_scores_path_val,test=True)

    # num_samples = Dataset.length
    # num_train = int(0.80 * num_samples)
    # num_val = num_samples - num_train
    # train_dataset, val_dataset = torch.utils.data.random_split(Dataset, [num_train, num_val])    
    val_loader = DataLoader(Dataset_val, batch_size=config['batch_size'], shuffle=True)
    for epoch in range (0,config['epochs']):
        print(f'epoch: {epoch+1}') 
        train_loader = DataLoader(Dataset, batch_size=config['batch_size'], shuffle=True)
        for i, train_batch in enumerate( train_loader ):
            if (subset and i >= config['subset_batches'] ): break 
            data, target, jet_mask = train_batch
            report = train_step(model, data, target,jet_mask, opt, loss_fn )
        evals.append(eval_fn(model, loss_fn,train_loader,val_loader,subset,device,config['subset_batches']) )         
        val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
            
        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    

    return evals, model


def get_preds(model,loader,subset,device):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate( loader ):
            if (i % 500) == 0: print('batch : ', i)
            if i==0:
                data, target, jet_mask = batch
                yi = model(data,jet_mask).detach().cpu().numpy()
                target = target.cpu().numpy()
            else: 
                yi = np.concatenate((yi,model(batch[0],batch[2]).detach().cpu().numpy()),axis=0)
                target = np.concatenate((target,batch[1].cpu().numpy()),axis=0)
            if (subset and i > 10): break    
    return yi,target




import torch
import torch.nn as nn
import torch.nn.functional as F




class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x,jet_mask):
        # compute the representation for each data point
        x = self.phi(x)*jet_mask[:,:,np.newaxis]

        # sum up the representations
        x = torch.sum(x, dim=1)

        # compute the output
        out = self.rho(x)

        return out

class InvariantModel_Latent(nn.Module):
    def __init__(self, rho: nn.Module):
        super().__init__()
        self.rho = rho

    def forward(self, x,jet_mask):
        out = self.rho(x)

        return out