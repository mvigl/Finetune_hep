#!/opt/anaconda3/bin/python

from Finetune_hep.python import ParT_latent
from Finetune_hep.python import definitions as df
import argparse
import yaml
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--modeltype', help='modeltype',default='LatentXbb_Aux')
parser.add_argument('--nlayer_mlp', type=int, help='nlayer_mlp',default=6)
parser.add_argument('--nodes_mlp', type=int, help='nodes_mlp',default=128)
parser.add_argument('--weights',  help='weights',default='no')
parser.add_argument('--config', help='config',default='../../Finetune_hep/config/myJetClass_full.yaml')
parser.add_argument('--data', help='data',default='/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')


args = parser.parse_args()

nlayer_mlp = args.nlayer_mlp
nodes_mlp = args.nodes_mlp
config_path = args.config
modeltype = args.modeltype
weights = args.weights
data = args.data


X_pfo_train, X_jet_train, njets_train, labels_train, X_label_train, evts_train = df.get_train_data(data)
X_pfo_test, X_jet_test, njets_test, labels_test, X_label_test, evts_test = df.get_test_data(data)

device = df.get_device()
with open(config_path) as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)  

model = ParT_latent.get_model(data_config,for_inference=False) 
X_jet_train = np.reshape(X_jet_train[evts_train],(-1,len(df.jVars)))
X_pfo_train = np.reshape(X_pfo_train[evts_train],(-1,110,len(df.pVars)))
X_label_train = np.reshape(X_label_train[evts_train],(-1,len(df.labelVars)))
X_jet_test = np.reshape(X_jet_test[evts_test],(-1,len(df.jVars)))
X_pfo_test = np.reshape(X_pfo_test[evts_test],(-1,110,len(df.pVars)))
X_label_test = np.reshape(X_label_test[evts_test],(-1,len(df.labelVars)))

model.to(device)
train = df.build_features_and_labels_single_jet(X_pfo_train,X_jet_train,X_label_train)
test = df.build_features_and_labels_single_jet(X_pfo_test,X_jet_test,X_label_test)

if modeltype == 'LatentXbb':
    weights = (f'../../Finetune_hep/models/ParTevent/ParTevent_hl3_nodes128_nj2_lr4e-05_bs512_WparT_training_0.pt')
elif modeltype == 'LatentXbb_Aux':
    weights = (f'../../Finetune_hep/models/Aux/Aux_hl3_nodes128_nj2_lr4e-05_bs512_alpha0.01_WparT_training_0.pt')
name = f'{modeltype}_scores'
model_path = (f'../../Finetune_hep/models/{modeltype}/{name}_0.npy')
model_path_test = (f'../../Finetune_hep/models/{modeltype}/{name}_0_test.npy')

if modeltype == 'LatentXbb':
    model = df.load_weights_ParT_mlp(model,modeltype,mlp_layers=4,ParT_params_path=weights,mlp_params_path='no')
elif modeltype == 'LatentXbb_Aux':
    model = df.load_weights_ParT_mlp(model,modeltype,mlp_layers=5,ParT_params_path=weights,mlp_params_path='no')
model.eval()
preds = ParT_latent.get_preds(model,train,X_label_train,device)
preds_test = ParT_latent.get_preds(model,test,X_label_test,device)
with open(model_path, 'wb') as f:
    np.save(f, np.reshape(preds,(-1,5,128)))
with open(model_path_test, 'wb') as f:
    np.save(f, np.reshape(preds_test,(-1,5,128)))        