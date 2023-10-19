from comet_ml import Experiment,ExistingExperiment
from comet_ml.integration.pytorch import log_model
from Finetune_hep.python import ParT_mlp
from Finetune_hep.python import ParT_mlp_Hl
from Finetune_hep.python import ParT_mlp_Xbb_Hl
from Finetune_hep.python import ParT_Xbb
from Finetune_hep.python import Mlp
from Finetune_hep.python import definitions as df
import torch
import argparse
import yaml

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float,  help='learning rate',default='0.00004')
parser.add_argument('--bs', type=int,  help='batch size',default='512')
parser.add_argument('--ep', type=int,  help='epochs',default='50')
parser.add_argument('--mess', help='message',default='training0')
parser.add_argument('--modeltype', help='modeltype',default='ParTevent')
parser.add_argument('--nlayer_mlp', type=int, help='nlayer_mlp',default=6)
parser.add_argument('--nodes_mlp', type=int, help='nodes_mlp',default=128)
parser.add_argument('--njets_mlp', type=int, help='njets_mlp',default=2)
parser.add_argument('--ParT_weights',  help='ParT_weights',default='../../Finetune_hep/models/ParT_full.pt')
parser.add_argument('--mlp_weights',  help='mlp_weights',default='no')
parser.add_argument('--config', help='config',default='../../Finetune_hep/config/myJetClass_full.yaml')
parser.add_argument('--data', help='data',default='/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/config/train_list.txt')
parser.add_argument('--data_val', help='data_val',default='/home/iwsatlas1/mavigl/Finetune_hep_dir/Finetune_hep/config/val_list.txt')
parser.add_argument('--Xbb', help='Xbb_scores_path',default='/home/iwsatlas1/mavigl/Hbb/ParT/Trained_ParT/data/ParT_Xbb.npy')
parser.add_argument('--Xbb_val', help='Xbb_scores_path_val',default='/home/iwsatlas1/mavigl/Hbb/ParT/Trained_ParT/data/ParT_Xbb.npy')
parser.add_argument('--project_name', help='project_name',default='Finetune_hep')
parser.add_argument('--subset',  action='store_true', help='subset', default=False)
parser.add_argument('--subset_batches',  type=float, help='njets_mlp',default=1)
parser.add_argument('--api_key', help='api_key',default='r1SBLyPzovxoWBPDLx3TAE02O')
parser.add_argument('--ws', help='workspace',default='mvigl')
parser.add_argument('--alpha', type=float,  help='alpha',default=0.01)
parser.add_argument('--checkpoint',  help='training-checkpoint',default='../../Finetune_hep/models/ParT_full.pt')
parser.add_argument('--check_message', help='check-exp-key',default='')
parser.add_argument('--start_epoch', type=int, help='start_epoch',default=0)
parser.add_argument('--yaml_file', help='yaml',default='')

args = parser.parse_args()

learning_rate = args.lr
batch_size = args.bs
epochs = args.ep
message = args.mess
nlayer_mlp = args.nlayer_mlp
nodes_mlp = args.nodes_mlp
njets_mlp = args.njets_mlp
config_path = args.config
modeltype = args.modeltype
ParT_weights = args.ParT_weights
mlp_weights = args.mlp_weights
Xbb_scores_path = args.Xbb
Xbb_scores_path_val = args.Xbb_val
project_name = args.project_name
subset = args.subset
api_key = args.api_key
workspace = args.ws
alpha = args.alpha
filelist = args.data
filelist_val = args.data_val
checkpoint = args.checkpoint
check_message = args.check_message
start_epoch = args.start_epoch
yaml_file = args.yaml_file
subset_batches = args.subset_batches

for m,w in zip(['Wmlp_','WparT_'],[mlp_weights,ParT_weights]):  
    if w != 'no':
        message = m + message

device = df.get_device()
idxmap = df.get_idxmap(filelist,subset_batches)
subset_batches_val=1
if subset: subset_batches_val = 0.1
idxmap_val = df.get_idxmap(filelist_val,subset_batches_val)

if modeltype in ['ParTevent','ParTXbb','Aux','ParTevent_frozen','ParTevent_Hl','ParTevent_Xbb_Hl']:
    with open(config_path) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)  

    if modeltype in['ParTevent','ParTevent_frozen','ParTevent_Hl','ParTevent_Xbb_Hl']:
        model = ParT_mlp.get_model(data_config,for_inference=False)  
        if modeltype == 'ParTevent_Hl': model = ParT_mlp_Hl.get_model(data_config,for_inference=False)  
        if modeltype == 'ParTevent_Xbb_Hl': model = ParT_mlp_Xbb_Hl.get_model(data_config,for_inference=False)  
        model.to(device)
        model = df.load_weights_ParT_mlp(model,modeltype,mlp_layers=1,ParT_params_path=ParT_weights,mlp_params_path=mlp_weights)  
        Xbb = False
        if modeltype == 'ParTevent_frozen':
            # Freeze all the weights except for layers containing 'fc' exactly
            for name, param in model.named_parameters():
                if 'fc.' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    elif modeltype == 'ParTXbb':
        idxmap = df.get_idxmap_Xbb(filelist,subset_batches)
        idxmap_val = df.get_idxmap_Xbb(filelist_val,subset_batches_val)
        model = ParT_Xbb.get_model(data_config,for_inference=False) 
        model.to(device)
        Xbb = True
        model = df.load_weights_ParT_mlp(model,modeltype,mlp_layers=1,ParT_params_path=ParT_weights,mlp_params_path=mlp_weights)

elif modeltype in ['mlpXbb','mlpHlXbb','baseline']:
    model = Mlp.InvariantModel( phi=Mlp.make_mlp(6,nodes_mlp,nlayer_mlp,binary=False),
                                rho=Mlp.make_mlp(nodes_mlp,nodes_mlp*2,nlayer_mlp))
    if modeltype == 'mlpXbb': 
        model = Mlp.InvariantModel( phi=Mlp.make_mlp(1,nodes_mlp,nlayer_mlp,binary=False),
                                    rho=Mlp.make_mlp(nodes_mlp,nodes_mlp*2,nlayer_mlp))
    model.to(device)
    if mlp_weights != 'no' : model.load_state_dict(torch.load(mlp_weights))

elif modeltype in ['mlpLatent']:
    model = Mlp.InvariantModel( phi=Mlp.make_mlp(128,128,3,binary=False),
                                rho=Mlp.make_mlp(128,128,3))
    #model = ParT_mlp.make_mlp(128,out_features=128,nlayer = 6,for_inference=False,binary=True)
    #model = Mlp.InvariantModel_Latent(rho=Mlp.make_mlp(128,nodes_mlp,nlayer_mlp))
    model.to(device)
    if mlp_weights != 'no' : model.load_state_dict(torch.load(mlp_weights))

elif modeltype in ['mlpLatentHl']:
    model = Mlp.InvariantModel( phi=Mlp.make_mlp(128+5,128,3,binary=False),
                                rho=Mlp.make_mlp(128,128,3))
    model.to(device)
    if mlp_weights != 'no' : model.load_state_dict(torch.load(mlp_weights))    

elif modeltype in ['LatentXbb','LatentXbb_Aux']:
    model = ParT_mlp.make_mlp(128,nodes_mlp,nlayer_mlp)
    model.to(device)
    if mlp_weights != 'no' : model.load_state_dict(torch.load(mlp_weights))

else:
    print('specify a model (ParTevent,ParTXbb,mlpXbb,mlpHlXbb,baseline)')    


hyper_params = {
   "learning_rate": learning_rate,
   "steps": epochs,
   "batch_size": batch_size,
   "alpha": alpha,
   "start_epoch": start_epoch, 
   "subset_batches": subset_batches,
   "subset_batches_val": subset_batches_val,
}



experiment_name = f'{modeltype}_hl{nlayer_mlp}_nodes{nodes_mlp}_nj{njets_mlp}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_{message}'
if modeltype == 'Aux': experiment_name = f'{modeltype}_hl{nlayer_mlp}_nodes{nodes_mlp}_nj{njets_mlp}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_alpha{hyper_params["alpha"]}_{message}'

if (checkpoint=='no') or (subset) or (check_message=='no'): 
    experiment = Experiment(
    api_key = api_key,
    project_name = project_name,
    workspace=workspace,
    log_graph=True, # Can be True or False.
    auto_metric_logging=True # Can be True or False
    )
    Experiment.set_name(experiment,experiment_name)
    print(experiment.get_key())
    experiment.log_parameter("exp_key", experiment.get_key())
    if ((modeltype in ['ParTevent','ParTevent_frozen','ParTXbb','Aux','ParTevent_Hl']) and (subset==False)):
        for i in range(10):
            with open(yaml_file) as file:
                check_config = yaml.load(file, Loader=yaml.FullLoader)  
                check_config['checkpoint'] = f'models/{experiment_name}.pt'
                check_config['check-message'] = experiment.get_key()
                check_config['start-epoch'] = i+1

            with open(f'config/training_{modeltype}_{message}_{i+1}.yaml', 'w') as file:
                yaml.dump(check_config, file)
else:
    experiment_check_name = f'{check_message}'
    experiment = ExistingExperiment(api_key=api_key, 
                                    previous_experiment=experiment_check_name,
                                    log_env_details=True,
                                    log_env_gpu=True
                                    )

experiment.log_parameters(hyper_params)

model_path = (f'models/{experiment_name}.pt' )

if modeltype not in ['mlpLatent','LatentXbb','LatentXbb_Aux','mlpXbb','mlpLatentHl']:
    scaler_path = (f'models/{experiment_name}.pkl' )
else:
    scaler_path = 'no'  
        
integer_file_map = df.create_integer_file_map(idxmap)
integer_file_map_val = df.create_integer_file_map(idxmap_val)

load_val_loss = False
if checkpoint!= 'no': 
    model.load_state_dict(torch.load(checkpoint))
    if subset: load_val_loss = True

print(model)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

if modeltype in ['ParTevent','ParTXbb','ParTevent_frozen']:
    evals_part, model_part = ParT_mlp.train_loop(
        model,
        idxmap,
        integer_file_map,
        idxmap_val,
        integer_file_map_val,
        device,
        experiment,
        model_path,
        subset,
        config = dict(    
            LR = hyper_params['learning_rate'],
            batch_size = hyper_params['batch_size'],
            epochs = hyper_params['steps'],
            Xbb = Xbb,
            start_epoch = hyper_params['start_epoch'],
            modeltype = modeltype,
            load_val_loss = load_val_loss
        )
    )

elif modeltype in ['ParTevent_Hl','ParTevent_Xbb_Hl']:    
        evals_part, model_part = ParT_mlp_Hl.train_loop(
        model,
        idxmap,
        integer_file_map,
        idxmap_val,
        integer_file_map_val,
        device,
        experiment,
        model_path,
        subset,
        config = dict(    
            LR = hyper_params['learning_rate'],
            batch_size = hyper_params['batch_size'],
            epochs = hyper_params['steps'],
            Xbb = Xbb,
            start_epoch = hyper_params['start_epoch'],
            modeltype = modeltype,
            load_val_loss = load_val_loss
        )
    )

elif modeltype in ['mlpXbb','mlpHlXbb','mlpLatent','baseline','LatentXbb','LatentXbb_Aux','mlpLatentHl']:
    evals_part, model_part = Mlp.train_loop(
        model,
        filelist,
        filelist_val,
        device,
        experiment,
        model_path,
        scaler_path,
        Xbb_scores_path,
        Xbb_scores_path_val,
        subset,
        modeltype,
        config = dict(    
            LR = hyper_params['learning_rate'],
            batch_size = hyper_params['batch_size'],
            epochs = hyper_params['steps'],
            subset_batches = hyper_params['subset_batches'],
            subset_batches_val = hyper_params['subset_batches_val']
        )
    )

if checkpoint == 'no' or subset:
    log_model(experiment, model, model_name = experiment_name )
else:
    log_model(experiment, model, model_name = experiment_check_name )