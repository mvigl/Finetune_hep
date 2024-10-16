from comet_ml import Experiment,ExistingExperiment
from comet_ml.integration.pytorch import log_model
from Finetune_hep.python import train,helpers,models
import torch
import argparse
import yaml
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float,  help='learning rate',default='0.001')
parser.add_argument('--bs', type=int,  help='batch size',default='512')
parser.add_argument('--ep', type=int,  help='epochs',default='100')
parser.add_argument('--num_workers', type=int,  help='num_workers',default='12')
parser.add_argument('--mess', help='message',default='Scratch_Xbb_hl')
parser.add_argument('--config', help='config',default='../Finetune_hep/config/ParT_Xbb_hlf_config.yaml')
parser.add_argument('--data', help='data',default='../Finetune_hep/data/test_list.txt')
parser.add_argument('--data_val', help='data_val',default='../Finetune_hep/data/test_list.txt')
parser.add_argument('--project_name', help='project_name',default='FM_SBI_test')
parser.add_argument('--subset',  type=float, help='njets_mlp',default=1)
parser.add_argument('--api_key', help='api_key',default='')#r1SBLyPzovxoWBPDLx3TAE02O
parser.add_argument('--ws', help='workspace',default='')#mvigl
parser.add_argument('--checkpoint',  help='training-checkpoint',default='') #checkpoint or pre-trained backbone
parser.add_argument('--start_epoch', type=int, help='start_epoch',default=0)
parser.add_argument('--out', help='out directory',default='Scratch_Xbb_hl')

args = parser.parse_args()
if (not os.path.exists(args.out)): os.system(f'mkdir {args.out}')
if (not os.path.exists(f'{args.out}/models')): os.system(f'mkdir {args.out}/models')
if (not os.path.exists(f'{args.out}/scores')): os.system(f'mkdir {args.out}/scores')
device = helpers.get_device()
model = models.full_model(args.config,for_inference=False)

#subset_val=1
#if args.subset!=1: subset_val = 0.01
subset_val=0.01
if model.Task == 'Xbb':
    idxmap = helpers.get_idxmap_Xbb(args.data,args.subset)
    idxmap_val = helpers.get_idxmap_Xbb(args.data_val,subset_val)
else:
    idxmap = helpers.get_idxmap(args.data,args.subset)
    idxmap_val = helpers.get_idxmap(args.data_val,subset_val)
integer_file_map = helpers.create_integer_file_map(idxmap)
integer_file_map_val = helpers.create_integer_file_map(idxmap_val)

hyper_params = {
   "learning_rate": args.lr,
   "epochs": args.ep,
   "batch_size": args.bs,
   "start_epoch": args.start_epoch, 
   "num_workers": args.num_workers,
   "subset": args.subset,
}
experiment_name = f'{args.mess}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_subset{args.subset}'
experiment = Experiment(
    api_key = args.api_key,
    project_name = args.project_name,
    workspace='mvigl',
    log_graph=True, # Can be True or False.
    auto_metric_logging=True # Can be True or False
    )
Experiment.set_name(experiment,experiment_name)
print(experiment.get_key())
experiment.log_parameter("exp_key", experiment.get_key())
experiment.log_parameters(hyper_params)

config = dict(    
            LR = hyper_params['learning_rate'],
            batch_size = hyper_params['batch_size'],
            epochs = hyper_params['epochs'],
            device = device,
            idxmap = idxmap,
            idxmap_val = idxmap_val,
            integer_file_map = integer_file_map,
            integer_file_map_val = integer_file_map_val,
            out_model_path =  f'{args.out}/models/{experiment_name}.pt',
            start_epoch = hyper_params['start_epoch'],
            num_workers = hyper_params['num_workers'],
            experiment = experiment
        )

if args.checkpoint != '': model = helpers.load_weights(model,args.checkpoint,device)

if __name__ == '__main__':    

    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    model.to(device)
    evals, trained_model = train.train_loop(model, config)

    log_model(experiment, model, model_name = experiment_name )