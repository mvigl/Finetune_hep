from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from Finetune_hep.python import head,helpers,models
import argparse
import yaml

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float,  help='learning rate',default='0.001')
parser.add_argument('--bs', type=int,  help='batch size',default='512')
parser.add_argument('--ep', type=int,  help='epochs',default='100')
parser.add_argument('--num_workers', type=int,  help='num_workers',default='12')
parser.add_argument('--mess', help='message',default='Frozen_Xbb_hl')
parser.add_argument('--config', help='config',default='config/ParT_Xbb_hlf_config.yaml')
parser.add_argument('--data', help='data',default='/raven/u/mvigl/Finetune_hep_dir/config/train_list.txt')
parser.add_argument('--data_val', help='data_val',default='/raven/u/mvigl/Finetune_hep_dir/config/val_list.txt')
parser.add_argument('--Xbb', help='data',default='/raven/u/mvigl/public/Finetune_hep/config/Xbb_train_list.txt')
parser.add_argument('--Xbb_val', help='data_val',default='/raven/u/mvigl/public/Finetune_hep/config/Xbb_val_list.txt')
parser.add_argument('--project_name', help='project_name',default='test')
parser.add_argument('--subset',  type=float, help='njets_mlp',default=0.1)
parser.add_argument('--api_key', help='api_key',default='r1SBLyPzovxoWBPDLx3TAE02O')
parser.add_argument('--ws', help='workspace',default='mvigl')
parser.add_argument('--checkpoint',  help='training-checkpoint',default='')
parser.add_argument('--start_epoch', type=int, help='start_epoch',default=0)
parser.add_argument('--scaler_path',  help='training-checkpoint',default='')
args = parser.parse_args()

device = helpers.get_device()
model = models.head(args.config,for_inference=False)
with open(args.config) as file:
    use_hlf = yaml.load(file, Loader=yaml.FullLoader) ['inputs']['hlf']['concatenate']
subset_val=1
if args.subset!=1: subset_val = 0.005

hyper_params = {
   "learning_rate": args.lr,
   "epochs": args.ep,
   "batch_size": args.bs,
   "start_epoch": args.start_epoch, 
   "num_workers": args.num_workers,
   "subset": args.subset,
   "Xbb_scores_path": args.Xbb, 
   "Xbb_scores_path_val": args.Xbb_val, 
   "scaler_path": args.scaler_path, 
   "filelist": args.data, 
   "filelist_val": args.data_val, 
}
experiment_name = f'{args.mess}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_subset{args.subset}'
experiment = Experiment(
    api_key = 'r1SBLyPzovxoWBPDLx3TAE02O',
    project_name = 'public_test',
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
            filelist = hyper_params['filelist'],
            filelist_val = hyper_params['filelist_val'],
            device = device,
            Xbb_scores_path = hyper_params['Xbb_scores_path'],
            Xbb_scores_path_val = hyper_params['Xbb_scores_path_val'],
            use_hlf = use_hlf,
            subset = hyper_params['subset'],
            subset_val = subset_val,
            out_model_path =  f'models/{experiment_name}.pt',
            start_epoch = hyper_params['start_epoch'],
            num_workers = hyper_params['num_workers'],
            scaler_path = hyper_params['scaler_path'],
            experiment = experiment
        )

if args.checkpoint != '': model = helpers.load_weights(model,args.checkpoint,device)

if __name__ == '__main__':    

    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    model.to(device)
    evals, trained_model = head.train_loop(model, config)

    log_model(experiment, model, model_name = experiment_name )