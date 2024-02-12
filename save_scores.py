from comet_ml import Experiment,ExistingExperiment
from comet_ml.integration.pytorch import log_model
from Finetune_hep.python import helpers,models
import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--out', help='message',default='/raven/u/mvigl/public/run/Xbb/scores')
parser.add_argument('--config', help='config',default='/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_config.yaml')
parser.add_argument('--data', help='data',default='/raven/u/mvigl/Finetune_hep_dir/config/val_list.txt')
parser.add_argument('--checkpoint',  help='training-checkpoint',default='/raven/u/mvigl/public/run/Xbb/models/Xbb_lr0.01_bs512_subset0.1.pt')
parser.add_argument('--repDim', type=int,  help='repDim',default='1')

args = parser.parse_args()

device = helpers.get_device()
model = models.full_model(args.config,save_representaions=False,for_inference=True)
model = helpers.load_weights(model,args.checkpoint,device)
if (not os.path.exists(args.out)): os.system(f'mkdir {args.out}')

if __name__ == '__main__':    

    print(model)
    model.to(device)
    models.save_rep(model,device,args.data,args.out,args.repDim)