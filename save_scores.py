from Finetune_hep.python import helpers,models
import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--out', help='message',default='/raven/u/mvigl/public/run/Finetuned_Xbb_hl/scores')
parser.add_argument('--config', help='config',default='/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_hlf_config.yaml')
parser.add_argument('--data', help='data',default='/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt')
parser.add_argument('--checkpoint',  help='training-checkpoint',default='/raven/u/mvigl/public/run/Finetuned_Xbb_hl/models/Finetuned_Xbb_hl_lr0.001_bs256_subset0.1.pt')
parser.add_argument('--repDim', type=int,  help='repDim',default='1')
parser.add_argument('--save_representaions',  action='store_true', help='save_representaions', default=False)

args = parser.parse_args()

device = helpers.get_device()
model = models.full_model(args.config,save_representaions=args.save_representaions,for_inference=True)
model = helpers.load_weights(model,args.checkpoint,device)
if (not os.path.exists(args.out)): os.system(f'mkdir {args.out}')

if __name__ == '__main__':    

    print(model)
    model.to(device)
    models.save_rep(model,device,args.data,args.out,args.repDim)