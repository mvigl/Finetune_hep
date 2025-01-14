from Finetune_hep.python import helpers,models
import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--out', help='message',default='/raven/u/mvigl/public/run/Frozen_Xbb_hl/scores')
parser.add_argument('--config', help='config',default='/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_hlf_config.yaml')
parser.add_argument('--data', help='data',default='/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt')
parser.add_argument('--checkpoint',  help='training-checkpoint',default='/raven/u/mvigl/public/run/Frozen_Xbb_hl/models/Frozen_Xbb_hl_lr0.001_bs512_subset0.1.pt')
parser.add_argument('--repDim', type=int,  help='repDim',default='1')
parser.add_argument('--out_dim', type=int,  help='out_dim',default='1')
parser.add_argument('--save_representaions',  action='store_true', help='save_representaions', default=False)
parser.add_argument('--ishead',  action='store_true', help='ishead', default=False)
parser.add_argument('--Xbb', help='data',default='/raven/u/mvigl/public/Finetune_hep/config/Xbb_test_list.txt')
parser.add_argument('--scaler_path',  help='scaler_path',default='')
parser.add_argument('--use_hlf',  action='store_true', help='use_hlf', default=True)
parser.add_argument('--LoRa',  action='store_true', help='use_LoRa', default=True)
parser.add_argument('--alphaXbb',  type=float, help='alphaXbb',default=0)
args = parser.parse_args()

device = helpers.get_device()
if args.ishead: 
    print('head')
    model = models.head_model(args.config,save_representaions=args.save_representaions,for_inference=True)
    model = helpers.load_weights(model,args.checkpoint,device)
else: 
    model = models.full_model(args.config,save_representaions=args.save_representaions,for_inference=True)
    if args.LoRa: 
        print("using LoRa!!!")
        lora_layers = ["mod.blocks", "mod.cls_blocks","mod.embed"]
        model_LoRa = helpers.apply_lora_to_model(model, lora_layers, rank=4)
        model_LoRa = helpers.load_weights(model_LoRa,args.checkpoint,device)
    else:
        model = helpers.load_weights(model,args.checkpoint,device)
if (not os.path.exists(args.out)): os.system(f'mkdir {args.out}')
if (not os.path.exists(f'{args.out}/scores')): os.system(f'mkdir {args.out}/scores')

if __name__ == '__main__':    

    if args.LoRa: 
        print(model_LoRa)
        model_LoRa.to(device)
        if args.ishead: 
            out_dim = args.out_dim
            if args.save_representaions: out_dim = args.repDim
            models.save_rep_head(model_LoRa,device,args.data,args.out,args.repDim,args.Xbb,args.use_hlf,args.scaler_path,out_dim,args.alphaXbb)
        else: models.save_rep(model_LoRa,device,args.data,args.out,args.repDim)
    else:    
        print(model)
        model.to(device)
        if args.ishead: 
            out_dim = args.out_dim
            if args.save_representaions: out_dim = args.repDim
            models.save_rep_head(model,device,args.data,args.out,args.repDim,args.Xbb,args.use_hlf,args.scaler_path,out_dim,args.alphaXbb)
        else: models.save_rep(model,device,args.data,args.out,args.repDim)
    