import os
import sys
import argparse

def GetParser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', type=float, help='learning rate',default='0.00004')
    parser.add_argument('--bs', type=int, help='batch size',default='512')
    parser.add_argument('--ep', type=int, help='epochs',default='2')
    parser.add_argument('--Ntrainings', type=int, help='Ntrainings',default='1')
    parser.add_argument('--nlayer_mlp', type=int, help='nlayer_mlp',default=6)
    parser.add_argument('--nodes_mlp', type=int, help='nodes_mlp',default=24)
    parser.add_argument('--njets_mlp', type=int, help='njets_mlp',default=2)
    parser.add_argument('--model', help='modeltype',default='')
    parser.add_argument('--ParT_weights',  help='ParT_weights',default='no')
    parser.add_argument('--mlp_weights',  help='mlp_weights',default='no')
    parser.add_argument('--config', help='config',default='/home/iwsatlas1/mavigl/Finetune_hep/source/config/myJetClass_full.yaml')
    parser.add_argument('--data', help='data',default='/home/iwsatlas1/mavigl/Hbb/ParT/Dataset')
    parser.add_argument('--Xbb', help='Xbb_scores_path',default='no')
    parser.add_argument('--project_name', help='project_name',default='Finetune_ParT')
    parser.add_argument('--default',  action='store_true', help='use default hp', default=False)
    parser.add_argument('--subset',  action='store_true', help='subset', default=False)
    parser.add_argument('--api_key', help='api_key',default='r1SBLyPzovxoWBPDLx3TAE02O')
    parser.add_argument('--ws', help='workspace',default='mvigl')


    args = parser.parse_args()
    return args

def InitializeOutputDir():

	directory=os.popen('pwd').read()
	print(directory.split('/')[-2])
	if (directory.split('/')[-2]!='run') : 
		print('\n*********\nYou are in the wrong directory!\nCreate an output dir in Finetune_hep/run and run from there\n*********\n')
		sys.exit()
	if (not os.path.exists('models')): os.system('mkdir models')
	if (not os.path.exists('plots')): os.system('mkdir plots')

	
def RunTraining(lr,bs,ep,Ntrainings,nlayer_mlp,nodes_mlp,njets_mlp,config_path,modeltype,ParT_weights,mlp_weights,data,Xbb_scores_path,project_name,subset,api_key,workspace) :

    macro = 'training.py'     

    for i in range(Ntrainings):
        mess = 'training_'+str(i)
        command='../../Finetune_hep/'+macro+' --mess '+mess+' --lr '+str(lr)+' --bs '+str(bs)+\
                ' --ep '+str(ep)+' --njets_mlp '+str(njets_mlp)+' --nodes_mlp '+str(nodes_mlp)+' --modeltype '+modeltype+\
                ' --nlayer_mlp '+str(nlayer_mlp)+' --config '+config_path+' --ParT_weights '+ParT_weights+\
                ' --mlp_weights '+mlp_weights+' --data '+data+' --Xbb '+Xbb_scores_path+' --project_name '+project_name+subset+\
                ' --api_key '+api_key+' --ws '+workspace
        print(command)
        os.system(command)

def Load_default(modeltype):

    if (modeltype =='ParTevent'): 
            lr = 0.00004
            bs = 512
            ep = 30
            nlayer_mlp = 3
            nodes_mlp = 128
            njets_mlp = 2
            config_path = '/home/iwsatlas1/mavigl/Finetune_hep/source/config/myJetClass_full.yaml'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = 'no'

    elif (modeltype =='ParTXbb'):    
            lr = 0.00004
            bs = 512
            ep = 50
            nlayer_mlp = 0
            nodes_mlp = 128
            njets_mlp = 1
            config_path = '/home/iwsatlas1/mavigl/Finetune_hep/source/config/myJetClass_full.yaml'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = 'no'

    elif (modeltype =='mlpXbb'):
            lr = 0.0001
            bs = 512
            ep = 50
            nlayer_mlp = 6
            nodes_mlp = 12
            njets_mlp = 2
            config_path = 'no'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = 'no'

    elif (modeltype =='mlpHlXbb'):
            lr = 0.0001
            bs = 512
            ep = 50
            nlayer_mlp = 6
            nodes_mlp = 24
            njets_mlp = 2
            config_path = 'no'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = 'no' 

    elif (modeltype =='baseline'):
            lr = 0.0001
            bs = 512
            ep = 50
            nlayer_mlp = 6
            nodes_mlp = 24
            njets_mlp = 2
            config_path = 'no'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = 'no'        

    return lr,bs,ep,nlayer_mlp,nodes_mlp,njets_mlp,config_path,ParT_weights,mlp_weights,Xbb_scores_path     

def main():

    args=GetParser()
    lr=args.lr
    bs=args.bs
    ep=args.ep
    nlayer_mlp = args.nlayer_mlp
    nodes_mlp = args.nodes_mlp
    njets_mlp = args.njets_mlp
    config_path = args.config
    modeltype = args.model
    ParT_weights = args.ParT_weights
    mlp_weights = args.mlp_weights
    Xbb_scores_path = args.Xbb

    InitializeOutputDir()
    
    if (args.default): 
        lr,bs,ep,nlayer_mlp,nodes_mlp,njets_mlp,config_path,ParT_weights,mlp_weights,Xbb_scores_path = Load_default(modeltype)   
    
    data = args.data
    project_name = args.project_name
    Ntrainings=args.Ntrainings    
    api_key = args.api_key
    workspace = args.ws

    if (args.subset):
        subset = ' --subset'
    else:
        subset = ''  
    
    RunTraining(lr,bs,ep,Ntrainings,nlayer_mlp,nodes_mlp,njets_mlp,config_path,modeltype,ParT_weights,mlp_weights,data,Xbb_scores_path,project_name,subset,api_key,workspace)
	

if __name__ == "__main__":
	main()


