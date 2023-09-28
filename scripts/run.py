import os
import sys
import yaml
import argparse

def GetParser():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', dest='config_file', required=True, help='YAML configuration file')
    return parser.parse_args()

def InitializeOutputDir():

        directory=os.popen('pwd').read()
        print(directory.split('/')[-2])
        if (directory.split('/')[-2]!='run') : 
                print('\n*********\nYou are in the wrong directory!\nCreate an output dir in Finetune_hep/run and run from there\n*********\n')
                sys.exit()
        if (not os.path.exists('models')): os.system('mkdir models')
        if (not os.path.exists('plots')): os.system('mkdir plots')
        if (not os.path.exists('config')): os.system('mkdir config')

	
def RunTraining(lr,bs,ep,Ntrainings,nlayer_mlp,nodes_mlp,njets_mlp,config_path,modeltype,
                ParT_weights,mlp_weights,data,data_val,Xbb_scores_path,
                Xbb_scores_path_val,project_name,subset,api_key,workspace,alpha,
                checkpoint,check_message,start_epoch,Fyaml) :

    macro = 'training.py'     

    mess = 'training_'+str(Ntrainings)
    if modeltype == 'Aux': mess = 'hlXbb3_'+mess
    command='python ../../Finetune_hep/'+macro+' --mess '+mess+' --lr '+str(lr)+' --bs '+str(bs)+\
            ' --ep '+str(ep)+' --njets_mlp '+str(njets_mlp)+' --nodes_mlp '+str(nodes_mlp)+' --modeltype '+modeltype+\
            ' --nlayer_mlp '+str(nlayer_mlp)+' --config '+config_path+' --ParT_weights '+ParT_weights+\
            ' --mlp_weights '+mlp_weights+' --data '+data+' --data_val '+data_val+' --Xbb '+Xbb_scores_path+' --Xbb_val '+Xbb_scores_path_val+' --project_name '+project_name+subset+\
            ' --api_key '+api_key+' --ws '+workspace+' --alpha '+str(alpha)+' --checkpoint '+checkpoint+' --check_message '+check_message+' --start_epoch '+str(start_epoch)+' --yaml_file '+Fyaml
    print(command)
    os.system(command)

def Load_default(modeltype):
    alpha = 0.01    
    if (modeltype =='ParTevent'): 
            lr = 1e-3
            bs = 256
            ep = 20
            nlayer_mlp = 3
            nodes_mlp = 128
            njets_mlp = 2
            config_path = '../../Finetune_hep/config/myJetClass_full.yaml'
            ParT_weights = '/home/iwsatlas1/mavigl/Finetune_hep_dir/run/Final_ParTXbb/models/ParTXbb_hl0_nodes128_nj1_lr0.001_bs512_WparT_training_0.pt'#'../../Finetune_hep/models/ParT_full.pt'
            mlp_weights = 'no'
            Xbb_scores_path = 'no'
            Xbb_scores_path_val = 'no'

    elif (modeltype =='Aux'): 
            lr = 0.00004
            bs = 512
            ep = 35
            nlayer_mlp = 3
            nodes_mlp = 128
            njets_mlp = 2
            config_path = '../../Finetune_hep/config/myJetClass_full.yaml'
            ParT_weights = '../../Finetune_hep/models/ParTXbb/ParTXbb_hl0_nodes128_nj1_lr4e-05_bs512_WparT_training_0.pt'
            mlp_weights = 'no'
            Xbb_scores_path = 'no'
            Xbb_scores_path_val = 'no'
            alpha = 1        

    elif (modeltype =='ParTXbb'):    
            lr = 1e-3
            bs = 512
            ep = 20
            nlayer_mlp = 0
            nodes_mlp = 128
            njets_mlp = 1
            config_path = '../../Finetune_hep/config/myJetClass_full.yaml'
            ParT_weights = '../../Finetune_hep/models/ParT_full.pt'
            mlp_weights = 'no'
            Xbb_scores_path = 'no'
            Xbb_scores_path_val = 'no'

    elif (modeltype =='LatentXbb'):    
            lr = 0.00004
            bs = 512
            ep = 100
            nlayer_mlp = 0
            nodes_mlp = 128
            njets_mlp = 1
            config_path = 'no'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = '../../Finetune_hep/models/LatentXbb/LatentXbb_scores_0.npy'    
            Xbb_scores_path_val = '../../Finetune_hep/models/LatentXbb/LatentXbb_scores_0_val.npy'    

    elif (modeltype =='LatentXbb_Aux'):    
            lr = 0.00004
            bs = 512
            ep = 100
            nlayer_mlp = 0
            nodes_mlp = 128
            njets_mlp = 1
            config_path = 'no'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = '../../Finetune_hep/models/LatentXbb_Aux/LatentXbb_Aux_scores_0.npy'   
            Xbb_scores_path_val = '../../Finetune_hep/models/LatentXbb_Aux/LatentXbb_Aux_scores_0_val.npy'            

    elif (modeltype =='mlpXbb'):
            lr = 5e-4
            bs = 512
            ep = 100
            nlayer_mlp = 6
            nodes_mlp = 12
            njets_mlp = 2
            config_path = 'no'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = '../../Finetune_hep/models/ParTXbb/ParTXbb_train_full.h5' 
            Xbb_scores_path_val = 'no'

    elif (modeltype =='mlpHlXbb'):
            lr = 1e-3
            bs = 512
            ep = 30
            nlayer_mlp = 3
            nodes_mlp = 24
            njets_mlp = 2
            config_path = 'no'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = '../../Finetune_hep/models/ParTXbb/Final_ParTXbb_train.h5'
            Xbb_scores_path_val = '../../Finetune_hep/models/ParTXbb/Final_ParTXbb_val.h5' 

    elif (modeltype =='mlpLatent'):
            lr = 5e-4
            bs = 512
            ep = 100
            nlayer_mlp = 6
            nodes_mlp = 128
            njets_mlp = 2
            config_path = 'no'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = '../../Finetune_hep/models/ParTXbb/ParT_latent_scores.npy'  
            Xbb_scores_path_val = '../../Finetune_hep/models/ParTXbb/ParT_latent_scores_val.npy'        

    elif (modeltype =='baseline'):
            lr = 1e-3
            bs = 512
            ep = 30
            nlayer_mlp = 3
            nodes_mlp = 24
            njets_mlp = 2
            config_path = 'no'
            ParT_weights = 'no'
            mlp_weights = 'no'
            Xbb_scores_path = 'no'        
            Xbb_scores_path_val = 'no'

    return lr,bs,ep,nlayer_mlp,nodes_mlp,njets_mlp,config_path,ParT_weights,mlp_weights,Xbb_scores_path,Xbb_scores_path_val,alpha     

def main():

    args=GetParser()
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

    lr= config['lr']
    bs= config['bs']
    ep= config['ep']
    nlayer_mlp = config['nlayer-mlp']
    nodes_mlp = config['nodes-mlp']
    njets_mlp = config['njets-mlp']
    config_path = config['config']
    modeltype = config['model']
    ParT_weights = config['ParT-weights']
    mlp_weights = config['mlp-weights']
    Xbb_scores_path = config['Xbb']
    Xbb_scores_path_val = config['Xbb-val']
    alpha = config['alpha']
    checkpoint= config['checkpoint']
    check_message = config['check-message']
    start_epoch = config['start-epoch']
    Fyaml = config['Fyaml']
    subset_batches = 1 #not used if subset == false

    InitializeOutputDir()
    
    if (config['default']): 
        lr,bs,ep,nlayer_mlp,nodes_mlp,njets_mlp,config_path,ParT_weights,mlp_weights,Xbb_scores_path,Xbb_scores_path_val,alpha = Load_default(modeltype)   
        print('using default')
    
    data = config['data']
    data_val = config['data-val']
    project_name = config['project-name']
    Ntrainings = config['Ntrainings'] 
    api_key = config['api-key']
    workspace = config['ws']

    if (config['subset']):
        subset = ' --subset_batches '+str(config['subset_batches'])+' --subset'
        print('using subset')
        print('subset batches: ', config['subset_batches'])
    else:
        subset = ''  
    
    RunTraining(lr,bs,ep,Ntrainings,nlayer_mlp,nodes_mlp,njets_mlp,config_path,
                modeltype,ParT_weights,mlp_weights,data,data_val,Xbb_scores_path,
                Xbb_scores_path_val,project_name,subset,api_key,workspace,alpha,
                checkpoint,check_message,start_epoch,Fyaml)
	

if __name__ == "__main__":
	main()


