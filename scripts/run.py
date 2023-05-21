import os
import sys
import argparse

def GetParser():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--lr', type=float, help='learning rate',default='0.00004')
	parser.add_argument('--bs', type=int, help='batch size',default='512')
	parser.add_argument('--ep', type=int, help='epochs',default='30')
	parser.add_argument('--Ntrainings', type=int, help='Ntrainings',default='10')
	parser.add_argument('--model', help='model',default='ParT_mlp')
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

	
def RunTraining(model, lr, bs, ep, Ntrainings) :

    macro = ''
    if (model =='ParT_mlp'): macro = 'combined.py'
    elif (model =='ParT_Xbb'): macro = 'jetTag.py'

    for i in range(Ntrainings):
        mess = 'training_'+str(i)
        command='../../source/'+macro+' --mess '+mess+' --lr '+str(lr)+' --bs '+str(bs)+' --ep '+str(ep)
        print(command)
        os.system(command)

def main():

    args=GetParser()
    model=args.model
    lr=args.lr
    bs=args.bs
    ep=args.ep
    Ntrainings=args.Ntrainings

    InitializeOutputDir()
    RunTraining(model, lr, bs, ep, Ntrainings)
	

if __name__ == "__main__":
	main()


