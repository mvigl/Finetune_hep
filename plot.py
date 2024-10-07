import argparse
import os
import re 
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import roc_curve,auc
parser = argparse.ArgumentParser(description='')
parser.add_argument('--out', help='message',default='/raven/u/mvigl/public/run/Frozen_Xbb_hl/scores')
parser.add_argument('--config', help='config',default='/raven/u/mvigl/public/Finetune_hep/config/ParT_Xbb_hlf_config.yaml')
parser.add_argument('--data', help='data',default='/raven/u/mvigl/Finetune_hep_dir/config/test_list.txt')
parser.add_argument('--checkpoint',  help='training-checkpoint',default='/raven/u/mvigl/public/run/Frozen_Xbb_hl/models/Frozen_Xbb_hl_lr0.001_bs512_subset0.1.pt')
parser.add_argument('--repDim', type=int,  help='repDim',default='1')
parser.add_argument('--save_representaions',  action='store_true', help='save_representaions', default=False)
parser.add_argument('--ishead',  action='store_true', help='ishead', default=False)
parser.add_argument('--Xbb', help='data',default='/raven/u/mvigl/public/Finetune_hep/config/Xbb_test_list.txt')
parser.add_argument('--scaler_path',  help='scaler_path',default='')
parser.add_argument('--use_hlf',  action='store_true', help='use_hlf', default=True)

args = parser.parse_args()

jVars = [f'fj_{v}' for v in ['pt','eta','doubleb','phi','mass','sdmass']]

def get_data(in_dir,filelist,feat_dir,Xbb_dir):
    cs =[7823.28,  648.174, 186.946, 32.2928]
    with open(filelist) as f:
        i=-1
        for line in f:
            filename = f'{in_dir}/{line.strip()}'
            filename_feat = f'{feat_dir}/{line.strip()}'
            filename_Xbb = f'{Xbb_dir}/{line.strip()}'
            print('reading : ',filename)
            pattern = r'test_(.*?)_'
            match = re.search(pattern, filename)
            match_s = re.search(r'(\d+)(?=\.\w+$)', filename)
            last_number = float(match_s.group(1))
            print(last_number)
            if match:
                number = float(match.group(1))
                print(number)
            else:
                print("No match found")
            with h5py.File(filename, 'r') as data:
                i+=1    
                if i==0:
                    evt_label = data['evt_label'][:]
                    evt_score = data['evt_score'][:]
                    evt_mass = (number*np.ones(len(data['evt_label'][:]))).astype(int)
                else:
                    evt_label_i = data['evt_label'][:]
                    evt_score_i = data['evt_score'][:]
                    evt_mass_i = (number*np.ones(len(data['evt_label'][:]))).astype(int) 
                    evt_label = np.concatenate((evt_label,evt_label_i),axis=0)
                    evt_score = np.concatenate((evt_score,evt_score_i),axis=0)
                    evt_mass = np.concatenate((evt_mass,evt_mass_i),axis=0)

            with h5py.File(filename_feat, 'r') as data:
                if i==0:
                    sdmass = data['X_jet'][:,:,jVars.index('fj_sdmass')]
                    mass = data['X_jet'][:,:,jVars.index('fj_mass')]
                else:
                    sdmass_i = data['X_jet'][:,:,jVars.index('fj_sdmass')]
                    mass_i = data['X_jet'][:,:,jVars.index('fj_mass')]
                    sdmass = np.concatenate((sdmass,sdmass_i),axis=0)
                    mass = np.concatenate((mass,mass_i),axis=0)

            with h5py.File(filename_Xbb, 'r') as data:
                if i==0:
                    Xbb_score = data['Xbb_score'][:]
                    Xbb_label = data['Xbb_label'][:]
                else:
                    Xbb_score_i = data['Xbb_score'][:]
                    Xbb_label_i = data['Xbb_label'][:]
                    Xbb_score = np.concatenate((Xbb_score,Xbb_score_i),axis=0)           
                    Xbb_label = np.concatenate((Xbb_label,Xbb_label_i),axis=0)    

            if number==0:
                if last_number < 70 : 
                    if i==0: weights = np.ones(len(evt_label))*cs[0]
                    else:
                        weights_i = np.ones(len(evt_label_i))*cs[0]
                        weights = np.concatenate((weights,weights_i),axis=0)   
                elif ((last_number>69) and (last_number < 125)) : 
                    if i==0: weights = np.ones(len(evt_label))*cs[1]
                    else:
                        weights_i = np.ones(len(evt_label_i))*cs[1]
                        weights = np.concatenate((weights,weights_i),axis=0) 
                elif ((last_number>124) and (last_number < 201)) : 
                    if i==0: weights = np.ones(len(evt_label))*cs[2]
                    else:
                        weights_i = np.ones(len(evt_label_i))*cs[2]
                        weights = np.concatenate((weights,weights_i),axis=0)  
                elif ((last_number>200) and (last_number < 261)) : 
                    if i==0: weights = np.ones(len(evt_label))*cs[3]
                    else:
                        weights_i = np.ones(len(evt_label_i))*cs[3]
                        weights = np.concatenate((weights,weights_i),axis=0) 
                else: 
                    if i==0: weights = np.ones(len(evt_label))
                    else:
                        weights_i = np.ones(len(evt_label_i))
                        weights = np.concatenate((weights,weights_i),axis=0)                
            else: 
                if i==0: weights = np.ones(len(evt_label))
                else:
                    weights_i = np.ones(len(evt_label_i))
                    weights = np.concatenate((weights,weights_i),axis=0)               
    print('===========')
    print('bkg samples : ',np.sum(evt_mass==0.0))
    for mass_point in [600,1000,1200,1400,1600,1800,2000,2500,3000,4000,4500]:
        print(f'{mass_point} sig samples : ',np.sum(evt_mass==mass_point))
    print('===========')
    return evt_score,evt_label,evt_mass,mass,sdmass,Xbb_score.reshape(-1,5),Xbb_label.reshape(-1,5),weights

def plot_auc(h5fw,mass=0):
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    plt.setp(ax.get_xticklabels(), visible=False)
    if mass!=0: ax.set_title(f'AUC Xm = {mass}')
    else: ax.set_title('AUC all mass points')
    ax.set_ylabel('AUC')
    ax.semilogx()
    r.semilogx()
    r.set_xlabel('#finetune / #pretrain')
    Scratch_Xbb_hl = np.array(   [h5fw['Scratch_Xbb_hl']['0_0001']['auc'][()],
                                h5fw['Scratch_Xbb_hl']['0_001']['auc'][()],
                                h5fw['Scratch_Xbb_hl']['0_01']['auc'][()],
                                h5fw['Scratch_Xbb_hl']['0_1']['auc'][()],
                                h5fw['Scratch_Xbb_hl']['1']['auc'][()],]
                            )
    Frozen_Xbb_hl = np.array(   [h5fw['Frozen_Xbb_hl']['0_0001']['auc'][()],
                                h5fw['Frozen_Xbb_hl']['0_001']['auc'][()],
                                h5fw['Frozen_Xbb_hl']['0_01']['auc'][()],
                                h5fw['Frozen_Xbb_hl']['0_1']['auc'][()],
                                h5fw['Frozen_Xbb_hl']['1']['auc'][()],]
                            )
    Finetune_Xbb_hl = np.array(     [h5fw['Finetune_Xbb_hl']['0_0001']['auc'][()],
                                    h5fw['Finetune_Xbb_hl']['0_001']['auc'][()],
                                    h5fw['Finetune_Xbb_hl']['0_01']['auc'][()],
                                    h5fw['Finetune_Xbb_hl']['0_1']['auc'][()],
                                    h5fw['Finetune_Xbb_hl']['1']['auc'][()],]
                            )                
    x=np.array([0.0001,0.001,0.01,0.1,1]) 
    if mass !=0:
        mass_filter = ((h5fw['Frozen_Xbb_hl']['0_0001']['evt_mass'][:]==mass)+(h5fw['Frozen_Xbb_hl']['0_0001']['evt_label'][:]==0)).astype(bool)
        y = h5fw['Frozen_Xbb_hl']['0_0001']['evt_label'][mass_filter][:]
        Scratch_Xbb_hl = []
        Frozen_Xbb_hl = []
        Finetune_Xbb_hl = []
        for sample in ['0_0001','0_001','0_01','0_1','1']:
            fpr, tpr, thresholds = roc_curve(y,h5fw['Scratch_Xbb_hl'][f'{sample}']['evt_score'][mass_filter])
            Scratch_Xbb_hl.append(auc(fpr,tpr))
            fpr, tpr, thresholds = roc_curve(y,h5fw['Frozen_Xbb_hl'][f'{sample}']['evt_score'][mass_filter])
            Frozen_Xbb_hl.append(auc(fpr,tpr))
            fpr, tpr, thresholds = roc_curve(y,h5fw['Finetune_Xbb_hl'][f'{sample}']['evt_score'][mass_filter])
            Finetune_Xbb_hl.append(auc(fpr,tpr))
        Scratch_Xbb_hl = np.array(Scratch_Xbb_hl)   
        Frozen_Xbb_hl = np.array(Frozen_Xbb_hl)   
        Finetune_Xbb_hl = np.array(Finetune_Xbb_hl)    
    ax.plot(x,Frozen_Xbb_hl,color='indianred',label='S+HLF Frozen')
    ax.plot(x,Finetune_Xbb_hl,color='red',label='S+HLF Finetuned')
    ax.plot(x,Scratch_Xbb_hl,color='maroon',label='S+HLF Scratch')
    r.plot(x,Frozen_Xbb_hl/Frozen_Xbb_hl,color='indianred')
    r.plot(x,Finetune_Xbb_hl/Frozen_Xbb_hl,color='red')
    r.plot(x,Scratch_Xbb_hl/Frozen_Xbb_hl,color='maroon')
    r.set_ylim(0.975,1.025)
    r.set_xlim(0.01,10)
    ax.set_xlim(0.01,10)
    ax.axvline(x=1, color='black', linestyle='--')
    r.axvline(x=1, color='black', linestyle='--')
    ax.legend()
    if mass !=0: plt.savefig(f'plots/auc_mass_{mass}.png')
    else: plt.savefig(f'plots/auc.png')

def plot_bkg_rej(h5fw,sample='0_1'):
    tpr_common = h5fw['Scratch_Xbb_hl'][f'{sample}']['tpr'][:]  
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title('Bkg rejection all mass points')
    ax.set_ylabel('Bkg_rej')
    ax.semilogy()
    r.set_xlabel('signal efficiency')
    Scratch_Xbb_hl =1/h5fw['Scratch_Xbb_hl'][f'{sample}']['fpr'][:]  
    Frozen_Xbb_hl = 1/h5fw['Frozen_Xbb_hl'][f'{sample}']['fpr'][:]
    Finetune_Xbb_hl =1/h5fw['Finetune_Xbb_hl'][f'{sample}']['fpr'][:]  
      
    ax.plot(tpr_common,Frozen_Xbb_hl,color='indianred',label='S+HLF Frozen')
    ax.plot(tpr_common,Finetune_Xbb_hl,color='red',label='S+HLF Finetuned')
    ax.plot(tpr_common,Scratch_Xbb_hl,color='maroon',label='S+HLF Scratch')
    r.plot(tpr_common,Frozen_Xbb_hl/Frozen_Xbb_hl,color='indianred')
    r.plot(tpr_common,Finetune_Xbb_hl/Frozen_Xbb_hl,color='red')
    r.plot(tpr_common,Scratch_Xbb_hl/Frozen_Xbb_hl,color='maroon')
    r.set_ylim(0,5)
    r.set_xlim(0.6,1)
    ax.set_xlim(0.6,1)
    ax.set_ylim(1,Finetune_Xbb_hl[int(len(tpr_common)*0.6)])
    ax.legend()
    plt.savefig(f'plots/bkg_rej_{sample}.png')      

def plot_bkg_rej_seff(mass=0,seff=0.9):
    tpr_common = h5fw['Scratch_Xbb_hl']['0_0001']['tpr'][:]  
    cut = int(len(tpr_common)*seff)
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    plt.setp(ax.get_xticklabels(), visible=False)
    if mass!=0: ax.set_title(f'Bkg rejection Xm = {mass}')
    else: ax.set_title('Bkg rejection all mass points')
    ax.set_ylabel('Bkg rej')
    ax.semilogx()
    ax.semilogy()
    r.semilogx()
    r.set_xlabel('#finetune / #pretrain')
    Scratch_Xbb_hl = np.array( [  1/h5fw['Scratch_Xbb_hl']['0_0001']['fpr'][cut],
                        1/h5fw['Scratch_Xbb_hl']['0_001']['fpr'][cut],
                        1/h5fw['Scratch_Xbb_hl']['0_01']['fpr'][cut],
                        1/h5fw['Scratch_Xbb_hl']['0_1']['fpr'][cut],
                        1/h5fw['Scratch_Xbb_hl']['1']['fpr'][cut],]  
    )

    Frozen_Xbb_hl = np.array( [  1/h5fw['Frozen_Xbb_hl']['0_001']['fpr'][cut],
                        1/h5fw['Frozen_Xbb_hl']['0_001']['fpr'][cut],
                        1/h5fw['Frozen_Xbb_hl']['0_01']['fpr'][cut],
                        1/h5fw['Frozen_Xbb_hl']['0_1']['fpr'][cut],
                        1/h5fw['Frozen_Xbb_hl']['1']['fpr'][cut],]  
    )
    Finetune_Xbb_hl = np.array( [  1/h5fw['Finetune_Xbb_hl']['0_0001']['fpr'][cut],
                        1/h5fw['Finetune_Xbb_hl']['0_001']['fpr'][cut],
                        1/h5fw['Finetune_Xbb_hl']['0_01']['fpr'][cut],
                        1/h5fw['Finetune_Xbb_hl']['0_1']['fpr'][cut],
                        1/h5fw['Finetune_Xbb_hl']['1']['fpr'][cut],]  
    )
    x=np.array([0.0001,0.001,0.01,0.1,1])  
    if mass !=0:
        mass_filter = ((h5fw['Frozen_Xbb_hl']['0_0001']['evt_mass'][:]==mass)+(h5fw['Frozen_Xbb_hl']['0_0001']['evt_label'][:]==0)).astype(bool)
        y = h5fw['Frozen_Xbb_hl']['0_0001']['evt_label'][mass_filter][:]
        Scratch_Xbb_hl = []
        Frozen_Xbb_hl = []
        Finetune_Xbb_hl = []
        for sample in ['0_0001','0_001','0_01','0_1','1']:
            fpr, tpr, thresholds = roc_curve(y,h5fw['Scratch_Xbb_hl'][f'{sample}']['evt_score'][mass_filter])
            fpr = np.interp(tpr_common,tpr,fpr)
            Scratch_Xbb_hl.append(1/fpr[cut])
            fpr, tpr, thresholds = roc_curve(y,h5fw['Frozen_Xbb_hl'][f'{sample}']['evt_score'][mass_filter])
            fpr = np.interp(tpr_common,tpr,fpr)
            Frozen_Xbb_hl.append(1/fpr[cut])
            fpr, tpr, thresholds = roc_curve(y,h5fw['Finetune_Xbb_hl'][f'{sample}']['evt_score'][mass_filter])
            fpr = np.interp(tpr_common,tpr,fpr)
            Finetune_Xbb_hl.append(1/fpr[cut])
        Scratch_Xbb_hl = np.array(Scratch_Xbb_hl)   
        Frozen_Xbb_hl = np.array(Frozen_Xbb_hl)   
        Finetune_Xbb_hl = np.array(Finetune_Xbb_hl) 
    ax.plot(x,Frozen_Xbb_hl,color='indianred',label='S+HLF Frozen')
    ax.plot(x,Finetune_Xbb_hl,color='red',label='S+HLF Finetuned')
    ax.plot(x,Scratch_Xbb_hl,color='maroon',label='S+HLF Scratch')
    r.plot(x,Frozen_Xbb_hl/Frozen_Xbb_hl,color='indianred')
    r.plot(x,Finetune_Xbb_hl/Frozen_Xbb_hl,color='red')
    r.plot(x,Scratch_Xbb_hl/Frozen_Xbb_hl,color='maroon')
    r.set_ylim(0,5)
    r.set_xlim(0.01,10)
    ax.set_xlim(0.01,10)
    ax.axvline(x=1, color='black', linestyle='--')
    r.axvline(x=1, color='black', linestyle='--')
    ax.legend()
    if mass !=0: plt.savefig(f'plots/bkg_rej_seff_{seff}_mass_{mass}.png')
    else: plt.savefig(f'plots/bkg_rej_seff_{seff}.png') 

def plot_all_masses(masses,seff=0.9,metric='auc'):
    tpr_common = h5fw['Scratch_Xbb_hl']['0_0001']['tpr'][:]  
    cut = int(len(tpr_common)*seff)
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(f'{metric} at {seff} signal efficiency at #pretrain = #finetune')
    ax.set_ylabel(f'{metric}')
    ax.semilogy()
    r.set_xlabel('Xm')
    x=masses
    Scratch_Xbb_hl = []
    Frozen_Xbb_hl = []
    Finetune_Xbb_hl = []
    sample = '1'
    for mass in masses:
        mass_filter = ((h5fw['Frozen_Xbb_hl'][f'{sample}']['evt_mass'][:]==mass)+(h5fw['Frozen_Xbb_hl'][f'{sample}']['evt_label'][:]==0)).astype(bool)
        y = h5fw['Frozen_Xbb_hl'][f'{sample}']['evt_label'][mass_filter][:]
        fpr, tpr, thresholds = roc_curve(y,h5fw['Scratch_Xbb_hl'][f'{sample}']['evt_score'][mass_filter])
        if metric=='auc': Scratch_Xbb_hl.append(auc(fpr,tpr))
        else: 
            fpr = np.interp(tpr_common,tpr,fpr)
            Scratch_Xbb_hl.append(1/fpr[cut])
        fpr, tpr, thresholds = roc_curve(y,h5fw['Frozen_Xbb_hl'][f'{sample}']['evt_score'][mass_filter])
        if metric=='auc': Frozen_Xbb_hl.append(auc(fpr,tpr))
        else:  
            fpr = np.interp(tpr_common,tpr,fpr)
            Frozen_Xbb_hl.append(1/fpr[cut])
        fpr, tpr, thresholds = roc_curve(y,h5fw['Finetune_Xbb_hl'][f'{sample}']['evt_score'][mass_filter])
        if metric=='auc': Finetune_Xbb_hl.append(auc(fpr,tpr))
        else:  
            fpr = np.interp(tpr_common,tpr,fpr)
            Finetune_Xbb_hl.append(1/fpr[cut])
    Scratch_Xbb_hl = np.array(Scratch_Xbb_hl)   
    Frozen_Xbb_hl = np.array(Frozen_Xbb_hl)   
    Finetune_Xbb_hl = np.array(Finetune_Xbb_hl) 
    ax.plot(x,Frozen_Xbb_hl,color='indianred',label='S+HLF Frozen')
    ax.plot(x,Finetune_Xbb_hl,color='red',label='S+HLF Finetuned')
    ax.plot(x,Scratch_Xbb_hl,color='maroon',label='S+HLF Scratch')
    r.plot(x,Frozen_Xbb_hl/Frozen_Xbb_hl,color='indianred')
    r.plot(x,Finetune_Xbb_hl/Frozen_Xbb_hl,color='red')
    r.plot(x,Scratch_Xbb_hl/Frozen_Xbb_hl,color='maroon')
    if metric=='auc': r.set_ylim(0.975,1.025)
    else: r.set_ylim(0,5)
    ax.set_xlim(600,4500)
    r.set_xlim(600,4500)
    ax.legend()
    plt.savefig(f'plots/{metric}_all_masses_seff_{seff}.png')

def plot_bkg_rej_vs_jmass(h5fw,sample='1',feat_name='sdmass',jetnumber=0, seff=0.7,bins=np.linspace(0,200,40),density=False,log=True,mess=''):
    tpr_common = np.linspace(0,1,10000)
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    #r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    #plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title('Bkg rejection all mass points')
    ax.set_ylabel('Bkg_rej')
    if log: ax.semilogy()
    #r.set_xlabel('signal efficiency')

    feat = h5fw['Finetune_Xbb_hl'][f'{sample}'][f'{feat_name}'][:,jetnumber]  
    #score_Scratch_Xbb_hl =h5fw['Scratch_Xbb_hl'][f'{sample}']['evt_score'][:]  
    #score_Frozen_Xbb_hl = h5fw['Frozen_Xbb_hl'][f'{sample}']['evt_score'][:]
    score_Finetune_Xbb_hl =h5fw['Finetune_Xbb_hl'][f'{sample}']['evt_score'][:]
    label =h5fw['Finetune_Xbb_hl'][f'{sample}']['evt_label'][:]
    Xbb_label =h5fw['Finetune_Xbb_hl'][f'{sample}']['Xbb_label'][:,jetnumber]    
    weights=h5fw['Finetune_Xbb_hl'][f'{sample}']['weights'][:]  
    if mess in ['noW','noWall']: weights=np.ones(len(weights))
    #fpr_Scratch_Xbb_hl, tpr_Scratch_Xbb_hl, thresholds_Scratch_Xbb_hl = roc_curve(label,score_Scratch_Xbb_hl)
    #fpr_Frozen_Xbb_hl, tpr_Frozen_Xbb_hl, thresholds_Frozen_Xbb_hl = roc_curve(label,score_Frozen_Xbb_hl)
    fpr_Finetune_Xbb_hl, tpr_Finetune_Xbb_hl, thresholds_Finetune_Xbb_hl = roc_curve(label,score_Finetune_Xbb_hl)
    #cut_Scratch_Xbb_hl = np.interp(tpr_common,tpr_Scratch_Xbb_hl,thresholds_Scratch_Xbb_hl)[int(len(tpr_common)*seff)]
    #cut_Frozen_Xbb_hl = np.interp(tpr_common,tpr_Frozen_Xbb_hl,thresholds_Frozen_Xbb_hl)[int(len(tpr_common)*seff)]
    cut_Finetune_Xbb_hl = np.interp(tpr_common,tpr_Finetune_Xbb_hl,thresholds_Finetune_Xbb_hl)[int(len(tpr_common)*seff)]
    print(cut_Finetune_Xbb_hl)
    print(score_Finetune_Xbb_hl)
    print(1*weights*(score_Finetune_Xbb_hl>=cut_Finetune_Xbb_hl)*Xbb_label)
    ax.hist(feat,density=density,bins=b,weights=1*weights*Xbb_label,histtype='step',color='red',label='Hbb')
    ax.hist(feat,density=density,bins=b,weights=1*weights*(score_Finetune_Xbb_hl>=cut_Finetune_Xbb_hl)*Xbb_label,histtype='step',linestyle='dashed',color='indianred',label=f'Hbb {seff*100}% sig-wp')
    ax.hist(feat,density=density,bins=b,weights=1*weights*(Xbb_label==0),histtype='step',color='blue',label='QCD')
    ax.hist(feat,density=density,bins=b,weights=1*weights*(score_Finetune_Xbb_hl>=cut_Finetune_Xbb_hl)*(Xbb_label==0),histtype='step',linestyle='dashed',color='blue',label=f'QCD {seff*100}% sig-wp')
    #ax.plot(feat_Finetune_Xbb_hl,color='red',label='S+HLF Finetuned')
    #ax.plot(feat_Frozen_Xbb_hl,color='maroon',label='S+HLF Scratch')
    #r.plot(feat_Frozen_Xbb_hl/feat_Frozen_Xbb_hl,color='indianred')
    #r.plot(feat_Finetune_Xbb_hl/feat_Frozen_Xbb_hl,color='red')
    #r.plot(feat_Scratch_Xbb_hl/feat_Frozen_Xbb_hl,color='maroon')
    #r.set_ylim(0,5)
    #r.set_xlim(0.6,1)
    #ax.set_xlim(0.6,1)
    #ax.set_ylim(1,feat_Finetune_Xbb_hl[int(len(tpr_common)*0.6)])
    ax.legend()
    ax.set_xlabel(f'{feat_name} [GeV]',loc='right')
    if density: plt.savefig(f'plots/SB_{seff}_{feat_name}_{sample}_norm{mess}.png')  
    else: plt.savefig(f'plots/SB_{seff}_{feat_name}_{sample}{mess}.png') 

def plot_QCD_rej_vs_jmass(h5fw,sample='1',feat_name='sdmass',jetnumber=0, seff=0.7,bins=np.linspace(0,200,40),density=False,log=True,mess=''):
    tpr_common = np.linspace(0,1,10000)
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    #r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    #plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title('Bkg rejection all mass points')
    ax.set_ylabel('Bkg_rej')
    if log: ax.semilogy()
    #r.set_xlabel('signal efficiency')

    feat = h5fw['Finetune_Xbb_hl'][f'{sample}'][f'{feat_name}'][:,jetnumber]  
    #score_Scratch_Xbb_hl =h5fw['Scratch_Xbb_hl'][f'{sample}']['evt_score'][:]  
    #score_Frozen_Xbb_hl = h5fw['Frozen_Xbb_hl'][f'{sample}']['evt_score'][:]
    score_Finetune_Xbb_hl =h5fw['Finetune_Xbb_hl'][f'{sample}']['evt_score'][:]
    label =h5fw['Finetune_Xbb_hl'][f'{sample}']['evt_label'][:]
    Xbb_label =h5fw['Finetune_Xbb_hl'][f'{sample}']['Xbb_label'][:,jetnumber]
    Xbb_score =h5fw['Finetune_Xbb_hl'][f'{sample}']['Xbb_score'][:,jetnumber]    
    print(Xbb_score)
    weights=h5fw['Finetune_Xbb_hl'][f'{sample}']['weights'][:] 
    if mess in ['noW','noWall']: weights=np.ones(len(weights))
    #fpr_Scratch_Xbb_hl, tpr_Scratch_Xbb_hl, thresholds_Scratch_Xbb_hl = roc_curve(label,score_Scratch_Xbb_hl)
    #fpr_Frozen_Xbb_hl, tpr_Frozen_Xbb_hl, thresholds_Frozen_Xbb_hl = roc_curve(label,score_Frozen_Xbb_hl)
    fpr_Finetune_Xbb_hl, tpr_Finetune_Xbb_hl, thresholds_Finetune_Xbb_hl = roc_curve(Xbb_label,Xbb_score)
    #cut_Scratch_Xbb_hl = np.interp(tpr_common,tpr_Scratch_Xbb_hl,thresholds_Scratch_Xbb_hl)[int(len(tpr_common)*seff)]
    #cut_Frozen_Xbb_hl = np.interp(tpr_common,tpr_Frozen_Xbb_hl,thresholds_Frozen_Xbb_hl)[int(len(tpr_common)*seff)]
    cut_Finetune_Xbb_hl = np.interp(tpr_common,tpr_Finetune_Xbb_hl,thresholds_Finetune_Xbb_hl)[int(len(tpr_common)*seff)]
    print(cut_Finetune_Xbb_hl)
    ax.hist(feat,density=density,bins=b,weights=1*weights*Xbb_label,histtype='step',color='red',label='Hbb')
    ax.hist(feat,density=density,bins=b,weights=1*weights*(Xbb_score>=cut_Finetune_Xbb_hl)*Xbb_label,histtype='step',linestyle='dashed',color='indianred',label=f'Hbb {seff*100}% Xbb-wp')
    ax.hist(feat,density=density,bins=b,weights=1*weights*(Xbb_label==0),histtype='step',color='blue',label='QCD')
    ax.hist(feat,density=density,bins=b,weights=1*weights*(Xbb_score>=cut_Finetune_Xbb_hl)*(Xbb_label==0),histtype='step',linestyle='dashed',color='blue',label=f'QCD {seff*100}% Xbb-wp')
    ax.legend()
    ax.set_xlabel(f'{feat_name} [GeV]',loc='right')
    if density: plt.savefig(f'plots/QCD_{seff}_{feat_name}_{sample}_norm{mess}.png')  
    else: plt.savefig(f'plots/QCD_{seff}_{feat_name}_{sample}{mess}.png')
if __name__ == '__main__':    

    filelist = '/u/mvigl/public/run/SB_samples.txt'
    tpr_common = np.linspace(0,1,10000)
    #with h5py.File(f'/u/mvigl/public/run/scores_SB.h5', 'w') as out_file: 
    #    for model in ['Finetune_Xbb_hl']:#['Scratch_Xbb_hl','Frozen_Xbb_hl','Finetune_Xbb_hl']:
    #        group = out_file.create_group(f'{model}')
    #        for sample in ['1']:#['0_0001','0_001','0_01','0_1','1']:
    #            in_dir = f'/u/mvigl/public/run/{model}/scores/{sample}'
    #            feat_dir = f'/ptmp/mvigl/H5_samples_full/'
    #            Xbb_dir = f'/u/mvigl/public/run/Xbb_scores_test/'
    #            evt_score,evt_label,evt_mass,mass,sdmass,Xbb_score,Xbb_label,weights = get_data(in_dir,filelist,feat_dir,Xbb_dir)
    #            Xbb_score=np.nan_to_num(Xbb_score)
    #            top_two_indices = np.argsort(Xbb_score, axis=1)[:, -2:]  # highest Xbb scores
    #            print(top_two_indices)
    #            Xbb_score = np.take_along_axis(Xbb_score, top_two_indices, axis=1)
    #            print(mass)
    #            mass = np.take_along_axis(mass, top_two_indices, axis=1)
    #            print(mass)
    #            sdmass = np.take_along_axis(sdmass, top_two_indices, axis=1)
    #            Xbb_label = np.take_along_axis(Xbb_label, top_two_indices, axis=1)
    #            fpr, tpr, thresholds = roc_curve(evt_label,evt_score)
    #            Auc = auc(fpr,tpr)
    #            fpr = np.interp(tpr_common,tpr,fpr)
    #            sub_group = group.create_group(f'{sample}')
    #            sub_group.create_dataset('evt_score', data=evt_score.reshape(-1))
    #            sub_group.create_dataset('evt_label', data=evt_label.reshape(-1),dtype='i4')
    #            sub_group.create_dataset('evt_mass', data=evt_mass.reshape(-1),dtype='i4')
    #            sub_group.create_dataset('Xbb_score', data=Xbb_score.reshape((-1,2)))
    #            sub_group.create_dataset('Xbb_label', data=Xbb_label.reshape((-1,2)),dtype='i4')
    #            sub_group.create_dataset('mass', data=mass.reshape((-1,2)))
    #            sub_group.create_dataset('sdmass', data=sdmass.reshape((-1,2)))
    #            sub_group.create_dataset('weights', data=weights.reshape(-1))
    #            sub_group.create_dataset('auc', data=Auc)
    #            sub_group.create_dataset('fpr', data=fpr.reshape(-1))
    #            sub_group.create_dataset('tpr', data=tpr_common.reshape(-1))
    #for sample in ['0_0001']:
    #    in_dir = f'/u/mvigl/public/run/Scratch_Xbb_hl/scores/{sample}'
    #    feat_dir = f'/ptmp/mvigl/H5_samples_full/'
    #    Xbb_dir = f'/u/mvigl/public/run/Xbb_scores_test/'
    #    evt_score,evt_label,evt_mass,mass,sdmass,Xbb_score = get_data(in_dir,filelist,feat_dir,Xbb_dir)
    #    print(Xbb_score.shape)
    #    print(mass.shape)
    #    print(sdmass.shape)

    with h5py.File('/u/mvigl/public/run/scores_SB.h5','r') as h5fw :
        b=np.linspace(80,200,51)
        plot_bkg_rej_vs_jmass(  h5fw,
                                sample='1',
                                feat_name='mass',
                                jetnumber=0, 
                                seff=0.9,
                                bins=b,
                                density=False,
                                log=True,
                                mess=''
                            )

        plot_QCD_rej_vs_jmass(  h5fw,
                                sample='1',
                                feat_name='mass',
                                jetnumber=0, 
                                seff=0.9,
                                bins=b,
                                density=False,
                                log=True,
                                mess=''
                            )          
                            
                                      
    #masses = [600,1000,1200,1400,1600,1800,2000,2500,3000,4000,4500]
    #with h5py.File('/u/mvigl/public/run/scores_SB.h5','r') as h5fw : 
    #    plot_all_masses(masses,seff=0.9,metric='auc')
    #    plot_all_masses(masses,seff=0.9,metric='bkg_rej')
    #    plot_auc(h5fw,mass=0)  
    #    plot_bkg_rej(sample='1')    
    #    plot_bkg_rej(sample='0_1')         
    #    plot_bkg_rej(sample='0_01')          
    #    plot_bkg_rej_seff(mass=0,seff=0.9)
    #    for mass in masses:
    #        plot_auc(h5fw,mass=mass)
    #        plot_bkg_rej_seff(mass=mass,seff=0.9)


    