import os
import sys

for mass in [600,1000,1200,1400,1600,1800,2000,2500,3000,4000,4500]:
        names = f'samples_sig_{mass}.txt'
        i=0
        with open(names) as f:
                for line in f:
                        fname = line.strip()   
                        cmd=f'python3 python/H5_full.py --fname {fname} --mass {mass} --mess {str(i)}'
                        print(cmd)
                        os.system(cmd)
                        i+=1


for mass in [0]:
        names = f'samples_bkg.txt'
        i=0
        with open(names) as f:
                for line in f:
                        fname = line.strip()   
                        cmd=f'python3 python/H5_full.py --fname {fname} --mass {mass} --mess {str(i)} --bkg'
                        print(cmd)
                        os.system(cmd)
                        i+=1

