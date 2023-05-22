# Finetune_hep

All scripts should be run from a 'run' dir outside of source run/my_run as 

````
 python ../../source/scripts/run.py --default --subset --model 'mlpHlXbb'
````
to get loss info in comet add

````
--ws <'my_workspace'> --api_key <'my_api_key'>
````

output:

models/*.pt

plots/*.png
