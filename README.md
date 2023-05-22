# Finetune_hep

All scripts should be run from a 'run' dir outside of Finetune_hep run/my_run as 

````
 python ../../Finetune_hep/scripts/run.py --default --subset --model 'mlpHlXbb'
````
to get loss info in comet.ml add

````
--ws <'my_workspace'> --api_key <'my_api_key'>
````

output dir will be created in run/my_run:

models/*.pt

plots/*.png
