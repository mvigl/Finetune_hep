from Finetune_hep.python import ParT
from Finetune_hep.python import ParT_hlf
import yaml

def get_model(config_file,concatenate_hlf=False, **kwargs):

    with open(config_file) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader) 
    
    if concatenate_hlf:
        model = ParT_hlf.get_model(data_config,**kwargs)  
    else:
        model = ParT.get_model(data_config,**kwargs)      
    
    return model