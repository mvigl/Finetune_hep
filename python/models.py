from Finetune_hep.python import ParT
from Finetune_hep.python import ParT_hlf
import yaml

def full_model(config_file, **kwargs):

    with open(config_file) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader) 
    
    if data_config['inputs']['hlf']['concatenate']:
        model = ParT_hlf.get_model(data_config,**kwargs)  
    else:
        model = ParT.get_model(data_config,**kwargs)      
    
    return model

def head(config_file, **kwargs):

    with open(config_file) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader) 
    
    if data_config['inputs']['hlf']['concatenate']:
        model = ParT_hlf.get_model(data_config,**kwargs).head  
    else:
        model = ParT.get_model(data_config,**kwargs).head   
    
    return model