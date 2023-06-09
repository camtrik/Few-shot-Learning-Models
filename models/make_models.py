import torch 

models = {}
def register_model(name):
    
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

import torch 

def make_model(name, **kwargs):
    """
    Given a model name and optional keyword arguments, 
    returns an instance of the corresponding registered model class.
    """
    if name is None:
        return None
    if name not in models:
        raise ValueError(f"Unrecognized model name '{name}'. Available models are: {list(models.keys())}")
    model = models[name](**kwargs)
    if torch.cuda.is_available():
        model.cuda()
    return model 



