import torch

def build_optimizer(config, model, criterion_params):
    params = list(model.parameters()) + criterion_params
    optimizer = None
    if config['train']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config['train']['learning_rate'])
    elif config['train']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params, lr=config['train']['learning_rate'], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config['train']['optimizer']}")
    
    return optimizer