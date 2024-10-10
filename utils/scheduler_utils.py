import torch

def build_scheduler(config, optimizer):
    if config['train']['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['step_size'], gamma=config['train']['gamma'])
    elif config['train']['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['T_max'], eta_min=config['train']['eta_min'])

    return scheduler