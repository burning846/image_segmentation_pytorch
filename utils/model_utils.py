import torch

def load_model(config):
    if config['model']['name'] == 'retinaface':
        from models.retinaface import RetinaFace
        config['model']['name'] = config['model']['backbone']
        model = RetinaFace(config['model'])

    if config['model']['name'] == 'unet':
        from models.unet import UNet
        model = UNet(config['model']['in_channels'], config['model']['out_channels'])

    if config['model']['pretrained_weights']:
        model.load_state_dict(torch.load(config['model']['pretrained_weights']), strict=True)

    return model