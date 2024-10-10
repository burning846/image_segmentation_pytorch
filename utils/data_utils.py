import pandas as pd
import torch
import torch.utils.data as data

def load_csv(csv_path):
    return pd.read_csv(csv_path)

def build_dataset(config, context='train', preproc=None):
    dataset_name = config['data']['dataset']
    if context == 'train':
        shuffle = True
    else:
        shuffle = False

    if dataset_name == 'FaceDetDataset':
        from datasets.face_det import FaceDetectionDataset
        dataset = FaceDetectionDataset(config['data'][f'{context}_data_path'], preproc=preproc)
        dataloader = data.DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=shuffle, num_workers=config['environment']['num_workers'])

    elif dataset_name == 'CardSegmentDataset':
        from datasets.card_seg import CardSegmentationDataset
        dataset = CardSegmentationDataset(config['data'][f'{context}_data_path'], config['image_processing'][context])
        dataloader = data.DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=shuffle, num_workers=config['environment']['num_workers'])
        
    else:
        raise ValueError('Unknown dataset: {}'.format(config['data']['dataset']))

    return dataset, dataloader