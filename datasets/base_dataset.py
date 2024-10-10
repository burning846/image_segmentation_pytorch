import torch
import numpy as np
from torch.utils.data import Dataset

from utils.image_processing_utils import align_face

class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _align_face(self, image, landmarks):
        landmarks = np.array(landmarks).astype(np.float32)
        aligned_image = align_face(image, landmarks, self.image_size)
        return aligned_image

    def _crop_face(self, image, bbox):
        bbox = np.array(bbox).astype(int)
        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[2] > image.shape[1]:
            bbox[2] = image.shape[1]
        if bbox[3] > image.shape[0]:
            bbox[3] = image.shape[0]
            
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass