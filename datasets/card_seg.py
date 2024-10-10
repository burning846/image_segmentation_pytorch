import cv2
import torch
import numpy as np

from datasets.base_dataset import BaseDataset
from utils.data_utils import load_csv

class CardSegmentationDataset(BaseDataset):
    def __init__(self, data_path, config):
        super(CardSegmentationDataset, self).__init__()

        self.data = load_csv(data_path)
        self.config = config

    def _color_jitter(self, img, brightness=0, contrast=0, saturation=0, hue=0):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = np.array(img, dtype = np.float32)

        # brightness
        if brightness != 0:
            img[:,:,2] = img[:,:,2] + (brightness * 255)
            img[:,:,2][img[:,:,2]>255] = 255
            img[:,:,2][img[:,:,2]<0] = 0

        # contrast
        if contrast > 0:
            img[:,:,2] = (img[:,:,2] - 127) * contrast + 127
            img[:,:,2][img[:,:,2]>255] = 255
            img[:,:,2][img[:,:,2]<0] = 0

        # saturation
        if saturation > 0:
            img[:,:,1] = img[:,:,1] * saturation
            img[:,:,1][img[:,:,1]>255] = 255
            img[:,:,1][img[:,:,1]<0] = 0

        # hue
        if hue != 0:
            img[:,:,0] = (img[:,:,0] + hue * 255) % 180

        img = np.array(img, dtype = np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def _preprocess(self, img, mask):
        h, w = img.shape[:2]
        # flip
        if self.config['horizontal_flip'] and np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # rotation
        if self.config['rotation']:
            angle = np.random.randint(-self.config['rotation'], self.config['rotation'])
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))

        # color jitter
        if self.config['colorjitter']:
            brightness = np.random.uniform(-self.config['colorjitter']['brightness'], self.config['colorjitter']['brightness'])
            contrast = np.random.uniform(1 - self.config['colorjitter']['contrast'], 1 + self.config['colorjitter']['contrast'])
            saturation = np.random.uniform(1 - self.config['colorjitter']['saturation'], 1 + self.config['colorjitter']['saturation'])
            hue = np.random.uniform(-self.config['colorjitter']['hue'], self.config['colorjitter']['hue'])

            img = self._color_jitter(img, brightness, contrast, saturation, hue)

        # crop
        if self.config['crop']:
            crop_h = np.random.randint(int(h * self.config['crop']['min_crop_ratio']), int(h * self.config['crop']['max_crop_ratio']))
            crop_w = np.random.randint(int(w * self.config['crop']['min_crop_ratio']), int(w * self.config['crop']['max_crop_ratio']))
            x = np.random.randint(0, w - crop_w)
            y = np.random.randint(0, h - crop_h)
            img = img[y:y+crop_h, x:x+crop_w]
            mask = mask[y:y+crop_h, x:x+crop_w]
        
        # resize
        if self.config['resize']:
            img = cv2.resize(img, (self.config['resize']['image_width'], self.config['resize']['image_height']))
            mask = cv2.resize(mask, (self.config['resize']['image_width'], self.config['resize']['image_height']))

        # normalization
        img = img.astype(np.float32)
        img = img / 255.0
        img = img.transpose(2, 0, 1)

        mask = mask.astype(np.float32)
        mask = mask / 255.0
        mask = mask[..., None].transpose(2, 0, 1)

        return img, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = cv2.imread(row['local_image_path'])
        
        poly = eval(row['Polygons'])
        
        try:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            poly = [np.array(x).astype(int) for x in poly]
            cv2.fillPoly(mask, poly, 255)    
        except:
            print(poly)
            raise ValueError('Polygons is not valid')

        img, mask = self._preprocess(img, mask)

        return {
            'image': torch.from_numpy(img).float()
        },{
            'mask': torch.from_numpy(mask).float()
        }

if __name__ == '__main__':
    import os
    import yaml
    config = yaml.load(open('configs/train_card.yaml', 'r'), Loader=yaml.FullLoader)
    dataset = CardSegmentationDataset('data/card/tianchi_val.csv', config['image_processing']['train'])
    for idx in range(20):
        x, y = dataset[idx]
        # print(x['image'].shape, y['mask'].shape)
        img = x['image'].numpy().transpose(1, 2, 0)
        img = img * 255
        img = img.clip(0, 255).astype(np.uint8)
        
        mask = y['mask'].numpy().transpose(1, 2, 0)
        mask = mask * 255
        mask = mask.clip(0, 255).astype(np.uint8)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        img = np.concatenate([img, mask], axis=1)

        # print(img)
        os.makedirs('test', exist_ok=True)
        cv2.imwrite(f'./test/{idx}.jpg', img)
        # break