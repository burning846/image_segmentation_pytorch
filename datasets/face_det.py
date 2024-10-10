import cv2
import torch
import numpy as np

from datasets.base_dataset import BaseDataset
from utils.data_utils import load_csv

class FaceDetectionDataset(BaseDataset):
    def __init__(self, data_path, preproc=None):
        super(FaceDetectionDataset, self).__init__()

        self.data = load_csv(data_path)
        self.preproc = preproc


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = cv2.imread(row['local_image_path'])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox = [eval(row['bbox'])]
        bbox = np.array(bbox).astype(np.float32)
        ldmk_5_points = [eval(row['ldmk_5_points'])]
        ldmk_5_points = np.array(ldmk_5_points).astype(np.float32)

        print(img.shape, bbox, ldmk_5_points)

        if self.preproc is not None:
            img, bbox, ldmk_5_points = self.preproc(img, bbox, ldmk_5_points)

        return {
            'image': torch.from_numpy(img).float()
        },{
            'bbox': bbox,
            'ldmk_5_points': ldmk_5_points,
        }

if __name__ == '__main__':
    import os
    from datasets.preprocess import Preproc
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dataset = FaceDetectionDataset('data/face/celeba/celeba_v1_test.csv', Preproc(640, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], context='train'))
    for idx in range(20):
        x, y = dataset[idx]
        print(x['image'].shape, y['bbox'], y['ldmk_5_points'])
        img = x['image'].numpy().transpose(1, 2, 0)
        img = (img * std + mean) * 255
        img = img.clip(0, 255).astype(np.uint8)
        height, width = img.shape[:2]

        for bbox in y['bbox']:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for ldmk in y['ldmk_5_points']:
            for i in range(0, 10, 2):
                x, y = ldmk[i], ldmk[i+1]
                x, y = x * width, y * height
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        # img = np.concatenate([x['image'], x['positive'], x['negative']], axis=1)

        # print(img)
        os.makedirs('test', exist_ok=True)
        cv2.imwrite(f'./test/{idx}.jpg', img)
        # break