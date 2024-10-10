from torchvision import transforms
from utils.misc import umeyama
import cv2
import numpy as np
import random

from utils.bbox_utils import matrix_iof
from utils.constants import FACE_LANDMARK_TEMPLATE, TARGET_SIZE_TEMPLATE
from datasets.preprocess import Preproc

def build_transform(config):
    transform_list = [transforms.ToPILImage()]
    if 'horizontal_flip' in config:
        transform_list.append(transforms.RandomHorizontalFlip(p=config['horizontal_flip']))
    if 'rotation' in config:
        transform_list.append(transforms.RandomRotation(degrees=config['rotation']))
    if 'colorjitter' in config:
        transform_list.append(transforms.ColorJitter(brightness=config['colorjitter']['brightness'], contrast=config['colorjitter']['contrast'], saturation=config['colorjitter']['saturation'], hue=config['colorjitter']['hue']))
    if 'resize' in config:
        transform_list.append(transforms.Resize((config['resize']['image_height'], config['resize']['image_width'])))
    if 'totensor'in config and config['totensor']:
        transform_list.append(transforms.ToTensor())
    if 'normalization' in config:
        transform_list.append(transforms.Normalize(mean=config['normalization']['mean'], std=config['normalization']['std']))

    return transforms.Compose(transform_list)

def build_preproc(config, context):
    preproc = Preproc(config['resize']['image_height'], config['normalization']['mean'], context=context)
    return preproc

def get_target_ldmk_by_size(target_size):
    origin_size = TARGET_SIZE_TEMPLATE
    origin_lmdk = FACE_LANDMARK_TEMPLATE

    target_ldmk = []
    for ldmk in origin_lmdk:
        x = ldmk[0] / origin_size[0] * target_size[0]
        y = ldmk[1] / origin_size[1] * target_size[1]
        target_ldmk.append([x, y])
    
    return target_ldmk
    
def align_face(image, image_ldmk, target_size, target_ldmk=None):
    '''
    target_size: (w, h)
    '''
    if target_ldmk is None:
        target_ldmk = get_target_ldmk_by_size(target_size)

    image_ldmk = np.array(image_ldmk).astype(np.float32).reshape(-1, 2)
    target_ldmk = np.array(target_ldmk).astype(np.float32).reshape(-1, 2)
    
    # 计算仿射变换矩阵
    M = umeyama(image_ldmk, target_ldmk, scale=1.0)[0:2]

    # 应用仿射变换
    aligned_image = cv2.warpAffine(image, M, target_size, flags=cv2.INTER_CUBIC)

    return aligned_image
