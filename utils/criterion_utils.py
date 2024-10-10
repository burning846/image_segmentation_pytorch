from collections import OrderedDict
import torch
import numpy as np

from utils.bbox_utils import nms
from utils.misc import bbox_iou

def build_criterion(config, device='cuda'):
    criterions = OrderedDict()
    paramters = list() # 有的loss需要训练权重
    if 'RetinaFaceLoss' in config['train']['loss_function']:
        from losses.face_det_loss import RetinaFaceLoss
        criterions['RetinaFaceLoss'] = {}
        criterions['RetinaFaceLoss']['fn'] = RetinaFaceLoss(
            config['loss']['RetinaFaceLoss']['alpha'],
            config['loss']['RetinaFaceLoss']['beta'],
            config['loss']['RetinaFaceLoss']['gamma'],
            config['loss']['RetinaFaceLoss']['iou_threshold']
        ).to(device)
        criterions['RetinaFaceLoss']['weight'] = config['loss']['RetinaFaceLoss']['weight']
        criterions['RetinaFaceLoss']['save_weights'] = False

    if 'RetinaCardLoss' in config['train']['loss_function']:
        from losses.face_det_loss import RetinaCardLoss
        criterions['RetinaCardLoss'] = {}
        criterions['RetinaCardLoss']['fn'] = RetinaCardLoss(
            config['loss']['RetinaCardLoss']['alpha'],
            config['loss']['RetinaCardLoss']['beta'],
            config['loss']['RetinaCardLoss']['iou_threshold']
        ).to(device)
        criterions['RetinaCardLoss']['weight'] = config['loss']['RetinaCardLoss']['weight']
        criterions['RetinaCardLoss']['save_weights'] = False

    if 'smoothl1loss' in config['train']['loss_function']:
        from losses.smoothl1loss import SmoothL1Loss
        criterions['smoothl1loss'] = {}
        criterions['smoothl1loss']['fn'] = SmoothL1Loss(
            config['loss']['smoothl1loss']['beta']
        ).to(device)
        criterions['smoothl1loss']['weight'] = config['loss']['smoothl1loss']['weight']
        criterions['smoothl1loss']['save_weights'] = False

    return criterions, paramters

def calculate_loss(outputs, labels, criterions):
    loss = 0
    loss_dict = OrderedDict()
    for name, criterion in criterions.items():
        loss_dict[name] = criterion['fn'](outputs, labels)
        loss += loss_dict[name] * criterion['weight']

    return loss, loss_dict

def calculate_accuracy_detection(output, label, anchor, threshold=0.5):
    pred_cls = output['classification']
    pred_bbox = output['bbox_regression']

    gt_boxes = label['bbox']

    batch_size = pred_cls.size(0)

    all_tp = 0
    all_fp = 0
    all_fn = 0

    for i in range(batch_size):

        pred_bbox_i = pred_bbox[i]
        pred_cls_i = pred_cls[i]
        gt_boxes_i = gt_boxes[i]

        # 从模型输出计算候选框
        bboxes = torch.cat([anchors[:, :2] + pred_bbox_i[:, :2] * anchors[:, 2:],
                            anchors[:, 2:] * torch.exp(pred_bbox_i[:, 2:])], 1)
        bboxes[:, :2] -= bboxes[:, 2:] / 2
        bboxes[:, 2:] += bboxes[:, :2]

        scores = pred_cls_i.data.cpu().numpy()[:, 1]

        bboxes, cnt = nms(bboxes.data.cpu().numpy(), scores, threshold)

        gt_boxes_i = gt_boxes_i.data.cpu().numpy()

        tp = 0  # True Positive
        fp = 0  # False Positive
        matched_gt = set()

        for bbox in bboxes:
            
            best_iou = 0
            best_gt_i = -1
            
            for gt_i, gt_box in enumerate(gt_boxes_i):
                iou = bbox_iou(bbox, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_i = gt_i

            if best_iou >= threshold and best_gt_i not in matched_gt:
                tp += 1
                matched_gt.add(best_gt_i)
            else:
                fp += 1

        fn = len(gt_boxes_i) - len(matched_gt)

        all_tp += tp
        all_fp += fp
        all_fn += fn

    accuracy = all_tp / (all_tp + all_fp + all_fn + 1e-6)
    return accuracy, all_tp, all_fp, all_fn
    