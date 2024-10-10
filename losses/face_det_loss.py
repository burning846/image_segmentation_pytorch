import torch
import torch.nn as nn

class RetinaFaceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, iou_threshold=0.5):
        """
        RetinaFace损失函数
        alpha: 分类损失权重
        beta: 边界框损失权重
        gamma: 关键点损失权重
        iou_threshold: 用于匹配正样本的IoU阈值
        """
        super(RetinaFaceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iou_threshold = iou_threshold

        self.cls_loss_fn = nn.CrossEntropyLoss()  # 二元交叉熵损失，用于分类
        self.reg_loss_fn = nn.SmoothL1Loss()       # Smooth L1，用于边界框回归
        self.landmark_loss_fn = nn.SmoothL1Loss()  # Smooth L1，用于关键点回归

    def forward(self, output, label):
        """
        pred_cls: 预测的分类结果 [N, num_anchors, 2]
        pred_bbox: 预测的边界框 [N, num_anchors, 4]
        pred_landmarks: 预测的关键点 [N, num_anchors, 10]，每个点 (x, y) * 5
        anchors: 锚框 [num_anchors, 4]
        gt_boxes: 真实边界框 [N, num_gt_boxes, 4]
        gt_labels: 真实类别 [N, num_gt_boxes]
        gt_landmarks: 真实关键点 [N, num_gt_boxes, 10]

        返回：
        - total_loss: 总损失
        - cls_loss: 分类损失
        - reg_loss: 边界框损失
        - landmark_loss: 关键点损失
        """
        pred_cls = output['classification']
        pred_bbox = output['bbox_regression']
        pred_landmarks = output['ldm_regression']

        anchors = label['anchor']
        gt_boxes = label['bbox']
        gt_landmarks = label['ldmk_5_points']
        
        # anchors = 
        
        batch_size = pred_cls.size(0)

        # 初始化损失
        cls_loss = 0.0
        reg_loss = 0.0
        landmark_loss = 0.0

        for i in range(batch_size):
            # 取出当前batch的预测和真实数据
            pred_cls_i = pred_cls[i]
            pred_bbox_i = pred_bbox[i]
            pred_landmarks_i = pred_landmarks[i]
            gt_boxes_i = gt_boxes[i]
            gt_landmarks_i = gt_landmarks[i]

            # 计算每个anchor与真实框的IoU
            iou_matrix = self.iou(anchors, gt_boxes_i)
            max_iou, matched_gt_indices = iou_matrix.max(dim=1)

            # 分类标签：正样本为1，负样本为0
            target_cls = (max_iou >= self.iou_threshold).float()

            # 边界框标签：正样本与其匹配的真实框计算偏移
            positive_indices = max_iou >= self.iou_threshold
            matched_boxes = gt_boxes_i[matched_gt_indices[positive_indices]]
            reg_targets = self.compute_regression_targets(anchors[positive_indices], matched_boxes)

            # 关键点标签：正样本才计算关键点损失
            matched_landmarks = gt_landmarks_i[matched_gt_indices[positive_indices]]

            # 计算分类损失
            cls_loss += self.cls_loss_fn(pred_cls_i.squeeze(-1), target_cls)

            # 计算边界框回归损失
            if positive_indices.sum() > 0:
                reg_loss += self.reg_loss_fn(pred_bbox_i[positive_indices], reg_targets)

                # 计算关键点损失
                landmark_loss += self.landmark_loss_fn(pred_landmarks_i[positive_indices], matched_landmarks)

        # 计算平均损失
        cls_loss /= batch_size
        reg_loss /= batch_size
        landmark_loss /= batch_size

        # 总损失
        total_loss = self.alpha * cls_loss + self.beta * reg_loss + self.gamma * landmark_loss

        return total_loss, cls_loss, reg_loss, landmark_loss

    @staticmethod
    def iou(box1, box2):
        """
        计算IoU
        box1: [num_anchors, 4]
        box2: [num_gt_boxes, 4]
        返回: IoU矩阵 [num_anchors, num_gt_boxes]
        """
        inter_xmin = torch.max(box1[:, None, 0], box2[:, 0])
        inter_ymin = torch.max(box1[:, None, 1], box2[:, 1])
        inter_xmax = torch.min(box1[:, None, 2], box2[:, 2])
        inter_ymax = torch.min(box1[:, None, 3], box2[:, 3])

        inter_area = (inter_xmax - inter_xmin).clamp(min=0) * (inter_ymax - inter_ymin).clamp(min=0)

        area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        union_area = area_box1[:, None] + area_box2 - inter_area
        iou_matrix = inter_area / union_area

        return iou_matrix

    @staticmethod
    def compute_regression_targets(anchors, matched_boxes):
        """
        计算边界框的回归目标。
        anchors: [num_pos, 4] 正样本的锚框
        matched_boxes: [num_pos, 4] 对应的真实框

        返回: [num_pos, 4] 回归目标
        """
        # 中心点偏移和宽高缩放
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_cx = anchors[:, 0] + 0.5 * anchor_w
        anchor_cy = anchors[:, 1] + 0.5 * anchor_h

        gt_w = matched_boxes[:, 2] - matched_boxes[:, 0]
        gt_h = matched_boxes[:, 3] - matched_boxes[:, 1]
        gt_cx = matched_boxes[:, 0] + 0.5 * gt_w
        gt_cy = matched_boxes[:, 1] + 0.5 * gt_h

        # 计算偏移量和缩放比例
        targets = torch.zeros_like(matched_boxes)
        targets[:, 0] = (gt_cx - anchor_cx) / anchor_w  # dx
        targets[:, 1] = (gt_cy - anchor_cy) / anchor_h  # dy
        targets[:, 2] = torch.log(gt_w / anchor_w)     # dw
        targets[:, 3] = torch.log(gt_h / anchor_h)     # dh

        return targets
