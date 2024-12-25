from detectron2.layers import cat
from detectron2.modeling.roi_heads.fast_rcnn import (
    _log_classification_stats,
    FastRCNNOutputLayers
)
from detectron2.structures import Instances
import torch.nn.functional as F
from tllib.modules.loss import LabelSmoothSoftmaxCEV1


import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import iqr
import scipy.stats as stats
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DynamicLabelSmoothSoftmaxCEV2(nn.Module):
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-1):
        super(DynamicLabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, target):
        logit = inputs.float()
        with torch.no_grad():
            num_classes = logit.size(1)
            label = target.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logit).fill_(lb_neg)\
                .scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logit)
        dynamic_weight = self.get_weight(inputs)
        loss = -torch.sum(logs * lb_one_hot, dim=1).to(device)
        loss = loss * dynamic_weight
        loss[ignore] = 0

        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

    @staticmethod
    def get_weight(inputs):
        num_classes = inputs.size(1)
        x = 1e-6
        n_sigma = 2
        logit = inputs.float()
        # (BatchSize, )
        probs_logits = F.softmax(logit.detach(), dim=-1)
        max_probs, max_idx = probs_logits.max(dim=-1)

        # 计算max_probs的全局均值和方差
        mean = max_probs.mean()
        var = max_probs.var()

        threshold = []
        for i in range(num_classes):
            prob = max_probs[max_idx == i]
            if len(prob) > 1:
                mean_t = torch.mean(prob)
                threshold.append(mean_t)
            else:
                threshold.append(mean)

        # 使用 torch.stack 来创建张量，确保每个元素都是张量
        threshold_tensor = torch.stack(threshold).to(device)
        u_t = threshold_tensor[max_idx]
        u_t = u_t.to(device)
        max_probs = max_probs.to(device)

        weight = torch.ones_like(max_probs)
        lambda_max = 1.0
        update = max_probs < u_t
        if torch.any(update):
            weight[update] = lambda_max * torch.exp(
                -((max_probs[update] - mean) ** 2) / (2 * var + x))

        return weight


# version 1: use torch.autograd
class DynamicLabelSmoothSoftmaxCEV1(nn.Module):
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-1):
        super(DynamicLabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):   
        logits = input.float()  # use fp32 to avoid nan    
        with torch.no_grad():
            num_classes = logits.size(1)
            label = target.clone().detach()  
            ignore = label.eq(self.lb_ignore)      
            n_valid = ignore.eq(0).sum()         
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg)\
                .scatter_(1, label.unsqueeze(1), lb_pos).detach()
   
        logs = self.log_softmax(logits)
        dynamic_weight = self.get_weight(input)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss = loss * dynamic_weight
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

    @staticmethod
    def get_weight(input):
        num_classes = input.size(1)
        x = 1e-6
        logits = input.float()
        probs_logits = F.softmax(logits, dim=1)
        max_pred_b, max_idx_b = torch.max(probs_logits, dim=1)
        max_pred_c, max_idx_c = torch.max(probs_logits, dim=0)
        u_t = torch.mean(max_pred_c, dim=0)
        u_t_tensor = torch.full_like(max_pred_b, u_t.item())
        diff_squared = (max_pred_c - u_t) ** 2
        variance_t = torch.mean(diff_squared, dim=0)

        weight = torch.ones_like(max_pred_b)
        lambda_max = 1.0
        update = max_pred_b < u_t_tensor
        if torch.any(update):
            weight[update] = lambda_max * torch.exp(
                -((max_pred_b[update] - u_t_tensor[update]) ** 2) / (2 * variance_t + x))

        return weight

class DynamicLabelSmoothSoftmaxCEV3(nn.Module):
    '''
    Optimized version of DynamicLabelSmoothSoftmaxCEV1.
    Increases the loss for less confident predictions to focus on hard examples.
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-1, gamma=2.0):
        super(DynamicLabelSmoothSoftmaxCEV3, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.gamma = gamma  # Focusing parameter for focal loss
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        logits = input.float()
        num_classes = logits.size(1)

        with torch.no_grad():
            # Label smoothing
            smooth_pos = 1.0 - self.lb_smooth
            smooth_neg = self.lb_smooth / num_classes

            lb_one_hot = torch.full_like(logits, smooth_neg)
            lb_one_hot.scatter_(1, target.unsqueeze(1), smooth_pos)

            # Handle ignore index
            if self.ignore_index >= 0:
                ignore = target.eq(self.ignore_index)
                lb_one_hot[ignore] = 0
                n_valid = ignore.eq(0).sum()
            else:
                n_valid = target.numel()

        # Compute log probabilities
        logs = self.log_softmax(logits)

        # Compute the standard cross-entropy loss
        loss = -torch.sum(logs * lb_one_hot, dim=1)

        # Compute the dynamic weights (focal loss style)
        probs = torch.exp(logs)
        pt = torch.sum(probs * lb_one_hot, dim=1)  # Model's estimated probability for the true class
        focal_weight = (1 - pt).pow(self.gamma)

        # Apply the focal weight
        loss = focal_weight * loss

        # Handle ignore index in loss
        if self.ignore_index >= 0:
            loss[ignore] = 0

        # Reduction
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def label_smoothing_cross_entropy(input, target, *, reduction="mean", **kwargs):
    """
    Same as `tllib.modules.loss.LabelSmoothSoftmaxCEV1`, but returns 0 (instead of nan)
    for empty inputs.
    """
    # 检查目标张量target是否为空，并且reduction参数是否设置为"mean"。
    # 如果两个条件都满足，说明输入为空，函数会返回一个值为0的张量，这是为了确保梯度能够正确传播
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    return LabelSmoothSoftmaxCEV1(reduction=reduction, **kwargs)(input, target)
    # return RefixLoss(reduction=reduction, **kwargs)(input, target)
    # return DynamicLabelSmoothSoftmaxCEV2(reduction=reduction, **kwargs)(input, target)
    # return DynamicLabelSmoothSoftmaxCEV1(reduction=reduction, **kwargs)(input, target)


def label_smoothing_cross_entropy1(input, target, *, reduction="mean", **kwargs):
    """
    Same as `tllib.modules.loss.LabelSmoothSoftmaxCEV1`, but returns 0 (instead of nan)
    for empty inputs.
    """
    # 检查目标张量target是否为空，并且reduction参数是否设置为"mean"。
    # 如果两个条件都满足，说明输入为空，函数会返回一个值为0的张量，这是为了确保梯度能够正确传播
    if target.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    # return LabelSmoothSoftmaxCEV1(reduction=reduction, **kwargs)(input, target)
    # return RefixLoss(reduction=reduction, **kwargs)(input, target)
    return DynamicLabelSmoothSoftmaxCEV1(reduction=reduction, **kwargs)(input, target)
    # return DynamicLabelSmoothSoftmaxCEV3(reduction=reduction, **kwargs)(input, target)
    # return DynamicLabelSmoothSoftmaxCEV2(reduction=reduction, **kwargs)(input, target)


class DecoupledFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores

    Replace cross-entropy with label-smoothing cross-entropy
    """

    def losses1(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            返回一个字典，包含不同类型的损失
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        # 从proposals列表中提取gt_classes字段，并使用cat函数将它们连接成一个张量，表示所有提议的真实类别
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        # 调用_log_classification_stats函数，用于记录分类的统计信息
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            # 将提议框的边界框张量连接成一个形状为(N, 4)的张量，并使用cat函数实现
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            # 通过断言检查proposal_boxes张量是否不需要梯度
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            # 根据是否存在gt_boxes字段，将真实边界框的张量连接到gt_boxes张量中
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            # 如果proposals列表为空，则创建一个形状为(0, 4)的空张量赋值给proposal_boxes和gt_boxes
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": label_smoothing_cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


    def losses2(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            返回一个字典，包含不同类型的损失
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        # 从proposals列表中提取gt_classes字段，并使用cat函数将它们连接成一个张量，表示所有提议的真实类别
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        # 调用_log_classification_stats函数，用于记录分类的统计信息
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            # 将提议框的边界框张量连接成一个形状为(N, 4)的张量，并使用cat函数实现
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            # 通过断言检查proposal_boxes张量是否不需要梯度
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            # 根据是否存在gt_boxes字段，将真实边界框的张量连接到gt_boxes张量中
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            # 如果proposals列表为空，则创建一个形状为(0, 4)的空张量赋值给proposal_boxes和gt_boxes
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": label_smoothing_cross_entropy1(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
