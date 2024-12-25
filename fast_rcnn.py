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


class RefixLoss(nn.Module):
    '''
    Adapted from https://github.com/CoinCheung/pytorch-loss
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-1):
        super(RefixLoss, self).__init__()
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
            lb_smooth_one_hot = torch.empty_like(logits).fill_(lb_neg)\
                .scatter_(1, label.unsqueeze(1), lb_pos).detach()
            lb_one_hot = torch.empty_like(logits).fill_(0)\
                .scatter_(1, label.unsqueeze(1), 1).detach()

        logs = self.log_softmax(logits)
        probs = F.softmax(logits, dim=1)
        # 每个样本的最大预测概率与对应的索引
        max_probs_per_sample, max_idx_per_sample = torch.max(probs, dim=-1)

        # 获取高置信度最小阈值和低置信度最大阈值
        high_threshold, low_threshold = self.gmm_three_policy(max_probs_per_sample)

        # 创建置信度掩码
        # 47.8
        high_confidence_mask = max_probs_per_sample > high_threshold
        # medium_confidence_mask = (max_probs_per_sample >= low_threshold) & (max_probs_per_sample <= high_threshold)
        # low_confidence_mask = max_probs_per_sample < low_threshold
        medium_confidence_mask = max_probs_per_sample <= high_threshold

        high_weights = self.get_weight(logits, high_confidence_mask)
        # low_weights = self.get_weight(logits, low_confidence_mask)
        # print("high_weights: ", high_weights)
        # medium_weight = self.get_weight(logits, medium_confidence_mask)
        # print("low_weights: ", low_weights)

        # 初始化损失
        loss = torch.zeros_like(max_probs_per_sample)
        # 计算高置信度样本的交叉熵损失
        high_confidence_loss = -torch.sum(logs[high_confidence_mask] * lb_one_hot[high_confidence_mask], dim=1)
        # 将高置信度样本的损失赋值给对应位置
        loss[high_confidence_mask] = high_confidence_loss * high_weights
        # 计算中置信度样本的余弦相似度损失
        medium_confidence_loss = -torch.sum(logs[medium_confidence_mask] *
                                            lb_smooth_one_hot[medium_confidence_mask], dim=1)
        # 将中置信度样本的损失赋值给对应位置
        loss[medium_confidence_mask] = medium_confidence_loss
        # 计算低置信度样本的交叉熵损失
        # low_confidence_loss = -torch.sum(logs[low_confidence_mask] *
        #                                  lb_one_hot[low_confidence_mask], dim=1)
        # 计算低置信度样本的KL散度损失
        # low_confidence_loss = F.kl_div(logs[low_confidence_mask], lb_one_hot[low_confidence_mask],
        #                                reduction='none').sum(dim=1)
        # # 将低置信度样本的损失赋值给对应位置
        # loss[low_confidence_mask] = low_confidence_loss * low_weights
        # 将忽略位置的损失置为0
        loss[ignore] = 0
        # 如果降维方式reduction为'mean'，则计算平均损失
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        # 如果降维方式reduction为'sum'，则将损失值求和
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_weight(self, logits, relevant_mask, epsilon=1e-6):
        relevant_logits = logits[relevant_mask]
        relevant_probs = F.softmax(relevant_logits, dim=1)
        # 取出每个样本的最大概率值和对应的索引
        max_probs_per_sample, max_idx_per_sample = torch.max(relevant_probs, dim=1)
        mean = torch.mean(max_probs_per_sample)
        var = torch.var(max_probs_per_sample, unbiased=False)
        weight = torch.ones_like(relevant_probs)
        lambda_max = 1.0
        weight = lambda_max * torch.exp(
            -((max_probs_per_sample - mean) ** 2) / (2 * var + epsilon))

        return weight


    def gmm_three_policy(self, scores):
        # 首先处理Tensor，确保它在CPU上，并且转换为numpy数组
        if isinstance(scores, torch.Tensor):
            if scores.requires_grad:
                scores = scores.detach()
            if scores.is_cuda:
                scores = scores.cpu()
            scores = scores.numpy()

        # 移除异常值
        # q25, q75 = np.percentile(scores, [25, 75])
        # cut_off = iqr(scores)
        # lower_bound, upper_bound = q25 - cut_off, q75 + cut_off
        # scores_clean = scores[(scores > lower_bound) & (scores < upper_bound)]
        # z_scores = stats.zscore(scores)
        # lower_bound, upper_bound = -3, 3  # 标准Z-score界限
        # scores_clean = scores[(z_scores > lower_bound) & (z_scores < upper_bound)]
        scores_clean = scores

        # 如果处理后的数据仍不足以进行GMM拟合，使用中位数作为阈值
        if len(scores_clean) < 4:
            median = np.percentile(scores_clean, 50)
            return median, median

        # 为GMM拟合准备数据
        scores_clean = scores_clean.reshape(-1, 1)
        # Initialize GMM with three components using more robust statistics
        median = np.median(scores_clean)
        means_init = [[np.min(scores_clean)], [median], [np.max(scores_clean)]]
        weights_init = [1 / 3] * 3  # Equal weights for three components
        precisions_init = [[[1.0]], [[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            n_components=3,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init
        )
        gmm.fit(scores_clean)
        gmm_assignment = gmm.predict(scores_clean)

        # Calculate thresholds based on GMM results
        min_positive = np.min(scores_clean[gmm_assignment == 2]) if (gmm_assignment == 2).any() else np.inf
        max_negative = np.max(scores_clean[gmm_assignment == 0]) if (gmm_assignment == 0).any() else -np.inf

        # Ensure that high threshold is actually higher than the low threshold
        high_thr = max(min_positive, median)  # Use median as a fallback
        low_thr = min(max_negative, median)  # Use median as a fallback

        if low_thr > high_thr:
            high_thr, low_thr = low_thr, high_thr  # Swap if necessary

        return high_thr, low_thr


    # def gmm_three_policy(self, scores):
    #     if len(scores) < 4:
    #         return np.percentile(scores, 50), np.percentile(scores, 50)  # Use median if insufficient data
    #     if isinstance(scores, torch.Tensor):
    #         scores = scores.detach().cpu().numpy()
    #     if len(scores.shape) == 1:
    #         scores = scores[:, np.newaxis]
    #
    #     # Initialize GMM with three components using more robust statistics
    #     median = np.median(scores)
    #     means_init = [[np.min(scores)], [median], [np.max(scores)]]
    #     weights_init = [1 / 3] * 3  # Equal weights for three components
    #     precisions_init = [[[1.0]], [[1.0]], [[1.0]]]
    #     gmm = skm.GaussianMixture(
    #         n_components=3,
    #         weights_init=weights_init,
    #         means_init=means_init,
    #         precisions_init=precisions_init
    #     )
    #     gmm.fit(scores)
    #     gmm_assignment = gmm.predict(scores)
    #
    #     # Calculate thresholds based on GMM results
    #     min_positive = np.min(scores[gmm_assignment == 2]) if (gmm_assignment == 2).any() else np.inf
    #     max_negative = np.max(scores[gmm_assignment == 0]) if (gmm_assignment == 0).any() else -np.inf
    #
    #     # Ensure that high threshold is actually higher than the low threshold
    #     high_thr = max(min_positive, median)  # Use median as a fallback
    #     low_thr = min(max_negative, median)  # Use median as a fallback
    #
    #     if low_thr > high_thr:
    #         high_thr, low_thr = low_thr, high_thr  # Swap if necessary
    #
    #     return high_thr, low_thr


    def gmm_policy(self, scores, given_gt_thr=0.2, policy='high'):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.

        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        # q1, q3 = np.percentile(scores, [25, 75])
        # means_init = [[q1], [q3]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (scores >= scores[indx]).squeeze()
                # print("正样本的所有分数值:", scores[pos_indx])  # 打印正样本的所有分数值
                # pos_thr = scores[pos_indx].mean()  # 使用平均值代替最小值
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)  # 你可以选择是否使用这行代码来保证阈值不低于给定的阈值
            else:
                pos_thr = given_gt_thr

        elif policy == 'middle':
            # For modified middle policy, calculate the average score of the most likely negative bbox
            if (gmm_assignment == 0).any():
                neg_scores = scores[gmm_assignment == 0]
                pos_thr = float(np.mean(neg_scores))
            else:
                pos_thr = given_gt_thr

        return pos_thr


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
    '''
    Adapted from https://github.com/CoinCheung/pytorch-loss
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-1):
        super(DynamicLabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        # LogSoftmax其实就是对softmax的结果进行log，即Log(Softmax(x))
        # dim=1：对每一行的所有元素进行softmax运算，并使得每一行所有元素和为1
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        # overcome ignored label
        # 将输入转换为float类型，以避免产生NaN值
        logits = input.float()  # use fp32 to avoid nan
        # 使用torch.no_grad()上下文管理器，禁止对计算梯度的操作进行跟踪
        with torch.no_grad():
            num_classes = logits.size(1)
            label = target.clone().detach()
            # 通过比较目标标签和要忽略的索引，得到一个布尔张量ignore，表示哪些位置应该被忽略。bool
            ignore = label.eq(self.lb_ignore)
            # 计算非忽略位置的数量n_valid
            n_valid = ignore.eq(0).sum()
            # 将目标标签中的忽略位置设置为0
            label[ignore] = 0
            # 根据平滑因子和类别数量，计算平滑后的标签分布，lb_pos表示非忽略位置的标签权重，lb_neg表示忽略位置的标签权重
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            # torch.empty_like(logits)：创建一个与logits具有相同形状的空张量;
            # fill_(lb_neg)：使用lb_neg的值填充整个张量。fill_()是一个原地操作，将张量的所有元素都设置为相同的值lb_neg
            # detach()：将张量从计算图中分离出来，返回一个新的张量;
            # scatter_(1, label.unsqueeze(1), lb_pos)：使用平滑后的标签分布lb_pos，根据label的值在第1维进行索引填充;
            # label.unsqueeze(1)将label张量的形状从 (batch_size,) 转换为 (batch_size, 1)，以适应scatter_()的要求
            # 1表示在第1维（列维度）上进行索引填充; lb_pos是要填充的值，非忽略位置的标签权重
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg)\
                .scatter_(1, label.unsqueeze(1), lb_pos).detach()

        # 计算交叉熵损失部分
        # 对logits进行log softmax操作，得到每个类别的概率分布
        logs = self.log_softmax(logits)
        dynamic_weight = self.get_weight(input)
        # 计算交叉熵损失，将log softmax的输出和平滑的one-hot标签分布相乘，然后在第1维上求和
        loss = -torch.sum(logs * lb_one_hot, dim=1)

        # 动态权重应用到整体损失上
        loss = loss * dynamic_weight

        # 将忽略位置的损失置为0，以确保在计算损失时不会被考虑
        loss[ignore] = 0
        # 如果降维方式reduction为'mean'，则计算平均损失，即将损失值求和并除以有效样本数n_valid
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        # 如果降维方式reduction为'sum'，则将损失值求和，即不除以有效样本数
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

    @staticmethod
    def get_weight(input):
        num_classes = input.size(1)
        x = 1e-6
        logits = input.float()
        probs_logits = F.softmax(logits, dim=1)
        # tensor(N, )每个样本最大预测概率值以及对应的类别索引0-C
        max_pred_b, max_idx_b = torch.max(probs_logits, dim=1)
        # tensor(C, )每个类别最大预测概率值以及对应的样本索引0-N
        max_pred_c, max_idx_c = torch.max(probs_logits, dim=0)
        u_t = torch.mean(max_pred_c, dim=0)
        # 创建一个和 max_pred_b 形状相同的张量，所有元素都是 u_t 的值
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
