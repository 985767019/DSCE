import math
import random
import time
import warnings
import os.path as osp
import argparse
from collections import deque
import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tllib.vision.transforms import ResizeImage
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from detectron2.modeling.box_regression import Box2BoxTransform


from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.modules.regressor import Regressor
from tllib.alignment.mdd import ImageRegressor, RegressionMarginDisparityDiscrepancy
from tllib.alignment.d_adapt.proposal import ProposalDataset, PersistentProposalList, flatten, ExpandCrop

import utils
from d_adaptation.extension.mcc import MinimumClassConfusionLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BoxTransform(nn.Module):
    def __init__(self):
        # 调用父类 nn.Module 的构造函数，确保正确地初始化神经网络模块
        super(BoxTransform, self).__init__()
        # 定义了一个包含四个元素的元组 BBOX_REG_WEIGHTS，其中的值分别表示边界框回归的权重。
        BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
        # 创建一个名为 box_transform 的成员变量，它是一个 Box2BoxTransform 类的实例对象。
        # Box2BoxTransform 是一个用于边界框变换的类，通过传入参数 weights=BBOX_REG_WEIGHTS 来初始化该实例。
        self.box_transform = Box2BoxTransform(weights=BBOX_REG_WEIGHTS)

    def forward(self, pred_delta, gt_classes, proposal_boxes):
        """
        pred_delta 是预测的边界框偏移量，gt_classes 是真实的类别标签，proposal_boxes 是参考的边界框。
        Args:
            - pred_delta: predicted bounding box offset for each classes
            - gt_classes: ground truth classes
            - proposal_boxes: referenced bounding box

        Returns:
            predicted bounding box offset for ground truth classes
            and  predicted bounding box
        """
        # 根据真实的类别标签 gt_classes 计算出一个索引矩阵 gt_class_cols。
        # torch.arange(4, device=device)：创建一个包含从 0 到 3 的整数序列的张量，对应于边界框的四个坐标偏移量
        # gt_classes 是一个包含真实类别标签的张量，形状为 (N,)，其中 N 是样本的数量。
        # gt_classes[:, None] 使用切片操作 [:, None] 将其形状转换为 (N, 1)
        gt_class_cols = 4 * gt_classes[:, None] + torch.arange(4, device=device)
        # 使用索引矩阵 gt_class_cols 从 pred_delta 中选择对应真实类别标签的边界框偏移量，将结果保存到 pred_delta 中。
        pred_delta = torch.gather(pred_delta, dim=1, index=gt_class_cols)
        # 调用 box_transform 对象的 apply_deltas 方法，将选择后的边界框偏移量 pred_delta 应用到
        # 参考边界框 proposal_boxes 上，得到预测的边界框 pred_box。
        pred_box = self.box_transform.apply_deltas(pred_delta, proposal_boxes)
        # 将预测的边界框偏移量 pred_delta 和预测的边界框 pred_box 作为结果返回
        return pred_delta, pred_box


def iou_between(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
    reduction: str = "none"
):
    """Intersections over Union between two boxes"""
    # 将 boxes1 张量按最后一个维度拆分成四个张量 x1、y1、x2、y2
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    # 将 boxes2 张量按最后一个维度拆分成四个张量 x1g、y1g、x2g、y2g
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    # 计算边界框的 x 坐标的最大值，表示两个边界框中对应位置的左上角 x 坐标的较大值
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    # 创建一个与 x1 张量形状相同的全零张量 intsctk，用于存储交集的面积
    intsctk = torch.zeros_like(x1)
    # 创建一个布尔掩码 mask，用于筛选出有效的交集。只有当交集的右下角坐标大于左上角坐标时，交集才是有效的。
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    # 根据掩码 mask，计算有效交集的面积，并将结果保存在 intsctk 张量中
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    # 计算并集的面积。将两个边界框的面积相加，然后减去交集的面积。
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    # 计算交并比（IoU），即将交集的面积除以并集的面积，同时加上一个很小的常数 eps，以避免除以零的情况
    iouk = intsctk / (unionk + eps)

    # ========= enclose box coordinates ========
    # enclose_x1 = torch.min(x1, x1g)
    # enclose_y1 = torch.min(y1, y1g)
    # enclose_x2 = torch.max(x2, x2g)
    # enclose_y2 = torch.max(y2, y2g)
    #
    # # ========= enclose box area ========
    # enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    # # Calculate GIOU
    # giouk = iouk - (enclose_area - unionk) / (enclose_area + eps)

    # Bounding box coordinates
    xmin = torch.min(x1, x1g)
    ymin = torch.min(y1, y1g)
    xmax = torch.max(x2, x2g)
    ymax = torch.max(y2, y2g)

    # Bounding box diagonal length
    c = torch.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)

    # Distance between box centers
    center_distance = torch.sqrt((0.5 * (x2g + x1g) - 0.5 * (x2 + x1)) ** 2
                                 + (0.5 * (y2g + y1g) - 0.5 * (y2 + y1)) ** 2)

    # Calculate Ldis
    L_dis = (center_distance ** 2) / (c ** 2 + eps)
    # Calculate DIoU
    diouk = iouk - L_dis

    # Aspect ratio regularization
    v = 4 / (math.pi ** 2) * \
        ((torch.atan((x2 - x1) / (y2 - y1)) - torch.atan((x2g - x1g) / (y2g - y1g))) ** 2)
    alpha = v / (1 - iouk + v)

    # Calculate CIoU
    ciouk = iouk - L_dis - alpha * v

    w_pred = x2 -x1
    h_pred = y2 -y1
    w_gt = x2g -x1g
    h_gt = y2g - y1g
    L_asp = ((w_gt - w_pred) ** 2) / ((xmax - xmin) ** 2) + ((h_gt - h_pred) ** 2) / ((ymax - ymin) ** 2)
    # Calculate EIoU
    eiouk = iouk - L_dis - L_asp
    # Calculate Focal-EIoU
    focal_eiouk = (iouk ** 0.5) * eiouk

    if reduction == 'mean':
        return focal_eiouk.mean()
    elif reduction == 'sum':
        return focal_eiouk.sum()
    else:
        return focal_eiouk


def inner_siou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    # 创建一个与 x1 张量形状相同的全零张量 intsctk，用于存储交集的面积
    intsctk = torch.zeros_like(x1)
    # 创建一个布尔掩码 mask，用于筛选出有效的交集。只有当交集的右下角坐标大于左上角坐标时，交集才是有效的。
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    # 根据掩码 mask，计算有效交集的面积，并将结果保存在 intsctk 张量中
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    # 计算并集的面积。将两个边界框的面积相加，然后减去交集的面积。
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    # 计算交并比（IoU），即将交集的面积除以并集的面积，同时加上一个很小的常数 eps，以避免除以零的情况
    iouk = intsctk / (unionk + eps)
    # Calculate SIoU
    s_ch = torch.abs(0.5 * (y2g + y1g) - 0.5 * (y2 + y1) + eps)
    s_cw = torch.abs(0.5 * (x2g + x1g) - 0.5 * (x2 + x1) + eps)
    sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
    sin_alpha_1 = s_ch / sigma
    sin_alpha_2 = s_cw / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    # 二倍角公式，计算角度损失
    angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
    cw = x2.maximum(x2g) - x1.minimum(x1g)  # convex (smallest enclosing box) width
    ch = y2.maximum(y2g) - y1.minimum(y1g)  # convex height
    rho_x = (s_cw / cw) ** 2
    rho_y = (s_ch / ch) ** 2
    gamma = 2 - angle_cost
    # 计算距离损失，与角度损失呈正相关
    distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

    omiga_w = torch.abs(w_pred - w_gt) / torch.max(w_pred, w_gt)
    omiga_h = torch.abs(h_pred - h_gt) / torch.max(h_pred, h_gt)
    # 计算形状损失，θ默认取4
    shape_cost = torch.pow(1 - torch.exp(-omiga_w), 4) + torch.pow(1 - torch.exp(-omiga_h), 4)
    # 计算 SIoU
    siouk = iouk - (distance_cost + shape_cost) / 2
    # 计算 inner_SIoU
    inner_ratio = 0.75
    # 预测框、真实框中心点坐标
    x_c = 0.5 * (x1 + x2)
    y_c = 0.5 * (y1 + y2)
    xg_c = 0.5 * (x1g + x2g)
    yg_c = 0.5 * (y1g + y2g)

    bl_gt = xg_c - (w_gt * inner_ratio) / 2
    br_gt = xg_c + (w_gt * inner_ratio) / 2
    bt_gt = yg_c - (h_gt * inner_ratio) / 2
    bb_gt = yg_c + (h_gt * inner_ratio) / 2
    bl = x_c - (w_pred * inner_ratio) / 2
    br = x_c + (w_pred * inner_ratio) / 2
    bt = y_c - (h_pred * inner_ratio) / 2
    bb = y_c + (h_pred * inner_ratio) / 2
    inter_i = (torch.min(br, br_gt) - torch.max(bl_gt, bl)) * (torch.min(bb_gt, bb) - torch.max(bt_gt, bt))
    union_i = (w_gt * h_gt) * (inner_ratio ** 2) + (w_pred * h_pred) * (inner_ratio ** 2) - inter_i
    inner_iou = inter_i / union_i
    # 计算 inner SIoU
    inner_siouk = inner_iou + siouk - iouk

    # 计算 L_inner-siou
    loss = 1 - inner_siouk
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def focal_eiou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    # 创建一个与 x1 张量形状相同的全零张量 intsctk，用于存储交集的面积
    intsctk = torch.zeros_like(x1)
    # 创建一个布尔掩码 mask，用于筛选出有效的交集。只有当交集的右下角坐标大于左上角坐标时，交集才是有效的。
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    # 根据掩码 mask，计算有效交集的面积，并将结果保存在 intsctk 张量中
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    # 计算并集的面积。将两个边界框的面积相加，然后减去交集的面积。
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    # 计算交并比（IoU），即将交集的面积除以并集的面积，同时加上一个很小的常数 eps，以避免除以零的情况
    iouk = intsctk / (unionk + eps)

    # Bounding box coordinates
    xmin = torch.min(x1, x1g)
    ymin = torch.min(y1, y1g)
    xmax = torch.max(x2, x2g)
    ymax = torch.max(y2, y2g)

    # Bounding box diagonal length
    c = torch.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)

    # Distance between box centers
    center_distance = torch.sqrt((0.5 * (x2g + x1g) - 0.5 * (x2 + x1)) ** 2
                                 + (0.5 * (y2g + y1g) - 0.5 * (y2 + y1)) ** 2)

    # Calculate Ldis
    L_dis = (center_distance ** 2) / (c ** 2 + eps)

    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g
    L_asp = ((w_gt - w_pred) ** 2) / ((xmax - xmin) ** 2) + ((h_gt - h_pred) ** 2) / ((ymax - ymin) ** 2)

    # Calculate EIoU
    eiouk = iouk - L_dis - L_asp
    # Calculate Focal-EIoU
    focal_eiouk = (iouk ** 0.5) * eiouk
    loss = 1 - focal_eiouk

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def clamp_single(box, w, h):
    x1, y1, x2, y2 = box
    # 通过调用clamp方法，将x1的值限制在0到w之间，确保不超出图像宽度的范围
    # 小于min时取min，大于max时取max
    x1 = x1.clamp(min=0, max=w)
    x2 = x2.clamp(min=0, max=w)
    y1 = y1.clamp(min=0, max=h)
    y2 = y2.clamp(min=0, max=h)
    # 将限制后的坐标值作为一个张量返回
    return torch.tensor((x1, y1, x2, y2))


def clamp(boxes, widths, heights):
    """clamp (limit) the values in boxes within the widths and heights of the image."""
    # 创建一个空列表，用于存储限制后的边界框
    clamped_boxes = []
    # 通过zip函数将边界框、图像宽度和图像高度进行迭代。
    for box, w, h in zip(boxes, widths, heights):
        # 调用clamp_single函数对每个边界框进行限制，并将结果添加到clamped_boxes列表中
        clamped_boxes.append(clamp_single(box, w, h))

    # 将clamped_boxes列表中的边界框张量按行堆叠起来，形成一个张量，并在维度0上进行堆叠。
    return torch.stack(clamped_boxes, dim=0)


class BoundingBoxAdaptor:
    def __init__(self, class_names, log, args):
        self.class_names = class_names
        # 遍历args对象中的关键字参数和对应的值
        for k, v in args._get_kwargs():
            setattr(args, k.replace("_b", ""), v)

        # 将修改后的args对象赋值给类的args属性
        self.args = args
        print(self.args)
        # 创建一个CompleteLogger对象，并将log传递给它
        self.logger = CompleteLogger(log)
        # create model
        print("=> using pre-trained model '{}'".format(args.arch))
        # 调用utils.get_model函数获取指定预训练模型的骨干网络（backbone）
        # args.arch指定了模型的名称，pretrain=not args.scratch表示是否使用预训练权重
        backbone = utils.get_model(args.arch, pretrain=not args.scratch)
        num_classes = len(class_names)
        bottleneck_dim = args.bottleneck_dim
        bottleneck = nn.Sequential(
            # 输入通道数为backbone.out_features，输出通道数为bottleneck_dim，使用3x3的卷积核，步长为1，填充为1
            nn.Conv2d(backbone.out_features, bottleneck_dim, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(1024, bottleneck_dim, kernel_size=3, stride=1, padding=1),
            # 二维批归一化层，对卷积输出进行批归一化操作，通道数为bottleneck_dim
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(),
        )
        # bottleneck = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        #     nn.Flatten()
        # )
        # head = nn.Sequential(
        #     nn.Linear(backbone.out_features, bottleneck_dim),
        #     nn.BatchNorm1d(bottleneck_dim),
        #     nn.ReLU(),
        #     nn.Linear(bottleneck_dim, bottleneck_dim),
        #     nn.BatchNorm1d(bottleneck_dim),
        #     nn.ReLU(),
        #     nn.Linear(bottleneck_dim, num_classes * 4),
        # )
        head = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(),
            # 自适应平均池化层，将输入的特征图进行自适应平均池化操作，输出大小固定为1x1
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # 展平层，将输入展平为一维向量
            nn.Flatten(),
            # 全连接层，输入大小为bottleneck_dim，输出大小为num_classes * 4，用于预测边界框的四个坐标值
            nn.Linear(bottleneck_dim, num_classes * 4),
        )
        # 遍历head层中的每一层
        for layer in head:
            # 如果层是nn.Conv2d或nn.Linear类型的层，则对该层的权重进行正态分布初始化
            # （均值为0，标准差为0.01），并将偏置初始化为常数0。
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

        # 定义了一个新的adv_head层
        adv_head = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, num_classes * 4),
        )
        # adv_head = nn.Sequential(
        #     nn.Linear(backbone.out_features, bottleneck_dim),
        #     nn.BatchNorm1d(bottleneck_dim),
        #     nn.ReLU(),
        #     nn.Linear(bottleneck_dim, bottleneck_dim),
        #     nn.BatchNorm1d(bottleneck_dim),
        #     nn.ReLU(),
        #     nn.Linear(bottleneck_dim, num_classes * 4),
        # )
        # 遍历adv_head层中的每一层，
        for layer in adv_head:
            # 如果层是nn.Conv2d或nn.Linear类型的层，则对该层的权重进行正态分布初始化
            # （均值为0，标准差为0.01），并将偏置初始化为常数0
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

        # 创建了一个ImageRegressor模型对象，将骨干网络backbone、
        # 类别数乘以4（因为每个类别有4个边界框坐标）作为参数传递给模型。
        # 同时，还传递了瓶颈层bottleneck、原始的head层和adv_head层
        # Freg = backbone + bottleneck
        # outputs = backbone + bottleneck + head
        # outputs_adv = backbone + bottleneck + grl_layers + adv_head
        self.model = ImageRegressor(
            backbone,
            num_classes * 4,
            bottleneck=bottleneck,
            head=head,
            adv_head=adv_head
        ).to(device)
        # 创建了一个BoxTransform对象，用于边界框的转换操作
        self.box_transform = BoxTransform()

    def load_checkpoint(self, path=None):
        if path is None:
            path = self.logger.get_checkpoint_path('latest')
        if osp.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            return True
        else:
            return False

    def prepare_training_data(self, proposal_list: PersistentProposalList, labeled=True):
        if not labeled:
            # remove (predicted) background proposals
            filtered_proposals_list = []
            for proposals in proposal_list:
                keep_indices = (0 <= proposals.pred_classes) & (proposals.pred_classes < len(self.class_names))
                filtered_proposals_list.append(proposals[keep_indices])
        else:
            # remove proposals with low IoU
            filtered_proposals_list = []
            for proposals in proposal_list:
                keep_indices = proposals.gt_ious > 0.3
                filtered_proposals_list.append(proposals[keep_indices])

        filtered_proposals_list = flatten(filtered_proposals_list, self.args.max_train)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # strong_transform = T.Compose([
        #     ResizeImage(self.args.resize_size),
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.RandomVerticalFlip(p=0.5),
        #     T.ToTensor(),
        #     normalize
        # ])
        # weak_transform = T.Compose([
        #     ResizeImage(self.args.resize_size),
        #     T.ToTensor(),
        #     normalize
        # ])
        # if not labeled:
        #     dataset = ProposalDataset(filtered_proposals_list, weak_transform, crop_func=ExpandCrop(self.args.expand))
        # else:
        #     dataset = ProposalDataset(filtered_proposals_list, strong_transform, crop_func=ExpandCrop(self.args.expand))
        transform = T.Compose([
            T.Resize((self.args.resize_size, self.args.resize_size)),
            # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            # T.RandomGrayscale(),
            T.ToTensor(),
            normalize
        ])
        # crop_func=ExpandCrop(self.args.expand):用于扩展和裁剪提案的函数，它接受一个扩展参数self.args.expand
        dataset = ProposalDataset(filtered_proposals_list, transform, crop_func=ExpandCrop(self.args.expand))
        # 创建了一个数据加载器DataLoader，它接受前面创建的数据集对象dataset作为参数
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=True, num_workers=self.args.workers, drop_last=True)
        return dataloader

    def prepare_validation_data(self, proposal_list: PersistentProposalList):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize((self.args.resize_size, self.args.resize_size)),
            T.ToTensor(),
            normalize
        ])

        # remove (predicted) background proposals
        filtered_proposals_list = []
        for proposals in proposal_list:
            # keep_indices = (0 <= proposals.gt_classes) & (proposals.gt_classes < len(self.class_names))
            keep_indices = (0 <= proposals.pred_classes) & (proposals.pred_classes < len(self.class_names))
            filtered_proposals_list.append(proposals[keep_indices])

        filtered_proposals_list = flatten(filtered_proposals_list, self.args.max_val)
        dataset = ProposalDataset(filtered_proposals_list, transform, crop_func=ExpandCrop(self.args.expand))
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.workers, drop_last=False)
        return dataloader

    def prepare_test_data(self, proposal_list: PersistentProposalList):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize((self.args.resize_size, self.args.resize_size)),
            T.ToTensor(),
            normalize
        ])

        dataset = ProposalDataset(proposal_list, transform, crop_func=ExpandCrop(self.args.expand))
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.workers, drop_last=False)
        return dataloader

    def predict(self, data_loader):
        # switch to evaluate mode
        self.model.eval()
        predictions = deque()

        with torch.no_grad():
            for images, labels in tqdm.tqdm(data_loader):
                images = images.to(device)
                pred_classes = labels['pred_classes'].to(device)
                pred_boxes = labels['pred_boxes'].to(device).float()
                # compute output
                pred_deltas = self.model(images)
                _, pred_boxes = self.box_transform(pred_deltas, pred_classes, pred_boxes)
                pred_boxes = clamp(pred_boxes.cpu(), labels['width'], labels['height'])
                pred_boxes = pred_boxes.numpy().tolist()
                for p in pred_boxes:
                    predictions.append(p)
        return predictions

    # 用于在具有标记数据的验证集上计算模型的基准性能，通过比较模型的预测边界框和真实边界框之间的IoU来评估模型的准确性
    def validate_baseline(self, val_loader):
        """call this function if you have labeled data for validation"""
        ious = AverageMeter("IoU", ":.4e")
        print("Calculate baseline IoU:")
        for _, labels in tqdm.tqdm(val_loader):
            gt_boxes = labels['gt_boxes']
            pred_boxes = labels['pred_boxes']
            # 使用iou_between(pred_boxes, gt_boxes)计算预测边界框和真实边界框之间的IoU，并取平均值
            ious.update(iou_between(pred_boxes, gt_boxes).mean().item(), gt_boxes.size(0))

        # 打印出计算得到的基准IoU的平均值ious.avg
        print(' * Baseline IoU {:.3f}'.format(ious.avg))
        # 返回基准IoU的平均值ious.avg
        return ious.avg

    @staticmethod
    def validate(val_loader, model, box_transform, args) -> float:
        """call this function if you have labeled data for validation"""
        batch_time = AverageMeter('Time', ':6.3f')
        ious = AverageMeter("IoU", ":.4e")
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, ious],
            prefix='Test: '
        )

        # switch to evaluate mode
        model.eval()

        # 使用torch.no_grad()上下文管理器，表示在进行推理时不需要计算梯度
        with torch.no_grad():
            # 记录当前时间，用于计算每个批次的推理时间
            end = time.time()
            # i是循环的索引,images是输入图像的批次，labels是与图像相关的标签
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                pred_classes = labels['pred_classes'].to(device)
                gt_boxes = labels['gt_boxes'].to(device).float()
                pred_boxes = labels['pred_boxes'].to(device).float()

                # compute output
                pred_deltas = model(images)
                _, pred_boxes = box_transform(pred_deltas, pred_classes, pred_boxes)
                pred_boxes = clamp(pred_boxes.cpu(), labels['width'], labels['height'])
                ious.update(iou_between(pred_boxes, gt_boxes.cpu()).mean().item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            print(' * IoU {:.3f}'.format(ious.avg))

        return ious.avg

    def fit(self, data_loader_source, data_loader_target, data_loader_validation=None):
        """When no labels exists on target domain, please set data_loader_validation=None"""
        args = self.args
        print(args)
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        cudnn.benchmark = True
        iter_source = ForeverDataIterator(data_loader_source)
        iter_target = ForeverDataIterator(data_loader_target)
        best_iou = 0.
        box_transform = self.box_transform

        # first pre-train on the source domain
        # backbone -> bottleneck -> head
        model = Regressor(
            self.model.backbone,
            len(self.class_names) * 4,
            # bottleneck_dim=1024
            bottleneck_dim=self.model.backbone.out_features,
            bottleneck=nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            ),
            # 定义一个名为head的线性层 (nn.Linear)，它将主干网络的输出特征映射到长度
            # 为len(self.class_names) * 4的向量，作为回归器模型的最终输出。
            head=nn.Linear(
                self.model.backbone.out_features,
                # 1024,
                len(self.class_names) * 4
            ),

        ).to(device)
        optimizer = Adam(model.get_parameters(), args.pretrain_lr, weight_decay=args.pretrain_weight_decay)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.pretrain_lr * (1. + args.pretrain_lr_gamma * float(x)) ** (-args.pretrain_lr_decay))

        for epoch in range(args.pretrain_epochs):
            print("lr:", lr_scheduler.get_last_lr()[0])
            batch_time = AverageMeter('Time', ':3.1f')
            data_time = AverageMeter('Data', ':3.1f')
            losses = AverageMeter('Loss', ':3.2f')
            ious = AverageMeter("IoU", ":.4e")
            progress = ProgressMeter(
                args.iters_per_epoch,
                [batch_time, data_time, losses, ious],
                prefix="Epoch: [{}]".format(epoch)
            )

            # switch to train mode
            model.train()
            end = time.time()
            for i in range(args.iters_per_epoch):
                # 从iter_source迭代器中获取下一个批次的数据，其中x_s是输入数据，labels_s是与输入数据对应的标签
                x_s, labels_s = next(iter_source)
                x_s = x_s.to(device)
                # bounding box offsets
                # 使用box_transform.get_deltas函数计算预测边界框和真实边界框之间的偏移量delta_s
                delta_s = box_transform.box_transform.get_deltas(labels_s['pred_boxes'], labels_s['gt_boxes']).to(device).float()
                pred_boxes_s = labels_s['pred_boxes'].to(device).float()
                gt_classes_s = labels_s['gt_fg_classes'].to(device)
                gt_boxes_s = labels_s['gt_boxes'].to(device).float()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                # 将源域数据x_s输入模型model，得到预测的边界框偏移量pred_delta_s
                pred_delta_s, _ = model(x_s)
                # 使用box_transform函数根据预测的边界框偏移量、真实前景类别和预测的边界框来计算
                # 最终的预测边界框pred_boxes_s
                pred_delta_s, pred_boxes_s = box_transform(pred_delta_s, gt_classes_s, pred_boxes_s)
                reg_loss = F.smooth_l1_loss(pred_delta_s, delta_s)
                # iou_loss = inner_siou_loss(boxes1=pred_boxes_s, boxes2=gt_boxes_s, reduction='sum')
                # loss = reg_loss + iou_loss
                loss = reg_loss

                losses.update(loss.item(), x_s.size(0))
                ious.update(iou_between(pred_boxes_s.cpu(), gt_boxes_s.cpu()).mean().item(), x_s.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            # evaluate on validation set
            if data_loader_validation is not None:
                iou = self.validate(data_loader_validation, model, box_transform, args)
                best_iou = max(iou, best_iou)

        # training on both domains
        model = self.model
        optimizer = SGD(model.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

        for epoch in range(args.epochs):
            print("lr:", lr_scheduler.get_last_lr()[0])
            # train for one epoch
            batch_time = AverageMeter('Time', ':3.1f')
            data_time = AverageMeter('Data', ':3.1f')
            losses = AverageMeter('Loss', ':3.2f')
            ious = AverageMeter("IoU", ":.4e")
            ious_t = AverageMeter("IoU (t)", ":.4e")
            ious_s_adv = AverageMeter("IoU (s, adv)", ":.4e")
            ious_t_adv = AverageMeter("IoU (t, adv)", ":.4e")
            trans_losses = AverageMeter('Trans Loss', ':3.2f')
            progress = ProgressMeter(
                args.iters_per_epoch,
                [batch_time, data_time, losses, trans_losses, ious, ious_t, ious_s_adv, ious_t_adv],
                prefix="Epoch: [{}]".format(epoch)
            )
            # switch to train mode iou_s_loss, iou_t_loss, iou_s_adv_loss, iou_t_adv_loss,
            model.train()
            # default: margin=4
            mdd = RegressionMarginDisparityDiscrepancy(args.margin).to(device)
            # mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

            end = time.time()
            # args.iters_per_epoch：1000
            for i in range(args.iters_per_epoch):
                x_s, labels_s = next(iter_source)
                x_t, labels_t = next(iter_target)
                x_s = x_s.to(device)
                x_t = x_t.to(device)

                # bounding box offsets
                delta_s = box_transform.box_transform.get_deltas(labels_s['pred_boxes'], labels_s['gt_boxes']).to(device).float()
                pred_boxes_s = labels_s['pred_boxes'].to(device).float()
                gt_classes_s = labels_s['gt_fg_classes'].to(device)
                gt_boxes_s = labels_s['gt_boxes'].to(device).float()
                pred_boxes_t = labels_t['pred_boxes'].to(device).float()
                gt_classes_t = labels_t['pred_classes'].to(device)
                gt_boxes_t = labels_t['gt_boxes'].to(device).float()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                x = torch.cat([x_s, x_t], dim=0)
                outputs, outputs_adv = model(x)
                pred_delta_s, pred_delta_t = outputs.chunk(2, dim=0)
                pred_delta_s_adv, pred_delta_t_adv = outputs_adv.chunk(2, dim=0)

                # box 变换
                pred_delta_s, pred_boxes_s = box_transform(pred_delta_s, gt_classes_s, pred_boxes_s)
                pred_delta_t, pred_boxes_t = box_transform(pred_delta_t, gt_classes_t, pred_boxes_t)
                pred_delta_s_adv, pred_boxes_s_adv = box_transform(pred_delta_s_adv, gt_classes_s, pred_boxes_s)
                pred_delta_t_adv, pred_boxes_t_adv = box_transform(pred_delta_t_adv, gt_classes_t, pred_boxes_t)

                reg_loss = F.smooth_l1_loss(pred_delta_s, delta_s)
                # compute margin disparity discrepancy between domains
                transfer_loss = mdd(pred_delta_s, pred_delta_s_adv, pred_delta_t, pred_delta_t_adv)
                # for adversarial classifier, minimize negative mdd is equal to maximize mdd
                # default: trade_off=0.1
                loss = reg_loss - transfer_loss * args.trade_off
                model.step()

                losses.update(loss.item(), x_s.size(0))
                ious.update(iou_between(pred_boxes_s.cpu(), gt_boxes_s.cpu()).mean().item(), x_s.size(0))
                ious_t.update(iou_between(pred_boxes_t.cpu(), gt_boxes_t.cpu()).mean().item(), x_s.size(0))
                ious_s_adv.update(iou_between(pred_boxes_s_adv.cpu(), gt_boxes_s.cpu()).mean().item(), x_s.size(0))
                ious_t_adv.update(iou_between(pred_boxes_t_adv.cpu(), gt_boxes_t.cpu()).mean().item(), x_s.size(0))
                trans_losses.update(transfer_loss.item(), x_s.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            # evaluate on validation set
            if data_loader_validation is not None:
                iou = self.validate(data_loader_validation, model, box_transform, args)
                best_iou = max(iou, best_iou)

            # save checkpoint
            torch.save(model.state_dict(), self.logger.get_checkpoint_path('latest'))

        print("best_iou = {:3.1f}".format(best_iou))

        self.logger.logger.flush()

    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        # dataset parameters
        parser.add_argument('--resize-size-b', type=int, default=224,
                            help='the image size after resizing')
        parser.add_argument('--max-train-b', type=int, default=10)
        parser.add_argument('--max-val-b', type=int, default=10)
        parser.add_argument('--expand-b', type=float, default=2.,
                            help='The expanding ratio between the input of the bounding box adaptor'
                                 '(the crops of objects) and the the original predicted box.')
        # model parameters
        parser.add_argument('--arch-b', metavar='ARCH', default='resnet101',
                            choices=utils.get_model_names(),
                            help='backbone architecture: ' +
                                 ' | '.join(utils.get_model_names()) +
                                 ' (default: resnet101)')
        parser.add_argument('--bottleneck-dim-b', default=1024, type=int,
                            help='Dimension of bottleneck')
        parser.add_argument('--no-pool-b', action='store_true',
                            help='no pool layer after the feature extractor.')
        parser.add_argument('--scratch-b', action='store_true', help='whether train from scratch.')
        parser.add_argument('--margin', type=float, default=4., help="margin hyper-parameter")
        parser.add_argument('--trade-off', default=0.1, type=float,
                            help='the trade-off hyper-parameter for transfer loss')
        # training parameters
        # 修改前：32   修改后：64
        parser.add_argument('--batch-size-b', default=32, type=int,
                            metavar='N',
                            help='mini-batch size (default: 64)')
        # 修改前：0.004  修改后：0.008
        parser.add_argument('--lr-b', default=0.004, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--lr-gamma-b', default=0.0002, type=float, help='parameter for lr scheduler')
        parser.add_argument('--lr-decay-b', default=0.75, type=float, help='parameter for lr scheduler')
        parser.add_argument('--weight-decay-b', default=5e-4, type=float,
                            metavar='W', help='weight decay (default: 5e-4)')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        parser.add_argument('--workers-b', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 2)')
        parser.add_argument('--epochs-b', default=2, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--pretrain-lr-b', default=0.001, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--pretrain-lr-gamma-b', default=0.0002, type=float, help='parameter for lr scheduler')
        parser.add_argument('--pretrain-lr-decay-b', default=0.75, type=float, help='parameter for lr scheduler')
        parser.add_argument('--pretrain-weight-decay-b', default=1e-3, type=float,
                            metavar='W', help='weight decay (default: 1e-3)')
        parser.add_argument('--pretrain-epochs-b', default=10, type=int, metavar='N',
                            help='number of total epochs to run')
        # 修改前：1000  修改后：500
        parser.add_argument('--iters-per-epoch-b', default=1000, type=int,
                            help='Number of iterations per epoch')
        parser.add_argument('--print-freq-b', default=100, type=int,
                            metavar='N', help='print frequency (default: 100)')
        # seed = 3407（114514）
        parser.add_argument('--seed-b', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--temperature', default=2.0,
                            type=float, help='parameter temperature scaling')
        parser.add_argument("--log-b", type=str, default='box',
                            help="Where to save logs, checkpoints and debugging images.")
        return parser


