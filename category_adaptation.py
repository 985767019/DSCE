"""
Training a category adaptor
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import logging
import os
import random
import time
import warnings
import sys
import argparse
import os.path as osp
from collections import deque
import tqdm
from typing import List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

# from d_adaptation.extension.cdan_module import ImageClassifier, ConditionalDomainAdversarialLoss
# from d_adaptation.extension.domain_discriminator import DomainDiscriminator

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

sys.path.append('../../../..')
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.cdan import ConditionalDomainAdversarialLoss, ImageClassifier

from tllib.alignment.d_adapt.proposal import ProposalDataset, flatten, Proposal
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter

from tllib.utils.logger import CompleteLogger
from tllib.vision.transforms import ResizeImage


import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConfidenceBasedDataSelector:
    """Select data point based on confidence"""
    def __init__(self, confidence_ratio=0.1, category_names=()):
        self.confidence_ratio = confidence_ratio
        # 创建一个空列表，用于存储数据点的类别
        self.categories = []
        # 创建一个空列表，用于存储数据点的置信度得分
        self.scores = []
        self.category_names = category_names
        self.per_category_thresholds = None

    # 定义了一个名为extend的方法，该方法用于扩展数据点的类别和置信度得分
    def extend(self, categories, scores):
        self.categories.extend(categories)
        self.scores.extend(scores)

    def calculate(self):
        # 创建一个字典per_category_scores，其中键是类别名称c，值是空列表。这个字典用于存储每个类别对应的置信度得分。
        per_category_scores = {c: [] for c in self.category_names}
        # 使用zip函数将self.categories和self.scores的元素一一配对，进入循环。循环中的c表示类别，s表示置信度得分
        for c, s in zip(self.categories, self.scores):
            # 将当前循环中的置信度得分s追加到对应类别c的列表中
            per_category_scores[c].append(s)

        # 创建一个空字典per_category_thresholds，用于存储每个类别的置信度阈值
        per_category_thresholds = {}
        # 打印per_category_scores字典中的所有键（类别名称）
        print(per_category_scores.keys())

        # 遍历per_category_scores字典中的键值对
        for c, s in per_category_scores.items():
            # 对当前类别的置信度得分列表s进行降序排序
            s.sort(reverse=True)
            # 打印当前类别的名称、该类别的数据点数量以及根据置信度比例计算得到的阈值索引。
            # int(self.confidence_ratio * len(s))表示根据置信度比例计算得到的阈值在排序后的列表中的索引位置
            # confidence_ratio=0.1(0.2),default:0.0
            print(c, len(s), int(self.confidence_ratio * len(s)))
            # 将当前类别的置信度阈值设置为排序后的列表s中根据索引计算得到的值。如果s列表为空，则将阈值设置为1.0
            per_category_thresholds[c] = s[int(self.confidence_ratio * len(s))] if len(s) else 1.

        print('----------------------------------------------------')
        print("confidence threshold for each category:")
        # 遍历类别名称列表self.category_names
        for c in self.category_names:
            # 打印当前类别的名称和对应的置信度阈值，保留3位小数
            print('\t', c, round(per_category_thresholds[c], 3))
        print('----------------------------------------------------')

        # 将计算得到的每个类别的置信度阈值保存到类的实例变量per_category_thresholds中
        self.per_category_thresholds = per_category_thresholds

    # 根据预先计算得到的置信度阈值，判断传入的数据点是否应该被选中
    def whether_select(self, categories, scores):
        assert self.per_category_thresholds is not None, "please call calculate before selection!"
        # 返回一个列表，其中每个元素表示对应数据点是否被选中。列表的生成通过遍历传入的类别列表categories
        # 和置信度得分列表scores，对于每个数据点，通过比较其置信度得分s与对应类别c的置信度阈值
        # self.per_category_thresholds[c]的大小来确定是否选中；选中为True，否则为False
        # 修改前：s > self.per_category_thresholds[c]
        return [s > self.per_category_thresholds[c] for c, s in zip(categories, scores)]


# 定义了一个鲁棒性较强的交叉熵损失函数，它通过偏移输入张量并对其进行修正，
# 以减少标签噪声对损失的影响，并计算修正后的交叉熵损失的平均值作为最终的损失值
class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-entropy that's robust to label noise"""
    def __init__(self, *args, offset=0.1, **kwargs):
        self.offset = offset
        super(RobustCrossEntropyLoss, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 使用torch.clamp函数将input张量加上"offset"并进行截断，确保其取值在0到1之间；
        # 设置reduction参数为'sum'以计算总损失；ignore_index参数指定忽略的类别索引
        return F.cross_entropy(torch.clamp(input + self.offset, max=1.), target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction='sum') / input.shape[0]


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.head = nn.Linear(1024, 21)

    def forward(self, x):
        x = self.pool_layer(x)
        features = self.bottleneck(x)
        predictions = self.head(features)
        # if self.training:
        #     return predictions, features
        # else:
        #     return predictions
        return predictions

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]


class CategoryAdaptor:
    def __init__(self, class_names, log, args):
        # 将传入的类别名称赋值给类的实例变量self.class_names
        self.class_names = class_names
        # 通过args._get_kwargs()方法获取参数对象args中的关键字参数和对应的值
        for k, v in args._get_kwargs():
            # 通过setattr()函数将关键字参数和对应的值设置为args对象的属性；
            # k.rstrip("_c")用于去除关键字参数名称末尾的"_c"字符。
            setattr(args, k.rstrip("_c"), v)
        self.args = args
        print(self.args)
        # 创建一个CompleteLogger对象，并将日志对象log传递给它
        self.logger = CompleteLogger(log)
        self.selector = ConfidenceBasedDataSelector(self.args.confidence_ratio, range(len(self.class_names) + 1))

        # create model
        # 打印一条消息，表示正在使用的模型名称
        print("=> using model '{}'".format(args.arch))
        # 根据模型名称args.arch使用utils.get_model函数获取模型的主干网络(backbone)。
        # args.scratch = False 表示是否使用预训练的权重
        backbone = utils.get_model(args.arch, pretrain=not args.scratch)
        # args.no_pool=False
        pool_layer = nn.Identity() if args.no_pool else None
        # pool_layer = nn.Identity()
        num_classes = len(self.class_names) + 1
        # args.bottleneck_dim=1024
        # backbone -> pool_layer -> bottleneck -> head
        # pool_layer: nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
        # bottleneck:  nn.Linear(backbone.out_features, bottleneck_dim),nn.BatchNorm1d(bottleneck_dim),nn.ReLU()
        # head: nn.Linear(self._features_dim, num_classes)
        self.model = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                     pool_layer=pool_layer, finetune=not args.scratch).to(device)

    def load_checkpoint(self):
        # 检查是否存在最新的检查点文件。这里使用了osp.exists函数来检查文件是否存在
        # self.logger.get_checkpoint_path('latest')用于获取最新检查点文件的路径
        # /root/autodl-tmp/Decoupled-Adaptation/d_adaptation/logs/faster_rcnn_R_101_C4/voc2clipart/phase3.1_feiou/cls/checkpoints/latest.pth
        if osp.exists(self.logger.get_checkpoint_path('latest')):
            # 加载最新的检查点文件。使用torch.load函数加载检查点文件，并使用map_location='cpu'参数将模型参数映射到CPU上
            checkpoint = torch.load(self.logger.get_checkpoint_path('latest'), map_location='cpu')
            # 将加载的检查点的状态字典(state_dict)加载到模型中。load_state_dict方法用于加载模型的参数
            self.model.load_state_dict(checkpoint)
            return True
        else:
            return False

    def prepare_validation_data(self, proposal_list: List[Proposal]):
        """call this function if you have labeled data for validation"""
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            ResizeImage(self.args.resize_size),
            T.ToTensor(),
            normalize
        ])

        # remove proposals with ignored classes
        filtered_proposals_list = []
        for proposals in proposal_list:
            keep_indices = proposals.gt_classes != -1
            filtered_proposals_list.append(proposals[keep_indices])

        filtered_proposals_list = flatten(filtered_proposals_list, self.args.max_val)
        dataset = ProposalDataset(filtered_proposals_list, transform)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.workers, drop_last=False)
        return dataloader

    def prepare_test_data(self, proposal_list: List[Proposal]):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            ResizeImage(self.args.resize_size),
            T.ToTensor(),
            normalize
        ])

        dataset = ProposalDataset(proposal_list, transform)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.workers, drop_last=False)
        return dataloader

    def prepare_training_data(self, proposal_list: List[Proposal], labeled=True):

        # 如果标签为 False，即没有标记的数据，则继续执行下面的操作
        if not labeled:
            # remove proposals with confidence score between (ignored_scores[0], ignored_scores[1])
            # 创建一个空列表filtered_proposals_list，用于存储过滤后的提议
            filtered_proposals_list = []
            # 使用断言语句确保self.args.ignored_scores的长度为2，并且第一个值小于等于第二个值
            assert len(self.args.ignored_scores) == 2 and self.args.ignored_scores[0] <= self.args.ignored_scores[1], \
                "Please provide a range for ignored_scores!"
            # 遍历proposal_list中的每个提议
            # ignored_scores = [0.05, 0.3]
            for proposals in proposal_list:
                # 计算出置信度在ignored_scores范围之外的提议的索引，
                keep_indices = ~((self.args.ignored_scores[0] < proposals.pred_scores)
                                 & (proposals.pred_scores < self.args.ignored_scores[1]))
                # 将过滤后的提议添加到filtered_proposals_list中
                filtered_proposals_list.append(proposals[keep_indices])

            # calculate confidence threshold for each category on the target domain
            for proposals in filtered_proposals_list:
                # 使用self.selector对象的extend方法将提议的预测类别和预测分数添加到选择器中
                self.selector.extend(proposals.pred_classes.tolist(), proposals.pred_scores.tolist())
            # 调用self.selector对象的calculate方法来计算目标域上每个类别的置信度阈值
            self.selector.calculate()

        # 如果labeled为True，即有标签的数据
        else:
            # default : ignored-scores=[0.05, 0.3]
            # remove proposals with ignored classes or ious between (ignored_ious[0], ignored_ious[1])
            filtered_proposals_list = []
            for proposals in proposal_list:
                keep_indices = (proposals.gt_classes != -1) & \
                               ~((self.args.ignored_ious[0] < proposals.gt_ious) &
                                 (proposals.gt_ious < self.args.ignored_ious[1]))
                filtered_proposals_list.append(proposals[keep_indices])

        # 使用flatten函数将过滤后的提议列表filtered_proposals_list扁平化，并限制最大训练样本数为self.args.max_train
        filtered_proposals_list = flatten(filtered_proposals_list, self.args.max_train)

        # 对图像进行归一化操作
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # mine
        # strong_transform = T.Compose([
        #     ResizeImage(self.args.resize_size),
        #     T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
        #     # T.RandomGrayscale(p=0.1),
        #     T.RandomGrayscale(p=0.2),
        #     T.GaussianBlur([3, 5], (0.1, 2.0)),
        #     T.ToTensor(),
        #     normalize
        # ])
        # weak_transform = T.Compose([
        #     ResizeImage(self.args.resize_size),
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.RandomVerticalFlip(p=0.5),
        #     T.ToTensor(),
        #     normalize
        # ])
        # finally voc2clipart
        strong_transform = T.Compose([
            ResizeImage(self.args.resize_size),
            # 随机颜色抖动，增强颜色变化
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            # 随机应用高斯模糊，模拟不同焦距
            # T.RandomApply([T.GaussianBlur([3, 5], (0.1, 2.0))], p=0.7), # 3
            # T.RandomApply([T.GaussianBlur([3, 5], (0.1, 2.0))], p=0.3),  # 1  47.6
            T.RandomApply([T.GaussianBlur([3, 5], (0.1, 2.0))], p=0.3),  # 1 47.4
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            normalize
        ])
        weak_transform = T.Compose([
            ResizeImage(self.args.resize_size),
            # 负面作用
            # T.RandomCrop(self.args.resize_size, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            # 上限关键：垂直翻转
            T.RandomVerticalFlip(p=0.5),
            # 低强度随机颜色抖动
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize
        ])
        #
        # strong_transform = T.Compose([
        #     ResizeImage(self.args.resize_size),
        #     # 随机颜色抖动，增强颜色变化
        #     T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
        #     # 随机应用高斯模糊，模拟不同焦距
        #     # T.RandomApply([T.GaussianBlur([3, 5], (0.1, 2.0))], p=0.7), # 3
        #     # T.RandomApply([T.GaussianBlur([3, 5], (0.1, 2.0))], p=0.3),  # 1  47.6
        #     T.RandomApply([T.GaussianBlur([3, 5], (0.1, 2.0))], p=0.3),  # 1 47.4
        #     T.RandomGrayscale(0.1),
        #     T.ToTensor(),
        #     normalize
        # ])
        # weak_transform = T.Compose([
        #     ResizeImage(self.args.resize_size),
        #     # 负面作用
        #     # T.RandomCrop(self.args.resize_size, padding=4),
        #     T.RandomHorizontalFlip(p=0.5),
        #     # 上限关键：垂直翻转
        #     T.RandomVerticalFlip(p=0.5),
        #     # 低强度随机颜色抖动
        #     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        #     T.ToTensor(),
        #     normalize
        # ])
        if not labeled:
            dataset = ProposalDataset(filtered_proposals_list, weak_transform)
        else:
            dataset = ProposalDataset(filtered_proposals_list, strong_transform)

        # author
        # transform = T.Compose([
        #     ResizeImage(self.args.resize_size),
        #     T.RandomHorizontalFlip(),
        #     T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
        #     # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
        #     # T.RandomHorizontalFlip(p=0.5),
        #     # 上限关键：垂直翻转
        #     # T.RandomVerticalFlip(p=0.5),
        #     # T.RandomApply([T.GaussianBlur([3, 5], (0.1, 2.0))], p=0.1),
        #     # T.RandomGrayscale(p=0.1),
        #     T.RandomGrayscale(),
        #     T.ToTensor(),
        #     normalize
        # ])
        # # 创建一个ProposalDataset数据集对象，传入过滤后的提议列表filtered_proposals_list和数据转换操作序列transform
        # dataset = ProposalDataset(filtered_proposals_list, transform)
        # drop_last=True丢弃最后一个不完整的批次
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=True, num_workers=self.args.workers, drop_last=True)
        # 返回数据加载器对象dataloader作为训练数据的准备结果
        return dataloader

    def fit(self, data_loader_source, data_loader_target, data_loader_validation=None):
        """When no labels exists on target domain, please set data_loader_validation=None"""
        # 获取self对象的args属性，并将其赋值给args变量。args包含了训练过程中的各种参数
        args = self.args
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        # 设置CuDNN的benchmark标志为True，以启用CuDNN的自动调优机制，提高训练速度
        cudnn.benchmark = True
        # 创建两个无限循环的数据迭代器：在训练过程中循环遍历数据集
        iter_source = ForeverDataIterator(data_loader_source)
        iter_target = ForeverDataIterator(data_loader_target)

        # start training
        # backbone -> pool_layer -> bottleneck -> head
        # pool_layer: nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
        # bottleneck:  nn.Linear(backbone.out_features, bottleneck_dim),nn.BatchNorm1d(bottleneck_dim),nn.ReLU()
        # head: nn.Linear(self._features_dim, num_classes)
        model = self.model


        # features_dim = 1024
        feature_dim = model.features_dim
        num_classes = len(self.class_names) + 1  # 21
        # 如果args.randomized为True，则创建一个具有args.randomized_dim维度和隐藏层大小为1024
        # 的DomainDiscriminator对象，并将其赋值给domain_discri变量
        # DomainDiscriminator是一个用于域分类的模型
        if args.randomized:
            domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)

        # 如果args.randomized为False，则创建一个具有feature_dim * num_classes维度和隐藏层大小
        # 为1024的DomainDiscriminator对象，并将其赋值给domain_discri变量。default: randomized = False
        else:
            # Lin->Batch->Relu->Lin->Batch->Relu->Lin->Sig       1024 * 21 = 21504
            # domain_discri = DomainDiscriminator(feature_dim * num_classes, hidden_size=1024).to(device)
            domain_discri = DomainDiscriminator(feature_dim * num_classes, hidden_size=1024).to(device)

        # 获取模型model和域分类器domain_discri的所有参数，并将它们连接成一个参数列表，赋值给all_parameters变量
        all_parameters = model.get_parameters() + domain_discri.get_parameters()
        # define optimizer and lr scheduler
        optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        # (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)：衰减因子
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
        domain_adv = ConditionalDomainAdversarialLoss(
            domain_discri, entropy_conditioning=args.entropy,
            num_classes=num_classes, features_dim=feature_dim, randomized=args.randomized,
            randomized_dim=args.randomized_dim
        ).to(device)
        best_acc1 = 0.
        # 开始一个循环，迭代args.epochs次，表示训练的总轮数，default ： epochs=10
        for epoch in range(args.epochs):
            # 打印当前学习率。lr_scheduler.get_last_lr()返回当前学习率的列表，通过索引[0]获取第一个学习率值
            print("lr:", lr_scheduler.get_last_lr()[0])
            # define loss function
            # 创建一个条件域对抗损失函数domain_adv。domain_discri：域分类器，用于将输入数据分类为源域或目标域；
            # entropy_conditioning：熵调整参数，用于控制损失函数中的熵惩罚项的权重。
            # randomized：随机化标志，指示是否采用随机策略进行多线性调整 。
            # default：entropy=False randomized=False  randomized_dim=1024 features_dim = 1024
            # CDAN: NIPS 2018
            # eps = 1 - 0.5 * ((epoch + 1) / args.epochs)
            # domain_adv = ConditionalDomainAdversarialLoss(
            #     domain_discri, entropy_conditioning=args.entropy,
            #     num_classes=num_classes, features_dim=feature_dim, randomized=args.randomized,
            #     randomized_dim=args.randomized_dim, eps=0.9
            # ).to(device)

            self.train(iter_source, iter_target, model, domain_adv, optimizer, lr_scheduler, epoch, args)
            # evaluate on validation set
            if data_loader_validation is not None:
                acc1 = self.validate(data_loader_validation, model, self.class_names, args)
                best_acc1 = max(acc1, best_acc1)

            # save checkpoint
            # 保存模型的状态字典（即模型的参数）到指定的文件路径
            torch.save(model.state_dict(), self.logger.get_checkpoint_path('latest'))

        # 将domain_adv对象移动到CPU设备。domain_adv可能是一个模型或者其他可调用对象
        domain_adv.to(torch.device("cpu"))
        # if args.phase == 'analysis':
        #     feature_extractor = nn.Sequential(
        #         model.backbone, model.pool_layer, model.bottleneck
        #     ).to(device)
        #
        #     # 收集特征
        #     source_feature = collect_feature(data_loader_source, feature_extractor, device).cpu().numpy()
        #     target_feature = collect_feature(data_loader_target, feature_extractor, device).cpu().numpy()
        #     # 计算每个域要抽样的数量
        #     sample_size = len(target_feature) // 10
        #     # 在源域特征中随机选择索引
        #     sampled_indices_target = np.random.choice(len(target_feature), sample_size, replace=False)
        #     # 从目标域特征中选择相同的索引（假设目标域特征的数量不少于源域特征）
        #     sampled_indices_source = sampled_indices_target.copy()  # 如果目标域特征数量不同，请调整这里的代码
        #     # 根据选定的索引获取源域和目标域特征
        #     sampled_source_feature = source_feature[sampled_indices_source]
        #     sampled_target_feature = target_feature[sampled_indices_target]
        #     # 连接特征和标签
        #     sampled_features = np.concatenate([sampled_source_feature, sampled_target_feature])
        #     sampled_labels = np.array([0] * len(sampled_source_feature) + [1] * len(sampled_target_feature))
        #
        #     # 运行 t-SNE
        #     tsne_results = TSNE(n_components=2, random_state=0).fit_transform(sampled_features)
        #
        #     # 设置日志记录
        #     logger = logging.getLogger(__name__)
        #     logger.setLevel(logging.DEBUG)
        #     log_dir = './fea_visualize'
        #     os.makedirs(log_dir, exist_ok=True)
        #     log_file = os.path.join(log_dir, 'tsne.log')
        #     file_handler = logging.FileHandler(log_file)
        #     file_handler.setLevel(logging.DEBUG)
        #     logger.addHandler(file_handler)
        #
        #     import matplotlib.colors as mcolors
        #
        #     # 定义新的颜色映射，这里使用了橙色和蓝色
        #     new_colors = ['#FF6F00', '#00C7E0']  # 橙色和蓝色的十六进制代码
        #     new_cmap = mcolors.ListedColormap(new_colors)
        #
        #     plt.figure(figsize=(8, 8))
        #     scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sampled_labels, s=3, cmap=new_cmap)
        #     plt.legend(*scatter.legend_elements(), title="Classes")
        #     plt.title('t-SNE Visualization')
        #
        #     # 保存 t-SNE 图形
        #     svg_filename = os.path.join(log_dir, 'TSNE.svg')
        #     plt.savefig(svg_filename, format='svg')
        #     logger.info("Saving t-SNE to %s", svg_filename)
        #     plt.close()

        # 输出最佳准确率best_acc1，格式化为浮点数，保留1位小数
        print("best_acc1 = {:3.1f}".format(best_acc1))
        self.logger.logger.flush()

    def train(self, iter_source: ForeverDataIterator, iter_target: ForeverDataIterator,
              model: ImageClassifier, domain_adv, optimizer, lr_scheduler,
              epoch: int, args: argparse.Namespace):

        # 创建一个AverageMeter对象batch_time，用于记录每个batch的时间
        batch_time = AverageMeter('Time', ':3.1f')
        # 创建一个AverageMeter对象data_time，用于记录数据加载的时间
        data_time = AverageMeter('Data', ':3.1f')
        # 创建一个AverageMeter对象losses，用于记录总体损失值
        losses = AverageMeter('Loss', ':3.2f')
        # 创建一个AverageMeter对象losses_t，用于记录训练损失值
        losses_t = AverageMeter('Loss(t)', ':3.2f')
        # 创建一个AverageMeter对象trans_losses，用于记录转换损失值
        trans_losses = AverageMeter('Trans Loss', ':3.2f')
        # 创建一个AverageMeter对象cls_accs，用于记录分类准确率
        cls_accs = AverageMeter('Cls Acc', ':3.1f')
        # 创建一个AverageMeter对象domain_accs，用于记录域准确率
        domain_accs = AverageMeter('Domain Acc', ':3.1f')
        # 创建一个ProgressMeter对象progress，用于显示训练进度。
        # args.iters_per_epoch表示每个epoch中的迭代次数，后面的列表中包含了需要显示的平均值的AverageMeter对象。
        progress = ProgressMeter(
            args.iters_per_epoch,
            [batch_time, data_time, losses, losses_t, trans_losses, cls_accs, domain_accs],
            prefix="Epoch: [{}]".format(epoch)
        )

        # switch to train mode
        # 将模型设置为训练模式，启用Batch Normalization和Dropout等层的训练行为
        model.train()
        # 将条件域对抗损失函数domain_adv设置为训练模式，启用其内部的域分类器的训练行为
        domain_adv.train()
        # 记录当前时间作为计时器的起始时间
        end = time.time()
        # train for one epoch
        # iters_per_epoch=1000
        for i in range(args.iters_per_epoch):
            # x_s = {Tensor:(64, 3, 112, 112)}
            x_s, labels_s = next(iter_source)
            # label_t = {dict:11}
            x_t, labels_t = next(iter_target)

            # assign pseudo labels for target-domain proposals with extremely high confidence
            # 根据目标域中的预测类别和置信度，使用self.selector选择是否给目标域的提议分配伪标签
            # selected: (64, ) tensor([False, ...])
            selected = torch.tensor(
                self.selector.whether_select(
                    labels_t['pred_classes'].numpy().tolist(),
                    labels_t['pred_scores'].numpy().tolist()
                )
            )
            # 根据选择的结果和预测类别，为目标域的提议分配伪标签。~selected表示对选择结果取反
            # pseudo_classes_t: (64, ) tensor([-1, ...])
            pseudo_classes_t = selected * labels_t['pred_classes'] + (~selected) * -1
            pseudo_classes_t = pseudo_classes_t.to(device)
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            # 将源域标签中的真实类别gt_classes_s移动到指定的设备
            # gt_classes_s = {Tensor:(64,)} tensor([20,20, ...])
            gt_classes_s = labels_s['gt_classes'].to(device)
            # measure data loading time
            # 计算并更新数据加载的时间，即从上一次循环结束到当前的时间差
            data_time.update(time.time() - end)

            # compute output
            # 将源域输入数据x_s和目标域输入数据x_t沿着指定的维度（dim=0）进行拼接，得到一个新的输入张量x。
            x = torch.cat((x_s, x_t), dim=0)
            # 将输入张量x通过模型model进行前向传播，得到预测结果y和特征表示f
            # backbone -> pool_layer -> bottleneck -> head
            # pool_layer: nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
            # bottleneck:  nn.Linear(backbone.out_features, bottleneck_dim),nn.BatchNorm1d(bottleneck_dim),nn.ReLU()
            # head: nn.Linear(self._features_dim, num_classes)
            # f = {Tensor:(128, 1024)}tensor([[0.0000, 0.0000, 0.4429, ..., 0.5024, 0.0000, 0.0000),\n ... ]])
            # y = {Tensor: (128, 21)}tensor([[-0.0266, 0.1251, -0.1407, ..., -0.1190, 0.1633, 0.0117),\n ... ]])
            y, f = model(x)
            # # 将预测结果y按照指定的维度（dim=0）进行分块（chunk），得到源域的预测结果y_s和目标域的预测结果y_t
            # # 形状: (N, C)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            # 计算源域预测结果y_s和源域真实类别gt_classes_s之间的交叉熵损失, 例如：cls_loss: tensor(3.4366, device=...)
            cls_loss = F.cross_entropy(y_s, gt_classes_s, ignore_index=-1)
            # cls_loss = robust_focal_loss(y_s, gt_classes_s)
            # 计算目标域预测结果y_t和目标域伪标签pseudo_classes_t之间的鲁棒交叉熵损失
            cls_loss_t = RobustCrossEntropyLoss(ignore_index=-1, offset=args.epsilon)(y_t, pseudo_classes_t)
            # 计算条件域对抗损失函数domain_adv在源域和目标域的预测结果和特征表示之间的转换损失
            transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
            # mcc_loss_value = mcc_loss(y_t)
            # cls_loss_t = CDB_loss()(y_t, pseudo_classes_t)
            # 获取域分类器在当前迭代中的准确率
            domain_acc = domain_adv.domain_discriminator_accuracy
            # # 计算总的损失，包括源域分类损失、转换损失和目标域分类损失
            # # args.trade_off是一个权重参数，用于控制转换损失的权重。default: trade_off=1.0
            loss = cls_loss + transfer_loss * args.trade_off + cls_loss_t
            # # 计算源域预测结果y_s和源域真实类别gt_classes_s之间的准确率
            cls_acc = accuracy(y_s, gt_classes_s)[0]

            # 更新总体损失的平均值，使用loss.item()获取当前批次的损失值，x_s.size(0)表示源域输入数据的批次大小
            losses.update(loss.item(), x_s.size(0))
            # 更新分类准确率的平均值
            cls_accs.update(cls_acc, x_s.size(0))
            # 更新域准确率的平均值
            domain_accs.update(domain_acc, x_s.size(0))
            # 更新转换损失的平均值
            trans_losses.update(transfer_loss.item(), x_s.size(0))
            # # 更新目标域分类损失的平均值
            losses_t.update(cls_loss_t.item(), x_s.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # measure elapsed time
            # 计算当前批次的训练时间，并更新平均批次时间
            batch_time.update(time.time() - end)
            # 记录当前时间，用于计算下一个批次的训练时间。
            end = time.time()

            # 如果当前批次的索引i可以被 args.print_freq 整除，就打印训练进度
            if i % args.print_freq == 0:
                # progress.display(i)是一个自定义函数，用于显示当前的训练进度。
                progress.display(i)

    def predict(self, data_loader):
        # switch to evaluate mode
        # 将模型切换到评估模式
        self.model.eval()
        # 创建一个空的双端队列，用于存储预测结果
        predictions = deque()

        # 在这个上下文中，禁用梯度计算，以减少内存消耗并加快计算速度。
        with torch.no_grad():
            # 遍历数据加载器中的每个批次。images是输入图像的张量，_是对应的标签
            for images, _ in tqdm.tqdm(data_loader):
                images = images.to(device)

                # compute output
                output = self.model(images)
                # 对模型的输出进行预测，通过取最大值所在的索引来确定预测结果。.argmax(-1)表示在最后一个维度上取最大值的索引
                prediction = output.argmax(-1).cpu().numpy().tolist()
                for p in prediction:
                    # 将每个预测结果p添加到predictions队列中
                    predictions.append(p)

        # 返回存储了所有预测结果的双端队列predictions
        return predictions

    @staticmethod
    def validate(val_loader, model, class_names, args) -> float:
        # 'Time'是显示在进度条中的名称，':6.3f'是格式化字符串，指定了显示时间的格式。
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            # prefix='Test: '是显示在进度条前缀的字符串
            prefix='Test: '
        )
        # switch to evaluate mode
        model.eval()
        # 创建一个混淆矩阵对象confmat，用于记录模型的预测结果和真实标签之间的关系。
        # len(class_names)+1表示混淆矩阵的大小，其中class_names是类别名称列表。
        confmat = ConfusionMatrix(len(class_names)+1)

        with torch.no_grad():
            # 记录当前时间，用于计算每个批次的耗时
            end = time.time()
            # 遍历验证集数据加载器中的每个批次，images是输入图像的张量，labels是对应的标签。
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                gt_classes = labels['gt_classes'].to(device)

                # compute output
                output = model(images)
                loss = F.cross_entropy(output, gt_classes)

                # measure accuracy and record loss
                # 计算Top-1准确率，accuracy函数用于计算模型的准确率，topk=(1,)表示只计算Top-1准确率。
                acc1, = accuracy(output, gt_classes, topk=(1,))
                # 更新混淆矩阵，将真实标签和模型的预测结果传递给混淆矩阵对象。
                confmat.update(gt_classes, output.argmax(1))
                # 更新损失的平均值，loss.item()获取损失的数值表示，images.size(0)表示当前批次中的图像数量。
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))

                # measure elapsed time
                # 更新批次时间的平均值，time.time() - end计算当前批次的耗时
                batch_time.update(time.time() - end)
                # 更新end为当前时间，用于计算下一个批次的耗时。
                end = time.time()
                if i % args.print_freq == 0:
                    progress.display(i)

            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
            # 打印混淆矩阵，class_names+["bg"]表示将类别名称列表和背景类别"bg"拼接在一起。
            print(confmat.format(class_names+["bg"]))

        # 返回平均Top-1准确率作为模型在验证集上的性能指标。
        return top1.avg

    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        # 创建一个argparse.ArgumentParser对象，并通过add_help=False参数禁用帮助信息的显示。
        parser = argparse.ArgumentParser(add_help=False)
        # dataset parameters
        parser.add_argument('--resize-size-c', type=int, default=112,
                            help='the image size after resizing')
        # 修改前:  default=[0.05, 0.3]
        # 修改后:  default=[0.0, 0.05]
        parser.add_argument('--ignored-scores-c', type=float, nargs='+', default=[0.05, 0.3])
        parser.add_argument('--max-train-c', type=int, default=10)
        parser.add_argument('--max-val-c', type=int, default=2)
        parser.add_argument('--ignored-ious-c', type=float, nargs='+', default=(0.4, 0.5),
                            help='the iou threshold for ignored boxes')
        # model parameters
        # 限制参数的可选值为utils.get_model_names()返回的模型名称列表
        parser.add_argument('--arch-c', metavar='ARCH', default='resnet101',
                            choices=utils.get_model_names(),
                            help='backbone architecture: ' +
                                 ' | '.join(utils.get_model_names()) +
                                 ' (default: resnet101)')
        parser.add_argument('--bottleneck-dim-c', default=1024, type=int,
                            help='Dimension of bottleneck')
        parser.add_argument('--no-pool-c', action='store_true',
                            help='no pool layer after the feature extractor.')
        parser.add_argument('--scratch-c', action='store_true', help='whether train from scratch.')
        # action='store_true'指定了当命令行中出现--scratch-c参数时，将其值设置为True。
        # store_true表示将参数解析为布尔值，并将其设置为True。
        parser.add_argument('--randomized-c', action='store_true',
                            help='using randomized multi-linear-map (default: False)')
        parser.add_argument('--randomized-dim-c', default=1024, type=int,
                            help='randomized dimension when using randomized multi-linear-map (default: 1024)')
        parser.add_argument('--entropy-c', default=False, action='store_true', help='use entropy conditioning')
        parser.add_argument('--trade-off-c', default=1., type=float,
                            help='the trade-off hyper-parameter for transfer loss')
        parser.add_argument('--confidence-ratio-c', default=0.0, type=float)
        parser.add_argument('--epsilon-c', default=0.01, type=float,
                            help='epsilon hyper-parameter in Robust Cross Entropy')
        # training parameters
        # 修改前：64  修改后：128
        parser.add_argument('--batch-size-c', default=64, type=int,
                            metavar='N',
                            help='mini-batch size (default: 64)')
        # 修改前：0.01  修改后：0.02
        parser.add_argument('--learning-rate-c', default=0.01, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--lr-gamma-c', default=0.001, type=float, help='parameter for lr scheduler')
        parser.add_argument('--lr-decay-c', default=0.75, type=float, help='parameter for lr scheduler')
        parser.add_argument('--momentum-c', default=0.9, type=float, metavar='M', help='momentum')
        parser.add_argument('--weight-decay-c', default=1e-3, type=float,
                            metavar='W', help='weight decay (default: 1e-3)',
                            dest='weight_decay')
        parser.add_argument('--workers-c', default=2, type=int, metavar='N',
                            help='number of data loading workers (default: 2)')
        parser.add_argument('--epochs-c', default=10, type=int, metavar='N',
                            help='number of total epochs to run')
        # 修改前：1000   修改后：500
        parser.add_argument('--iters-per-epoch-c', default=1000, type=int,
                            help='Number of iterations per epoch')
        parser.add_argument('--print-freq-c', default=100, type=int,
                            metavar='N', help='print frequency (default: 100)')
        parser.add_argument('--rho', type=float, default=0.05, help="GPU ID")
        # seed = 3407（114514）
        parser.add_argument('--seed-c', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--temperature', default=2.0,
                            type=float, help='parameter temperature scaling')
        parser.add_argument('--eps', default=1.0, type=float,
                            help='hyper-parameter for environemnt label smoothing.')
        parser.add_argument("--log-c", type=str, default='cdan2',
                            help="Where to save logs, checkpoints and debugging images.")
        parser.add_argument("--phase", type=str, default='analysis', choices=['train', 'test', 'analysis'],
                            help="When phase is 'test', only test the model."
                                 "When phase is 'analysis', only analysis the model.")
        return parser
