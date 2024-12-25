import logging
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import sys
import pprint
import numpy as np
import warnings
warnings.filterwarnings('ignore')


import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.engine import default_writers, launch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
import detectron2.utils.comm as comm
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    MetadataCatalog
)
from detectron2.utils.events import EventStorage
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling import build_backbone

sys.path.append('../../../..')
import tllib.alignment.d_adapt.modeling.meta_arch as models
from tllib.alignment.d_adapt.proposal import ProposalGenerator, ProposalMapper, PersistentProposalList, flatten
from tllib.alignment.d_adapt.feedback import get_detection_dataset_dicts, DatasetMapper

sys.path.append('..')
import utils

import category_adaptation
import bbox_adaptation


# 使用给定的模型在指定数据集上生成前景和背景提议，并将它们保存到磁盘。如果之前已经生成并保存了提议，则直接加载提议。
# 否则，在每个数据集上进行推断和生成提议，并将生成的提议保存到磁盘。最后，返回前景和背景提议列表
def generate_proposals(model, num_classes, dataset_names, cache_root, cfg):
    """Generate foreground proposals and background proposals from `model` and save them to the disk"""
    # 创建一个PersistentProposalList对象，用于存储前景提议，并指定保存路径。
    # cache_root是缓存根目录，dataset_names[0]是数据集名称，"{}_fg.json"用于构造前景提议文件名。
    fg_proposals_list = PersistentProposalList(os.path.join(cache_root, "{}_fg.json".format(dataset_names[0])))
    bg_proposals_list = PersistentProposalList(os.path.join(cache_root, "{}_bg.json".format(dataset_names[0])))

    # 检查是否存在之前保存的前景和背景提议,如果不存在
    if not (fg_proposals_list.load() and bg_proposals_list.load()):
        # 循环遍历dataset_names中的每个数据集名称
        for dataset_name in dataset_names:
            # 使用build_detection_test_loader函数构建一个数据加载器，用于加载数据集dataset_name的测试集
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=ProposalMapper(cfg, False))
            # 创建一个ProposalGenerator对象，用于生成提议
            generator = ProposalGenerator(num_classes=num_classes)
            # 使用model、data_loader和generator对数据集进行推断，生成前景和背景提议数据
            fg_proposals_list_data, bg_proposals_list_data = inference_on_dataset(model, data_loader, generator)
            # 将前景、背景提议数据分别添加到前景、背景提议列表中
            fg_proposals_list.extend(fg_proposals_list_data)
            bg_proposals_list.extend(bg_proposals_list_data)

        # 将前景、背景提议列表中的数据保存到磁盘
        fg_proposals_list.flush()
        bg_proposals_list.flush()
    return fg_proposals_list, bg_proposals_list


# 为给定的提议列表中的每个提议生成类别标签，并将带有类别标签的提议保存到磁盘。
# 如果之前已经生成并保存了带有类别标签的提议，则直接加载提议。否则，将每个提议添加到带有类别标签的提议列表中，
# 并使用类别适配器对列表中的提议进行预测，得到类别标签。最后，返回带有类别标签的提议列表。
def generate_category_labels(prop, category_adaptor, cache_filename):
    """Generate category labels for each proposals in `prop` and save them to the disk"""
    # 创建一个PersistentProposalList对象，用于存储带有类别标签的提议，并指定保存路径为cache_filename
    prop_w_category = PersistentProposalList(cache_filename)
    # 检查是否存在之前保存的带有类别标签的提议。如果不存在
    if not prop_w_category.load():
        # 遍历prop中的每个提议p
        for p in prop:
            # 将提议p添加到带有类别标签的提议列表prop_w_category中
            prop_w_category.append(p)

        # 将提议列表prop_w_category展平并准备为类别适配器进行测试的数据加载器
        data_loader_test = category_adaptor.prepare_test_data(flatten(prop_w_category))
        # 使用类别适配器对测试数据进行预测，得到类别预测结果
        predictions = category_adaptor.predict(data_loader_test)
        # 遍历prop_w_category中的每个提议p
        for p in prop_w_category:
            # 将类别预测结果赋值给提议p的pred_classes属性。
            # 通过从predictions队列中弹出预测结果，为每个提议中的边界框分配类别标签
            p.pred_classes = np.array([predictions.popleft() for _ in range(len(p))])

        # 将带有类别标签的提议列表中的数据保存到磁盘
        prop_w_category.flush()
    return prop_w_category


def generate_bounding_box_labels(prop, bbox_adaptor, class_names, cache_filename):
    """Generate bounding box labels for each proposals in `prop` and save them to the disk"""
    # 创建一个PersistentProposalList对象，用于存储带有边界框标签的提议，并指定保存路径为cache_filename
    prop_w_bbox = PersistentProposalList(cache_filename)
    # 检查是否存在之前保存的带有边界框标签的提议。如果不存在
    if not prop_w_bbox.load():
        # remove (predicted) background proposals
        # 遍历prop中的每个提议p
        for p in prop:
            # 根据提议p的预测类别，生成一个布尔索引数组，用于过滤掉背景提议和无效的类别索引
            keep_indices = (0 <= p.pred_classes) & (p.pred_classes < len(class_names))
            # 将过滤后的提议p添加到带有边界框标签的提议列表prop_w_bbox中
            prop_w_bbox.append(p[keep_indices])

        # 将提议列表prop_w_bbox展平并准备为边界框适配器进行测试的数据加载器
        data_loader_test = bbox_adaptor.prepare_test_data(flatten(prop_w_bbox))
        # 使用边界框适配器对测试数据进行预测，得到边界框预测结果
        predictions = bbox_adaptor.predict(data_loader_test)
        # 遍历prop_w_bbox中的每个提议p
        for p in prop_w_bbox:
            # 将边界框预测结果赋值给提议p的pred_boxes属性。
            # 通过从predictions队列中弹出预测结果，为每个提议中的边界框分配边界框标签
            p.pred_boxes = np.array([predictions.popleft() for _ in range(len(p))])
        prop_w_bbox.flush()
    return prop_w_bbox


def train(model, logger, cfg, args, args_cls, args_box):
    # 将模型设置为训练模式，启用训练相关的操作
    model.train()
    # 检查当前是否在分布式训练环境中。
    # comm.get_world_size()返回当前分布式环境中的进程数量，如果大于1，则表示在分布式环境中。
    distributed = comm.get_world_size() > 1
    if distributed:
        # 获取模型的非并行版本。在分布式训练中，模型通常被包装在
        # torch.nn.DataParallel或torch.nn.DistributedDataParallel中，通过model.module可以获取到原始的模型。
        model_without_parallel = model.module
    else:
        # 直接将模型赋值给model_without_parallel，因为模型本身没有被包装在并行模型中。
        model_without_parallel = model

    # define optimizer and lr scheduler
    params = []
    # 遍历模型的每个子模块以及对应的学习率。
    # model_without_parallel.get_parameters()是一个函数，它返回模型的每个子模块以及其对应的学习率。
    for module, lr in model_without_parallel.get_parameters(cfg.SOLVER.BASE_LR):
        # 将每个子模块的参数添加到params列表中
        params.extend(
            get_default_optimizer_params(
                module,
                base_lr=lr,
                weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
                bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            )
        )
    optimizer = maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    scheduler = utils.build_lr_scheduler(cfg, optimizer)

    # resume from the last checkpoint
    # 创建一个检查点管理器checkpointer，用于保存和加载模型、优化器和调度器的状态。
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler, save_to_disk=True
    )
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    start_iter = 0
    # start_iter = (
    #     checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume).get("iteration", -1) + 1
    # )
    # 使用检查点管理器的resume_or_load方法从最后一个检查点恢复模型、优化器和调度器的状态，并获取迭代次数。
    # 如果没有可用的检查点，则返回-1，表示从头开始训练。然后将迭代次数加1，作为训练的起始迭代次数。
    # start_iter = (
    #     checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume).get("iteration", -1) + 1
    # )
    max_iter = cfg.SOLVER.MAX_ITER

    # 创建一个定期检查点管理器periodic_checkpointer，用于定期保存模型、优化器和调度器的状态。
    # cfg.SOLVER.CHECKPOINT_PERIOD：配置文件中定义的检查点保存周期，即多少个迭代周期保存一次检查点。
    # max_to_keep=3
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    # 根据输出目录和最大迭代次数创建写入器列表。
    # default_writers是一个自定义函数，用于创建写入器。如果当前进程是主进程，则创建写入器列表，否则创建一个空列表。
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # generate proposals from detector
    # 从元数据目录中获取目标类别的类名列表。
    # args.targets[0]是目标数据集的名称，通过该名称获取对应数据集的元数据，并从中提取类名列表。
    classes = MetadataCatalog.get(args.targets[0]).thing_classes
    # 构建用于缓存提议的根目录路径。
    # 该路径是基于配置中指定的输出目录cfg.OUTPUT_DIR，将子目录"cache/proposal"追加到输出目录路径中。
    cache_proposal_root = os.path.join(cfg.OUTPUT_DIR, "cache", "proposal")
    # 调用generate_proposals函数生成目标/源数据集的前景和背景提议。
    # 传递了模型、类别数量、目标数据集名称、提议缓存根目录和配置作为参数，并返回目标数据集的前景和背景提议。
    # category_adaptor = category_adaptation.CategoryAdaptor(classes, os.path.join(cfg.OUTPUT_DIR, "cls"), args_cls)
    # backbone = model.backbone
    # # 创建输入张量 x
    # x = torch.randn(128, 3, 112, 112)
    # x = x.to(torch.device(cfg.MODEL.DEVICE))
    #
    # # 使用骨干网络进行前向传播
    # output_dict = backbone(x)
    # # 打印输出字典的键
    # print(output_dict.keys())
    # final_output = output_dict['res4']

    # prop_s_bg = {PersistentProposalList:16551}
    prop_s_fg, prop_s_bg = generate_proposals(model, len(classes), args.sources, cache_proposal_root, cfg)
    # prop_t_bg = {PersistentProposalList:500}
    prop_t_fg, prop_t_bg = generate_proposals(model, len(classes), args.targets, cache_proposal_root, cfg)
    model = model.to(torch.device('cpu'))
    # train the category adaptor
    # 创建一个CategoryAdaptor对象，用于类别适配。传递类别列表、类别适配输出目录和其他参数作为参数
    category_adaptor = category_adaptation.CategoryAdaptor(classes, os.path.join(cfg.OUTPUT_DIR, "cls"), args_cls)
    # 检查是否存在先前保存的类别适配器的检查点。如果检查点不存在
    if not category_adaptor.load_checkpoint():
        # 为源数据集的前景和背景提议准备训练数据加载器。
        # 将前景提议和背景提议合并，并指定is_source=True表示这是源数据集的训练数据
        data_loader_source = category_adaptor.prepare_training_data(prop_s_fg + prop_s_bg, True)
        data_loader_target = category_adaptor.prepare_training_data(prop_t_fg + prop_t_bg, False)
        data_loader_validation = category_adaptor.prepare_validation_data(prop_t_fg + prop_t_bg)
        # 使用准备的训练数据和验证数据，对类别适配器进行训练。
        # 这将调整适配器模型的参数以最佳地适应源数据集和目标数据集之间的类别差异
        # category_adaptor.fit(source_data, target_data, data_loader_validation)
        category_adaptor.fit(data_loader_source, data_loader_target, data_loader_validation)
        # category_adaptor.fit(data_loader_source, data_loader_target, train_target_imageloader, data_loader_validation)

    # generate category labels for each proposal
    # 构建用于缓存反馈的根目录路径。
    # 该路径是基于配置中指定的输出目录cfg.OUTPUT_DIR，将子目录"cache/feedback"追加到输出目录路径中。
    cache_feedback_root = os.path.join(cfg.OUTPUT_DIR, "cache", "feedback")
    # 调用generate_category_labels函数为目标数据集的前景提议生成类别标签。
    # 传递目标数据集的前景提议、类别适配器对象和保存类别标签的文件路径作为参数，并将生成的类别标签赋值给prop_t_fg
    prop_t_fg = generate_category_labels(
        prop_t_fg, category_adaptor, os.path.join(cache_feedback_root, "{}_fg.json".format(args.targets[0]))
    )
    prop_t_bg = generate_category_labels(
        prop_t_bg, category_adaptor, os.path.join(cache_feedback_root, "{}_bg.json".format(args.targets[0]))
    )
    category_adaptor.model.to(torch.device("cpu"))

    # 检查是否需要进行边界框细化
    if args.bbox_refine:
        # train the bbox adaptor
        # 创建一个BoundingBoxAdaptor对象，用于边界框适配。传递类别列表、边界框适配输出目录和其他参数作为参数
        bbox_adaptor = bbox_adaptation.BoundingBoxAdaptor(classes, os.path.join(cfg.OUTPUT_DIR, "bbox"), args_box)
        # 检查是否存在先前保存的边界框适配器的检查点。如果检查点不存在
        if not bbox_adaptor.load_checkpoint():
            # 为源数据集的前景提议准备边界框适配的训练数据加载器
            data_loader_source = bbox_adaptor.prepare_training_data(prop_s_fg, True)
            data_loader_target = bbox_adaptor.prepare_training_data(prop_t_fg, False)
            data_loader_validation = bbox_adaptor.prepare_validation_data(prop_t_fg)
            # 对边界框适配器的基准模型进行验证。使用准备的验证数据加载器对基准模型进行评估，以评估其性能
            bbox_adaptor.validate_baseline(data_loader_validation)
            # 使用准备的训练数据和验证数据，对边界框适配器进行训练。
            # 这将调整适配器模型的参数以最佳地适应源数据集和目标数据集之间的边界框差异
            bbox_adaptor.fit(data_loader_source, data_loader_target, data_loader_validation)

        # generate bounding box labels for each proposals
        # 构建用于缓存边界框反馈的根目录路径。该路径是基于配置中指定的输出目录cfg.OUTPUT_DIR，
        # 将子目录"cache/feedback_bbox"追加到输出目录路径中。
        cache_feedback_root = os.path.join(cfg.OUTPUT_DIR, "cache", "feedback_bbox")
        # 调用generate_bounding_box_labels函数为目标数据集的前景提议生成边界框标签。传递目标数据集的前景提议、
        # 边界框适配器对象、类别列表和保存边界框标签的文件路径作为参数，并将生成的边界框标签赋值给prop_t_fg_refined。
        prop_t_fg_refined = generate_bounding_box_labels(
            prop_t_fg, bbox_adaptor, classes,
            os.path.join(cache_feedback_root, "{}_fg.json".format(args.targets[0]))
        )
        prop_t_bg_refined = generate_bounding_box_labels(
            prop_t_bg, bbox_adaptor, classes,
            os.path.join(cache_feedback_root, "{}_bg.json".format(args.targets[0]))
        )
        # 将经过边界框适配后的前景提议添加到原始的前景提议列表中
        prop_t_fg += prop_t_fg_refined
        prop_t_bg += prop_t_bg_refined
        bbox_adaptor.model.to(torch.device("cpu"))
        # bbox_adaptor.model.to(torch.device(cfg.MODEL.DEVICE))

    # 检查是否需要减少提议（proposals）的数量
    if args.reduce_proposals:
        # remove proposals
        # 创建一个空列表prop_t_bg_new，用于存储经过减少提议的 背景提议
        prop_t_bg_new = []
        # 对目标数据集的背景提议进行迭代
        for p in prop_t_bg:
            # 根据背景提议中的pred_classes属性，创建一个布尔索引数组keep_indices，用于标记需要保留的背景提议
            keep_indices = p.pred_classes == len(classes)
            prop_t_bg_new.append(p[keep_indices])

        # 将经过减少提议后的背景提议列表赋值给原始的背景提议列表prop_t_bg
        prop_t_bg = prop_t_bg_new
        # 创建一个空列表prop_t_fg_new，用于存储经过减少提议的 前景提议
        prop_t_fg_new = []
        # 对目标数据集的前景提议进行迭代
        for p in prop_t_fg:
            # 将前20个前景提议添加到prop_t_fg_new列表中
            prop_t_fg_new.append(p[:20])
        prop_t_fg = prop_t_fg_new

    model = model.to(torch.device(cfg.MODEL.DEVICE))
    # Data loading code
    # 调用get_detection_dataset_dicts函数，根据提供的源数据集参数args.sources获取用于训练的源数据集。
    # 返回一个表示源数据集的字典列表，并将其赋值给train_source_dataset
    train_source_dataset = get_detection_dataset_dicts(args.sources)
    # augmentation = utils.build_augmentation(cfg=cfg, is_train=False)
    # train_source_loader = build_detection_train_loader(
    #     dataset=train_source_dataset,
    #     cfg=cfg,
    #     mapper=DatasetMapper(cfg, is_train=True, augmentations=augmentation)
    # )
    # 使用build_detection_train_loader函数构建用于训练的源数据集加载器
    train_source_loader = build_detection_train_loader(dataset=train_source_dataset, cfg=cfg)
    # 调用get_detection_dataset_dicts函数，根据提供的目标数据集参数args.targets
    # 以及前景和背景提议列表prop_t_fg和prop_t_bg获取用于训练的目标数据集
    train_target_dataset = get_detection_dataset_dicts(args.targets, proposals_list=prop_t_fg+prop_t_bg)

    # 创建一个DatasetMapper对象mapper，用于将目标数据集字典转换为模型训练所需的输入格式。
    # 该对象根据提供的配置参数cfg进行配置，并使用precomputed_proposal_topk=1000和
    # augmentations=utils.build_augmentation(cfg, True)进行初始化。
    mapper = DatasetMapper(cfg, precomputed_proposal_topk=1000, augmentations=utils.build_augmentation(cfg, True))
    # 使用build_detection_train_loader函数构建用于训练的目标数据集加载器。该函数接受目标数据集
    # train_target_dataset、配置参数cfg、映射器mapper和总批次大小cfg.SOLVER.IMS_PER_BATCH作为参数，
    # 并返回一个数据加载器train_target_loader，用于在训练过程中按批次加载目标数据集。
    train_target_loader = build_detection_train_loader(dataset=train_target_dataset, cfg=cfg, mapper=mapper,
                                                       total_batch_size=cfg.SOLVER.IMS_PER_BATCH)

    # training the object detector
    # 使用日志记录器（logger）输出训练开始的信息，其中start_iter是训练的起始迭代次数
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data_s, data_t, iteration in zip(train_source_loader, train_target_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            optimizer.zero_grad()

            # compute losses and gradient on source domain
            loss_dict_s = model(data_s)
            # 当前进程或节点的总损失值（局部总损失）
            losses_s = sum(loss_dict_s.values())
            assert torch.isfinite(losses_s).all(), loss_dict_s

            loss_dict_reduced_s = {"{}_s".format(k): v.item() for k, v in comm.reduce_dict(loss_dict_s).items()}
            # 归约后的（全局）总损失
            losses_reduced_s = sum(loss for loss in loss_dict_reduced_s.values())
            losses_s.backward()

            # compute losses and gradient on target domain
            # 使用模型model对目标数据data_t进行前向传播，得到损失字典loss_dict_t。
            # 这里labeled=False表示不使用标签信息，可能是因为这是针对无标签数据的训练
            loss_dict_t = model(data_t, labeled=False)
            losses_t = sum(loss_dict_t.values())
            assert torch.isfinite(losses_t).all()

            loss_dict_reduced_t = {"{}_t".format(k): v.item() for k, v in comm.reduce_dict(loss_dict_t).items()}
            # 根据总损失losses_t和权衡参数args.trade_off计算加权损失，并执行反向传播，计算梯度
            (losses_t * args.trade_off).backward()

            # 如果是主进程
            if comm.is_main_process():
                storage.put_scalars(total_loss_s=losses_reduced_s, **loss_dict_reduced_s, **loss_dict_reduced_t)

            # do SGD step
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # evaluate on validation set
            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                utils.validate(model, logger, cfg, args)
                comm.synchronize()

            if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def main(args, args_cls, args_box):
    # 创建一个用于记录日志的日志记录器，其中 "detectron2" 是日志记录器的名称
    logger = logging.getLogger("detectron2")
    # 根据命令行参数args设置配置项，返回一个配置对象 cfg，其中包含了各种训练和模型配置的参数；
    cfg = utils.setup(args)

    # dataset
    # 调用 utils 模块中的 build_dataset 函数，分别构建源、目标和测试数据集对象
    # :: 表示切片的语法，即start:stop:step
    # args.source[::2] 表示从第一个元素开始获取索引为偶数的元素，这里指数据集类别，例如VOC2007
    # args.source[1::2] 表示从第二个元素开始获取索引为奇数的元素，这里指对应数据集的路径
    args.sources = utils.build_dataset(args.sources[::2], args.sources[1::2])
    args.targets = utils.build_dataset(args.targets[::2], args.targets[1::2])
    args.test = utils.build_dataset(args.test[::2], args.test[1::2])

    # create model
    # 根据配置中指定的模型架构名称创建模型对象
    model = models.__dict__[cfg.MODEL.META_ARCHITECTURE](cfg, finetune=args.finetune)
    # backbone = utils.get_model(args.arch, pretrain=not args.scratch)

    # 将模型对象移动到配置中指定的设备上
    model.to(torch.device(cfg.MODEL.DEVICE))
    # backbone.to(torch.device(cfg.MODEL.DEVICE))
    # 使用日志记录器输出模型的信息，包括模型的结构和参数。
    logger.info("Model:\n{}".format(model))
    # logger.info("Backbone:\n{}".format(backbone))

    # 检查是否只进行评估
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return utils.validate(model, logger, cfg, args)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    train(model, logger, cfg, args, args_cls, args_box)

    # evaluate on validation set
    return utils.validate(model, logger, cfg, args)


if __name__ == "__main__":
    # 使用category_adaptation.CategoryAdaptor模块中的get_parser方法获取参数解析器，
    # 并使用parse_known_args方法解析命令行参数。args_cls是包含解析后参数的命名空间对象，argv是剩余的未解析的命令行参数
    args_cls, argv = category_adaptation.CategoryAdaptor.get_parser().parse_known_args()
    print("Category Adaptation Args:")
    pprint.pprint(args_cls)

    args_box, argv = bbox_adaptation.BoundingBoxAdaptor.get_parser().parse_known_args(args=argv)
    print("Bounding Box Adaptation Args:")
    pprint.pprint(args_box)

    parser = argparse.ArgumentParser(add_help=True)
    # dataset parameters
    parser.add_argument('-s', '--sources', nargs='+', help='source domain(s)')
    parser.add_argument('-t', '--targets', nargs='+', help='target domain(s)')
    parser.add_argument('--test', nargs='+', help='test domain(s)')
    # model parameters
    parser.add_argument('--finetune',
                        action='store_true',
                        default=False,
                        help='whether use 10x smaller learning rate for backbone',)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
             "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    # 向参数解析器添加一个新的命令行选项--trade-off，该选项接受一个浮点数值，默认值为1.0;用于调整目标域上损失的权衡参数
    parser.add_argument('--trade-off', default=1., type=float,
                        help='trade-off hyper-parameter for losses on target domain')
    # 向参数解析器添加一个新的命令行选项 - -bbox - refine，该选项不接受参数值，只有在命令行中出现时，将其设置为True
    # 用于指示是否执行边界框细化操作
    parser.add_argument('--bbox-refine', action='store_true', default=False,
                        help='whether perform bounding box refinement')
    # 向参数解析器添加一个新的命令行选项--reduce-proposals，该选项不接受参数值，只有在命令行中出现时，将其设置为True
    # 用于指示是否移除一些低质量的候选框（proposals），对于RetinaNet模型来说是有帮助的
    parser.add_argument('--reduce-proposals', action='store_true',
                        help='whether remove some low-quality proposals.'
                             'Helpful for RetinaNet')
    # training parameters
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0,
                        help="the rank of this machine (unique per machine)")
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    port = 29500
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # args = parser.parse_args()
    args, argv = parser.parse_known_args(argv)
    print("Detection Args:")
    pprint.pprint(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, args_cls, args_box),
    )
