#!/usr/bin/env python
# coding=utf-8
"""
run script
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import PIL
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import argparse
import sys
import os

from utils.backbone_utils import resnet_fpn_backbone, densenet_fpn_backbone
from faster_rcnn import FasterRCNN, FastRCNNPredictor
from datasets import transforms as T
from torchvision import transforms
from datasets.data import get_dataset
import _utils
from tools import engine


def get_transform(train):
    """
    Data augmentation ops
    train: a boolean flag to do diff transform in train or val, test
    """
    if not train:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        return transforms.Compose(transform_list)
    else:
        transform_list = [
            # T.ImgAugTransform(),  # 别的方法不增强
            T.ToTensor(),
            T.Normalize() 
        ]
        return T.Compose(transform_list)
  
# def get_transform(train):
#     """
#     数据增强操作
#     """
#     transforms_list = []
#     transforms_list.append(T.ToTensor())
#     if train:
#         transforms_list.append(T.RandomHorizontalFlip(0.5))
#         transforms_list.append(T.RandomVerticalFlip(0.5))

#     return T.Compose(transforms_list)


def parse_args(args):
    parser = argparse.ArgumentParser(description="TCT目标检测")
    subparsers = parser.add_subparsers(
        help="使用的优化器类型",
        dest="optimizer_type"
    )
    subparsers.required = True

    parser.add_argument("--model_name",
                        help="所使用的backbone网络",
                        type=str,
                        default="resnet50")
    parser.add_argument("--pretrained",
                        help="backbone网络是否使用预训练权重",
                        type=bool,
                        default=True)
    parser.add_argument("--device",
                        help="使用cuda还是cpu",
                        type=str,
                        default="cuda:0")
    parser.add_argument("--seed",
                        help="训练使用的种子数",
                        type=int,
                        default=7)
    parser.add_argument("--root",
                        help="图像所在的根目录",
                        type=str,
                        default="/home/stat-caolei/code/FasterDetection/data/VOC2007_/")
    parser.add_argument("--train_batch_size",
                        help="训练集的batch size",
                        type=int,
                        default=2)
    parser.add_argument("--val_batch_size",
                        help="验证集的batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--test_batch_size",
                        help="测试集的batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--num_workers",
                        help="训练时的线程数",
                        type=int,
                        default=12)
    parser.add_argument("--log_dir",
                        help="tensorboard生成的log存储路径",
                        type=str,
                        default="./logs")
    sgd_parser = subparsers.add_parser("SGD")
    sgd_parser.add_argument("--sgd_lr",
                            help="SGD的学习率",
                            type=float,
                            default=0.005)
    sgd_parser.add_argument("--momentum",
                            help="SGD的momentum",
                            type=float,
                            default=0.9)
    sgd_parser.add_argument("--weight_decay",
                            help="SGD的权值衰减",
                            type=float,
                            default=5e-4)
    adam_parser = subparsers.add_parser("Adam")
    adam_parser.add_argument("--adam_lr",
                             help="Adam的学习率",
                             type=float,
                             default=0.01)
    parser.add_argument("--step_size",
                        help="StepLR的学习率衰减步数",
                        type=int,
                        default=8)
    parser.add_argument("--gamma",
                        help="StepLR学习率衰减时的gamma值",
                        type=float,
                        default=0.1)
    parser.add_argument("--num_epochs",
                        help="迭代的Epoch次数",
                        type=int,
                        default=26)
    parser.add_argument("--save_model_path",
                       help="模型的存储位置",
                       type=str,
                       default="./results/saved_models/resnet50.pth")
    parser.add_argument("--record_iter",
                       help="每隔多少次写入一次损失",
                       type=int,
                       default=10)
    parser.add_argument("--voc_results_dir",
                        help="预测框文件存储的位置",
                        type=str,
                        default="/home/stat-caolei/code/Final_TCT_Detection/tmp/detection_results/")
    parser.add_argument("--pretrained_resnet50_coco",
                        help="是否使用resnet50在coco上的预训练权重",
                        type=bool,
                        default=False)
    parser.add_argument("--ReduceLROnPlateau",
                        help="使用动态衰减lr的方式",
                        type=bool,
                        default=False)
    
    return parser.parse_args(args)


def main(args=None):
    model_urls = {
        "fasterrcnn_resnet50_fpn_coco":
        "http://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    }
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    CLASSES = {"__background__", "Positive"}
    if args.pretrained_resnet50_coco:
        backbone = resnet_fpn_backbone("resnet50", False)
        model = FasterRCNN(backbone, num_classes=91)
        state_dict = load_state_dict_from_url(
            model_urls["fasterrcnn_resnet50_fpn_coco"],
            progress=True
        )
        model.load_state_dict(state_dict)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                          len(CLASSES))
        
    else:
        # backbone = resnet_fpn_backbone(args.model_name, args.pretrained)
        # model = FasterRCNN(backbone, num_classes=len(CLASSES))
        backbone = densenet_fpn_backbone(args.model_name, args.pretrained)
        model = FasterRCNN(backbone, num_classes=len(CLASSES))
    print(model)


    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 加载数据
    print("====Loading data====")
    #下面是原始的
    # dataset = get_dataset("./statistic_description/tmp/train.csv",
    #                       datatype="train",
    #                       transform=get_transform(train=True))
    # dataset_val = get_dataset("./statistic_description/tmp/val.csv",
    #                           datatype="val",
    #                           transform=get_transform(train=False))
    # dataset_test = get_dataset("./statistic_description/tmp/test.csv",
    #                            datatype="test",
    #                            transform=get_transform(train=False))
    #下面是修稿的
    dataset = get_dataset("./statistic_description/tmp/xiugao_train.csv",
                          datatype="train",
                          transform=get_transform(train=True))
    dataset_val = get_dataset("./statistic_description/tmp/xiugao_val.csv",
                              datatype="val",
                              transform=get_transform(train=False))
    dataset_test = get_dataset("./statistic_description/tmp/xiugao_test.csv",
                               datatype="test",
                               transform=get_transform(train=False))
    # dataset = get_custom_voc(
    #     root=args.root,
    #     transforms=get_transform(train=True),
    #     dataset_flag="train"
    # )
    # dataset_val = get_custom_voc(
    #     root=args.root,
    #     transforms=get_transform(train=False),
    #     dataset_flag="val"
    # )
    # dataset_test = get_custom_voc(
    #     root=args.root,
    #     transforms=get_transform(train=False),
    #     dataset_flag="test"
    # )

    print("====Creating dataloader====")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_utils.collate_fn
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_utils.collate_fn
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False,
        num_workers=8,
        collate_fn=_utils.collate_fn
    )
    dataloaders = {
        "train": dataloader,
        "val": dataloader_val,
        "test": dataloader_test,
    }
    logdir = args.log_dir
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    print("====Loading model====")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.sgd_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD([
        #     {"params": [p for p in model.backbone.body.parameters()
        #                 if p.requires_grad]},
        #     {"params": [p for p in model.backbone.fpn.parameters()
        #                 if p.requires_grad], "lr": 1e-2},
        #     {"params": [p for p in model.backbone.bottom_up.parameters()
        #                 if p.requires_grad], "lr": 1e-2},
        #     {"params": [p for p in model.rpn.parameters()
        #                 if p.requires_grad], "lr": 1e-2},
        #     {"params": [p for p in model.roi_heads.parameters()
        #                 if p.requires_grad], "lr": 1e-2},
        # ], lr=1e-3, momentum=args.momentum,
        # weight_decay=args.weight_decay)

        if args.ReduceLROnPlateau:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.1, patience=10,
                verbose=False, threshold=0.0001, threshold_mode="rel",
                cooldown=0, min_lr=0, eps=1e-8
            )
        else:
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(
            #     optimizer, step_size=args.step_size, gamma=args.gamma
            # )
            # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[8, 24], gamma=args.gamma
            # )
            # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[10, 30], gamma=args.gamma
            # )
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[50, 60], gamma=args.gamma
            )

    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.adam_lr)
        lr_scheduler = None

    print("====Start training====")
    # engine.train_process(model=model, optimizer=optimizer,
    #                      lr_sche=lr_scheduler,
    #                      dataloaders=dataloaders,
    #                      num_epochs=args.num_epochs,
    #                      use_tensorboard=True,
    #                      device=device,
    #                      save_model_path=args.save_model_path,
    #                      record_iter=args.record_iter,
    #                      writer=writer,
    #                      ReduceLROnPlateau=args.ReduceLROnPlateau)
    engine.train_process(model=model, optimizer=optimizer,
                         lr_sche=lr_scheduler,
                         dataloaders=dataloaders,
                         num_epochs=args.num_epochs,
                         use_tensorboard=True,
                         device=device,
                         save_model_path=args.save_model_path,
                         record_iter=args.record_iter,
                         writer=writer,
                         ReduceLROnPlateau=args.ReduceLROnPlateau)


if __name__ == "__main__":
    main()

