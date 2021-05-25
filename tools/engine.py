import math
import sys
import time
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageDraw
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

import torchvision.models.detection.mask_rcnn
from torchvision.ops import nms

import sys
sys.path.append(".")
sys.path.append("..")
from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import utils
from .voc_eval import write_custom_voc_results_file, do_python_eval
from .voc_eval_new import custom_voc_eval
# 使用tensorboard可视化损失
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import copy


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def train_process(model, optimizer, lr_sche,
                  dataloaders, num_epochs,
                  use_tensorboard,
                  device,
                 # model save params
                 save_model_path,
                 record_iter,
                 # tensorboard
                 writer=None,
                 ReduceLROnPlateau=False):
    savefig_flag = True
    model.train()
    model.apply(freeze_bn)
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    best_score = 0.0
    best_stat_dict = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        lr_scheduler = None
        print("====Epoch {0}====".format(epoch))
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(dataloaders['train']) - 1)
            lr_scheduler = utils.warmup_lr_scheduler(optimizer,
                                                    warmup_iters,
                                                    warmup_factor)
        for i, (images, targets) in enumerate(tqdm(dataloaders['train']), 0):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                      for t in targets]
            optimizer.zero_grad()
            # 得到损失值字典
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            losses_total = losses.item()
            # roi分类损失
            loss_classifier = loss_dict['loss_classifier'].item()
            # roi回归损失
            loss_box_reg = loss_dict['loss_box_reg'].item()
            # rpn分类损失
            loss_objectness = loss_dict['loss_objectness'].item()
            # rpn回归损失
            loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()
            # 学习率
            lr = optimizer.param_groups[0]['lr']
            # lr_small = optimizer.param_groups[0]["lr"]
            # lr_large = optimizer.param_groups[1]["lr"]
            running_loss += losses_total
            running_loss_classifier += loss_classifier
            running_loss_box_reg += loss_box_reg
            running_loss_objectness += loss_objectness
            running_loss_rpn_box_reg += loss_rpn_box_reg

            if (i+1) % record_iter == 0:
                print('''Epoch{0} loss:{1:.4f}
                         loss_classifier:{2:.4f} loss_box_reg:{3:.4f}
                         loss_objectness:{4:.4f} loss_rpn_box_reg:{5:.4f}\n'''.format(
                          epoch,
                          losses_total, loss_classifier,
                          loss_box_reg, loss_objectness,
                          loss_rpn_box_reg
                      ))
                if use_tensorboard:
                    # 写入tensorboard
                    writer.add_scalar("Total loss",
                                     running_loss / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("RoI classification loss",
                                     running_loss_classifier / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("RoI reg loss",
                                     running_loss_box_reg / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("RPN classification loss",
                                     running_loss_objectness / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("RPN reg loss",
                                     running_loss_rpn_box_reg / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("lr",
                                     lr,
                                     epoch * len(dataloaders['train']) + i)
                    # writer.add_scalar("lr_small",
                    #                  lr_small,
                    #                  epoch * len(dataloaders['train']) + i)
                    # writer.add_scalar("lr_large",
                    #                  lr_large,
                    #                  epoch * len(dataloaders['train']) + i)
                    running_loss = 0.0
                    running_loss_classifier = 0.0
                    running_loss_box_reg = 0.0
                    running_loss_objectness = 0.0
                    running_loss_rpn_box_reg = 0.0

        # val_mAP, val_fig = custom_voc_evaluate(
        #     model, dataloaders['val'], device=device,
        #     voc_results_dir=voc_results_dir,
        #     savefig_flag=savefig_flag
        # )
        # val_mAP, acc, roc_auc = custom_voc_evaluate(
        #     model, dataloaders["val"], device=device,
        #     gt_csv_path="/home/stat-caolei/code/TCT_V3/statistic_description/tmp/val.csv",
        #     cls_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/cls.csv",
        #     loc_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/loc.csv"
        # )
        val_mAP = custom_voc_evaluate(
            model, dataloaders["val"], device=device,
            gt_csv_path="/home/stat-caolei/code/TCT_V3/statistic_description/tmp/xiugao_val.csv",
            cls_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/xiugao_cls.csv",
            loc_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/xiugao_loc.csv"
        )
        # print("Epoch: ", epoch, "| val mAP: %.4f" % val_mAP,
        #       "| val acc: %.4f" % acc, "| val auc: %.4f" % roc_auc)
        print("Epoch: ", epoch, "| val mAP: %.4f" % val_mAP)
        if not ReduceLROnPlateau:
            lr_sche.step()
        else:
            lr_sche.step(val_mAP)
        if val_mAP > best_score:
            best_score = val_mAP
            best_stat_dict = copy.deepcopy(model.state_dict())
            savefig_flag = True
            # dataiter_val = iter(dataloaders['val'])
            # val_imgs, val_labels = dataiter_val.next()
            # val_imgs = val_imgs[0].cuda()
            # prediction = model([val_imgs])
            # box = prediction[0]['boxes']
            # score = prediction[0]['scores']
            # label = prediction[0]['labels']
            # true_box = val_labels[0]['boxes']
            # true_label = val_labels[0]['labels']
            # topil = ToPILImage()
            # val_imgs = topil(val_imgs.cpu())
            # draw = ImageDraw.Draw(val_imgs)
            # for k in range(box.shape[0]):
            #     if label[k].item() == 1 and score[k].item() > 0.5:
            #         color = (0, 255, 0)
            #         draw.rectangle(box[k].tolist(),
            #                       outline=color, width=5)
            #         draw.text(
            #             box[k].tolist()[:2], "%.2f"%score[k].item(),
            #             font=OPT.setFont, fill='red'
            #         )
            # for j in range(true_box.shape[0]):
            #     if true_label[j].item() == 1:
            #         color = (255, 215, 0)
            #         draw.rectangle(
            #             true_box[j].tolist(), outline=color,
            #             width=5
            #         )
            # val_imgs = ToTensor()(val_imgs)
            # writer.add_image('val_pred', val_imgs,
            #                 global_step=epoch)
        else:
            savefig_flag = False
        if use_tensorboard:
            # writer.add_figure(
            #     "Validation PR-curve",
            #     val_fig,
            #     global_step=epoch
            # )
            writer.add_scalar(
                'Validation mAP',
                val_mAP,
                global_step=epoch
            )
            # writer.add_scalar(
            #     "Validation acc",
            #     acc,
            #     global_step=epoch
            # )
            # writer.add_scalar(
            #     "Validation auc",
            #     roc_auc,
            #     global_step=epoch
            # )
        model.train()
        model.apply(freeze_bn)

    print("====训练完成====")
    print("Best Valid mAP: %.4f" % best_score)
    torch.save(best_stat_dict, save_model_path)

    print("====开始测试====")
    model.load_state_dict(best_stat_dict)
    # test_mAP, test_fig = custom_voc_evaluate(
    #     model, dataloaders['test'],
    #     device=device,
    #     voc_results_dir=voc_results_dir,
    #     savefig_flag=True
    # )
    # test_mAP, test_acc, test_roc_auc = custom_voc_evaluate(
    #     model, dataloaders["test"], device=device,
    #     gt_csv_path="/home/stat-caolei/code/TCT_V3/statistic_description/tmp/test.csv",
    #     cls_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/cls.csv",
    #     loc_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/loc.csv"
    # )
    test_mAP = custom_voc_evaluate(
        model, dataloaders["test"], device=device,
        gt_csv_path="/home/stat-caolei/code/TCT_V3/statistic_description/tmp/xiugao_test.csv",
        cls_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/xiugao_cls.csv",
        loc_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/xiugao_loc.csv"
    )
    # print("Test mAP: %.4f" % test_mAP,
    #       "Test acc: %.4f" % test_acc,
    #       "Test auc: %.4f" % test_roc_auc)
    print("Test mAP: %.4f" % test_mAP)
    if use_tensorboard:
        # writer.add_figure(
        #     'Test PR-curve',
        #     test_fig,
        #     global_step=0
        # )
        writer.close()


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.no_grad()
def custom_voc_evaluate(model, data_loader, device,
                        gt_csv_path,
                        cls_csv_path,
                        loc_csv_path,
                        savefig_flag=False):
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test'

    preds = []
    labels = []
    locs = []
    for image, label, _ in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        outputs = model(image)


        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                  for t in outputs]
        coords = []  # used to save pred coords x1 y1 x2 y2
        coords_score = []  # used to save pred box scores
        if len(outputs[-1]["boxes"]) == 0:
            # if no pred boxes, means that the image is negative
            preds.append(0)
            coords.append([])
            coords_score.append("")
            locs.append("")

        else:
            # preds.append(torch.max(outputs[-1]["scores"]).tolist())

            # we keep those pred boxes whose score is more than 0.1
            new_output_index = torch.where(outputs[-1]["scores"] > 0.1)
            new_boxes = outputs[-1]["boxes"][new_output_index]
            new_scores = outputs[-1]["scores"][new_output_index]
            if len(new_boxes) != 0:
                preds.append(torch.max(new_scores).tolist())
            else:
                preds.append(0)
            
            for i in range(len(new_boxes)):
                new_box = new_boxes[i].tolist()
                coords.append([new_box[0], new_box[1],
                               new_box[2], new_box[3]])
            coords_score += new_scores.tolist()
            line = ""
            for i in range(len(new_boxes)):
                if i == len(new_boxes) - 1:
                    line += str(coords_score[i]) + ' ' + str(coords[i][0]) + ' ' + \
                            str(coords[i][1]) + ' ' + str(coords[i][2]) + ' ' + \
                            str(coords[i][3])
                else:
                    line += str(coords_score[i]) + ' ' + str(coords[i][0]) + ' ' + \
                            str(coords[i][1]) + ' ' + str(coords[i][2]) + ' ' + \
                            str(coords[i][3]) + ';'

            locs.append(line)

    cls_res = pd.DataFrame(
        {"image_path": data_loader.dataset.image_files_list,
         "prediction": preds}
    )
    print("====write cls pred results to csv====")
    cls_res.to_csv(cls_csv_path, columns=["image_path", "prediction"],
                   sep=',', index=None)
    loc_res = pd.DataFrame(
        {"image_path": data_loader.dataset.image_files_list,
         "prediction": locs}
    )
    print("====write loc pred results to csv====")
    loc_res.to_csv(loc_csv_path, columns=["image_path", "prediction"],
                   sep=',', index=None)
    gt_anno = pd.read_csv(gt_csv_path, na_filter=False)
    gt_label = gt_anno.annotation.astype(bool).astype(float).values
    pred = cls_res.prediction.values
    recall, precision, ap = custom_voc_eval(gt_csv_path, loc_csv_path)
    # import ipdb;ipdb.set_trace()
    # acc = ((pred >= 0.5) == gt_label).mean()
    # fpr, tpr, _ = roc_curve(gt_label, pred)
    # roc_auc = auc(fpr, tpr)
    # return ap, acc, roc_auc
    return ap



