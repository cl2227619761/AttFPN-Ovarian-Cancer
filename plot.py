#!/usr/bin/env python
# coding=utf-8
"""
draw rectangles on images
"""
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

from tools import utils
from utils.backbone_utils import resnet_fpn_backbone, densenet_fpn_backbone
from faster_rcnn import FasterRCNN
from datasets.data import get_dataset
from run import get_transform
import pandas as pd
import time


setFont = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf",
                             size=30)
# TRUE_POS_DIR = "./plot_images/true_pos/"
# TRUE_NEG_DIR = "./plot_images/true_neg/"
# FALSE_POS_DIR = "./plot_images/false_pos/"
# FALSE_NEG_DIR = "./plot_images/false_neg/"
# INFER_IMG_DIR = "./plot_images/infer_image/"

TRUE_POS_DIR = "./plot_images/true_pos/"
TRUE_NEG_DIR = "./plot_images/true_neg/"
FALSE_POS_DIR = "./plot_images/false_pos/"
FALSE_NEG_DIR = "./plot_images/false_neg/"
# INFER_IMG_DIR = "./plot_images/infer_image/"
INFER_IMG_DIR = "./plot_images/test300/"


def draw_instances(image, boxes, color=(255, 63, 0),
                   figsize=(16,16), title=None,fig=None,
                   ax=None, savefig=False, savename=None):
    """
    Draw rectangle in a PIL image
    image: PIL image
    boxes: boxes coordinates in the form of [x1, y1, x2, y2]
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=color, width=5)
    width, height = image.size
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width+10)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    if savefig:
        assert savename is not None
        fig.savefig(savename+".svg", dpi=300, format="svg")
    ax.imshow(image)


def get_model(model_path):
    backbone = resnet_fpn_backbone("resnet50", True)
    # backbone = resnet_fpn_backbone("resnet101", True)
    # backbone = resnet_fpn_backbone("resnet152", True)
    model = FasterRCNN(backbone, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    return model


def get_dense_model(model_path):
    backbone = densenet_fpn_backbone("densenet121", True)
    model = FasterRCNN(backbone, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    return model


class UnNormalize(object):
    """
    UnNormalize the image to plot
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class PlotDataset(object):
    """
    dataset for ploting
    """

    def __init__(self, csv_file, with_gt_box=True, transform=None):
        self.with_gt_box = with_gt_box
        self.transform = transform
        image_info_df = pd.read_csv(csv_file, na_filter=False)
        self.image_files_list = list(image_info_df.image_path)
        if with_gt_box:
            self.image_annos_info = dict(
                zip(image_info_df.image_path, image_info_df.annotation)
            )
            self.annotations = [self.image_annos_info[i]
                                for i in self.image_files_list]

    def __getitem__(self, idx):
        # load image
        img_path = self.image_files_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.with_gt_box:
            annotation = self.image_annos_info[img_path]
            boxes = []
            if type(annotation) == str:
                if annotation == "":
                    boxes.append([])
                    label = 0
                else:
                    annotation_list = annotation.split(";")
                    for anno in annotation_list:
                        x = []
                        y = []
                        anno = anno[2:]
                        anno = anno.split(" ")
                        for i in range(len(anno)):
                            if i % 2 == 0:
                                x.append(float(anno[i]))
                            else:
                                y.append(float(anno[i]))
                        xmin = min(x)
                        xmax = max(x)
                        ymin = min(y)
                        ymax = max(y)
                        boxes.append([xmin, ymin, xmax, ymax])
                    label = 1
            target = {}
            target["boxes"] = boxes
            target["labels"] = label

            if self.transform is not None:
                img = self.transform(img)

            return img, target, img_path

        else:
            if self.transform is not None:
                img = self.transform(img)
            return img, img_path

    def __len__(self):
        return len(self.image_files_list)


@torch.no_grad()
def visualize(model, data, with_gt_box=True, device="cuda:0",
              flag=True):
    """
    plot rectangle on image

    model: trained model
    data: differ depending on with_gt_box flag
    flag: whether to classify as true pos, true neg, false pos, false neg
    """
    unorm = UnNormalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
    topil = transforms.ToPILImage()
    total_time = []
    model.eval()
    model.to(device)
    if flag:
        if with_gt_box:
            for idx, case in enumerate(data):
                print("===process the %d image====" % idx)
                img = case[0]
                img_path = case[2][0]
                img_name = img_path.split("/")[-1]
                img = img.to(device)
                start = time.clock()
                prediction = model(img)
                stop = time.clock()
                elapsed = stop - start
                box = prediction[0]["boxes"]
                score = prediction[0]["scores"]
                total_time.append(elapsed)
                true_box = case[1]["boxes"]
                true_image_label = case[1]["labels"]
                img = topil(unorm(img.squeeze(0)).cpu())
                draw = ImageDraw.Draw(img)
                if true_image_label == 1 and box.shape[0] > 0 and\
                   score.max().item() > 0.5:
                    for i in range(box.shape[0]):
                        if score[i].item() > 0.5:
                            color = (255, 63, 0)
                            draw.rectangle(box[i].tolist(), outline=color,
                                           width=5)
                            draw.text(
                                box[i].tolist()[:2], "%.2f" % score[i].item(),
                                font=setFont, fill="red"
                            )
                    for j in true_box:
                        color = (63, 127, 63)
                        true_box_coord = [i.item() for i in j]
                        draw.rectangle(true_box_coord,
                                       outline=color, width=5)
                    print("===saving to true pos dir===")
                    img.save(TRUE_POS_DIR + img_name)
                elif true_image_label == 0 and box.shape[0] == 0:
                    print("===saving to true neg dir===")
                    img.save(TRUE_NEG_DIR + img_name)
                elif true_image_label == 0 and box.shape[0] > 0:
                    new_output_index = torch.where(score > 0.5)
                    new_boxes = box[new_output_index]
                    new_scores = score[new_output_index]
                    if len(new_boxes) == 0:
                        print("===saving to true neg dir===")
                        img.save(TRUE_NEG_DIR + img_name)
                    else:
                        for i in range(new_boxes.shape[0]):
                            color = (255, 63, 0)
                            draw.rectangle(new_boxes[i].tolist(), outline=color,
                                        width=5)
                            draw.text(
                                new_boxes[i].tolist()[:2], "%.2f" % new_scores[i].item(),
                                font=setFont,
                                fill="red"
                            )
                        print("===saving to false pos dir===")
                        img.save(FALSE_POS_DIR + img_name)
                # else:
                elif true_image_label == 1 and box.shape[0] > 0 and\
                   score.max().item() <= 0.5:
                    for j in true_box:
                        color = (63, 127, 63)
                        true_box_coord = [i.item() for i in j]
                        draw.rectangle(true_box_coord,
                                       outline=color,
                                       width=5)
                    for i in range(box.shape[0]):
                        if score[i].item() > 0.5:
                            color = (255, 63, 0)
                            draw.rectangle(box[i].tolist(), outline=color,
                                           width=5)
                            draw.text(
                                box[i].tolist()[:2], "%.2f" % score[i].item(),
                                font=setFont, fill="red"
                            )
                    print("===saving to false neg dir===")
                    img.save(FALSE_NEG_DIR + img_name)

            print("time used: %.4f" % np.mean(total_time))
            print(total_time)

        else:
            for idx, case in enumerate(data):
                img = case[0].to(device)
                # img_name = case[1][0].split("/")[-1]
                img_name = "_".join((case[1][0].split("/")[-2], case[1][0].split("/")[-1]))
                start = time.clock()
                prediction = model(img)
                box = prediction[0]["boxes"]
                score = prediction[0]["scores"]
                elapsed = time.clock() - start
                total_time.append(elapsed)
                img = topil(unorm(img.squeeze(0)).cpu())
                draw = ImageDraw.Draw(img)
                print("===process image %d===" % idx)
                if box.shape[0] == 0 or score.max().item() <= 0.8765869:
                    # img.save(os.path.join(INFER_IMG_DIR, "true_neg", img_name))
                    img.save(os.path.join(INFER_IMG_DIR, img_name))
                elif box.shape[0] > 0 and score.max().item() > 0.8765869:
                    for i in range(box.shape[0]):
                        if score[i].item() > 0.8765869:
                            color = (255, 63, 0)
                            draw.rectangle(box[i].tolist(), outline=color,
                                           width=5)
                            draw.text(
                                box[i].tolist()[:2], "%.2f" % score[i].item(),
                                font=setFont, fill="red"
                            )
                    # img.save(os.path.join(INFER_IMG_DIR, "false_pos", img_name))
                    img.save(os.path.join(INFER_IMG_DIR, img_name))
                elif box.shape[0] > 0 and score.max().item() <= 0.8765869:
                    img.save(os.path.join(INFER_IMG_DIR, img_name))

    else:
        for idx, case in enumerate(data):
            print("===process the %d image====" % idx)
            img = case[0]
            img_path = case[2][0]
            img_name = img_path.split("/")[-1]
            img = img.to(device)
            start = time.clock()
            prediction = model(img)
            stop = time.clock()
            elapsed = stop - start
            box = prediction[0]["boxes"]
            score = prediction[0]["scores"]
            total_time.append(elapsed)
            true_box = case[1]["boxes"]
            true_image_label = case[1]["labels"]
            img = topil(unorm(img.squeeze(0)).cpu())
            draw = ImageDraw.Draw(img)
            if true_image_label == 1 and box.shape[0] > 0 and\
               score.max().item() > 0.5:
                for i in range(box.shape[0]):
                    if score[i].item() > 0.5:
                        color = (255, 63, 0)
                        draw.rectangle(box[i].tolist(), outline=color,
                                       width=5)
                        draw.text(
                            box[i].tolist()[:2], "%.2f" % score[i].item(),
                            font=setFont, fill="red"
                        )
                for j in true_box:
                    color = (63, 127, 63)
                    true_box_coord = [i.item() for i in j]
                    draw.rectangle(true_box_coord,
                                   outline=color, width=5)
                img.save(INFER_IMG_DIR + img_name)
            elif true_image_label == 0 and box.shape[0] == 0:
                img.save(INFER_IMG_DIR + img_name)
            elif true_image_label == 0 and box.shape[0] > 0:
                new_output_index = torch.where(score > 0.5)
                new_boxes = box[new_output_index]
                new_scores = score[new_output_index]
                if len(new_boxes) == 0:
                    img.save(INFER_IMG_DIR + img_name)
                else:
                    for i in range(new_boxes.shape[0]):
                        color = (255, 63, 0)
                        draw.rectangle(new_boxes[i].tolist(), outline=color,
                                    width=5)
                        draw.text(
                            new_boxes[i].tolist()[:2], "%.2f" % new_scores[i].item(),
                            font=setFont,
                            fill="red"
                        )
                    img.save(INFER_IMG_DIR + img_name)
            # else:
            elif true_image_label == 1 and box.shape[0] > 0 and\
               score.max().item() <= 0.5:
                for j in true_box:
                    color = (63, 127, 63)
                    true_box_coord = [i.item() for i in j]
                    draw.rectangle(true_box_coord,
                                   outline=color,
                                   width=5)
                for i in range(box.shape[0]):
                    if score[i].item() > 0.5:
                        color = (255, 63, 0)
                        draw.rectangle(box[i].tolist(), outline=color,
                                       width=5)
                        draw.text(
                            box[i].tolist()[:2], "%.2f" % score[i].item(),
                            font=setFont, fill="red"
                        )
                img.save(INFER_IMG_DIR + img_name)

        print("time used: %.4f" % np.mean(total_time))
        print(total_time)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # model_path = "./results/saved_models/resnet50.pth"
    # model = get_model(model_path)
    model_path = "./results/all_models/latest/att_densenet169.pth"
    model = get_dense_model(model_path)
    # dataset = get_dataset("./statistic_description/tmp/test.csv",
    #                       datatype="test",
    #                       transform=get_transform(train=False))
    # data_loader = DataLoader(dataset, batch_size=2, shuffle=False,
    #                          num_workers=8)

    # visualize(model, data_loader, device="cuda:1",
    #           gt_csv="./statistic_description/tmp/test.csv")
    # csv_file = "./statistic_description/tmp/test.csv"
    # csv_file = "./doctor_gt.csv"
    # dataset = PlotDataset(csv_file, transform=get_transform(train=False))
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
    #                          num_workers=8)
    # visualize(model, data_loader, device="cuda:0")
    # csv_file = "/home/stat-caolei/code/TCT_V3/statistic_description/tmp/test_pos_neg.csv"
    csv_file = "/home/stat-caolei/code/TCT_V3/doctor_gt.csv"
    dataset = PlotDataset(csv_file, with_gt_box=False,
                          transform=get_transform(train=False))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=8)
    visualize(model, data_loader, with_gt_box=False, device="cuda:0",
              flag=True)
