#!/usr/bin/env python
# coding=utf-8
import numpy as np

from collections import namedtuple


Object = namedtuple("Object",
                    ["image_path", "object_id", "object_type",
                     "coordinates"])
Prediction = namedtuple("Prediction",
                        ["image_path", "probability", "coordinates"])


def voc_ap(recall, precision, use_07_metric=False):
    """
    Calculate the AP value using recall and precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p/11.

    else:
        mrec = np.concatenate(([0.], recall, [1.]))
        mprec = np.concatenate(([0.], precision, [0.]))
        for i in range(mprec.size - 1, 0, -1):
            mprec[i-1] = np.maximum(mprec[i-1], mprec[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i+1] - mrec[i]) * mprec[i+1])

    return ap


def custom_voc_eval(gt_csv, pred_csv, ovthresh=0.5, use_07_metric=False):
    """
    Do custom eval, include mAP and FROC
    
    gt_csv: path/to/ground_truth_csv
    pred_csv: path/to/pred_csv
    ovthresh: iou threshold
    """
    # parse ground truth csv, by parsing the ground truth csv,
    # we get ground box info
    num_image = 0
    num_object = 0
    object_dict = {}
    with open(gt_csv) as f:
        # skip header
        next(f)
        for line in f:
            image_path, annotation = line.strip("\n").split(",")
            if annotation == "":
                num_image += 1
                continue
            
            object_annos = annotation.split(";")
            for object_anno in object_annos:
                fields = object_anno.split(" ")  # one box
                object_type = fields[0]
                coords = np.array(list(map(float, fields[1:])))
                # one box info
                obj = Object(image_path, num_object, object_type, coords)
                if image_path in object_dict:
                    object_dict[image_path].append(obj)
                else:
                    object_dict[image_path] = [obj]
                num_object += 1
            num_image += 1

    # parse prediction csv, by parsing pred csv, we get the pre box info
    preds = []
    with open(pred_csv) as f:
        # skip header
        next(f)
        for line in f:
            image_path, prediction = line.strip("\n").split(",")
            
            if prediction == "":
                continue

            coord_predictions = prediction.split(";")
            for coord_prediction in coord_predictions:
                fields = coord_prediction.split(" ")
                probability, x1, y1, x2, y2 = list(map(float, fields))
                pred = Prediction(image_path, probability,
                                  np.array([x1, y1, x2, y2]))
                preds.append(pred)
                
    # sort prediction by probability, decrease order
    preds = sorted(preds, key=lambda x: x.probability, reverse=True)
    nd = len(preds)  # total number of pred boxes
    object_hitted = set()
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # loop over each pred box to see if it matches one ground box
    if nd > 0:
        for d in range(nd):
            if preds[d].image_path in object_dict:
               # one pred box coords
               bb = preds[d].coordinates.astype(float)
               image_path = preds[d].image_path
               # set the initial max overlap iou
               ovmax = -np.inf
               # ground box on the image
               R = [i.coordinates for i in object_dict[image_path]]
               try:
                   BBGT = np.stack(R, axis=0)
               except ValueError:
                   import ipdb;ipdb.set_trace()
               R_img_id = [i.object_id for i in object_dict[image_path]]
               BBGT_hitted_flag = np.stack(R_img_id, axis=0)

               if BBGT.size > 0:
                   # cal the iou between pred box and all the gt boxes on
                   # the image
                   ixmin = np.maximum(BBGT[:, 0], bb[0])
                   iymin = np.maximum(BBGT[:, 1], bb[1])
                   ixmax = np.minimum(BBGT[:, 2], bb[2])
                   iymax = np.minimum(BBGT[:, 3], bb[3])

                   # cal inter area width
                   iw = np.maximum(ixmax - ixmin + 1., 0.)
                   ih = np.maximum(iymax - iymin + 1., 0.)
                   inters = iw * ih  # inter area

                   # cal iou
                   union = (
                       (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - \
                       inters
                   )
                   overlaps = inters / union
                   # find the max iou
                   ovmax = np.max(overlaps)
                   # find the index of the max iou
                   jmax = np.argmax(overlaps)

               if ovmax > ovthresh:
                    # if the max iou greater than the iou thresh
                    if BBGT_hitted_flag[jmax] not in object_hitted:
                        tp[d] = 1.
                        object_hitted.add(BBGT_hitted_flag[jmax])
                    else:
                        fp[d] = 1.

               else:
                    fp[d] = 1.
            else:
                # fp[d] = 1.
                continue
        
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        # cal recall
        rec = tp / float(num_object)
        # cal precision
        prec = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap


if __name__ == "__main__":
    # gt_csv = "../statistic_description/tmp/test.csv"
    # pred_csv = "../tmp/detection_results/loc.csv"
    gt_csv = "/home/stat-caolei/code/TCT_V3/doctor_gt.csv"
    pred_csv = "/home/stat-caolei/code/TCT_V3/doctor_pred.csv"

    recall, precision, ap = custom_voc_eval(gt_csv, pred_csv)
    import ipdb;ipdb.set_trace()

