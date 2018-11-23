import glob
import json
import os
import re
import sys
import time

import cntk
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from cntk import load_model
from easydict import EasyDict

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))

from lib.FasterRCNN_eval import FasterRCNN_Evaluator
from Helpers import utils
from lib.utils.nms_wrapper import apply_nms_to_single_image_results
from lib.utils.rpn.bbox_transform import regress_rois


def evaluate_image(image_path, label_lookup, model, cfg, output_directory, plot_bboxes=False, with_map_eval=False):
    evaluator = FasterRCNN_Evaluator(model, cfg)
    image, scale, padding = utils.resize_and_pad(img=cv2.cvtColor(cv2.imread(image_path),
                                                                  cv2.COLOR_BGR2RGB),
                                                 width=cfg.IMAGE_WIDTH,
                                                 height=cfg.IMAGE_HEIGHT)
    cls_probs, regressed_rois, scores = predict(evaluator, image_path)
    filtered_bboxes, filtered_labels, filtered_scores = apply_non_maxima_suppression(cfg, cls_probs, regressed_rois,
                                                                                     scores)

    filtered_bboxes, filtered_labels, filtered_scores = remove_background_predictions(filtered_bboxes, filtered_labels,
                                                                                      filtered_scores, label_lookup)
    if with_map_eval:
        gt_labels, image, scaled_bboxes = get_ground_truth_info(image, scale, padding)
        utils.write_ground_truth(image_path,
                                 gt_boxes=scaled_bboxes,
                                 gt_labels=gt_labels,
                                 base_directory=output_directory)
        utils.write_prediction(image_path, filtered_bboxes, filtered_scores, filtered_labels,
                               base_directory=output_directory)
    if plot_bboxes:
        plot_predicted_bounding_boxes(filtered_bboxes, filtered_labels, filtered_scores, image)


def apply_non_maxima_suppression(cfg, cls_probs, regressed_rois, scores):
    nmsKeepIndices = apply_nms_to_single_image_results(regressed_rois, cls_probs, scores,
                                                       use_gpu_nms=cfg.USE_GPU_NMS,
                                                       device_id=cfg.GPU_ID,
                                                       nms_threshold=0.3,
                                                       conf_threshold=cfg.RESULTS_NMS_CONF_THRESHOLD)
    filtered_bboxes = regressed_rois[nmsKeepIndices]
    filtered_labels = cls_probs[nmsKeepIndices]
    filtered_scores = scores[nmsKeepIndices]
    return filtered_bboxes, filtered_labels, filtered_scores


def predict(evaluator, image_path):
    out_cls_pred, out_rpn_rois, out_bbox_regr, dims = evaluator.process_image_detailed(image_path)
    cls_probs = out_cls_pred.argmax(axis=1)
    scores = out_cls_pred.max(axis=1)
    regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, cls_probs, dims)
    return cls_probs, regressed_rois, scores


def plot_predicted_bounding_boxes(filtered_bboxes, filtered_labels, filtered_scores, image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for score, label, roi in zip(filtered_scores, filtered_labels, filtered_bboxes):
        x1, y1, x2, y2 = roi
        if not label == "__background__":
            rect = mpatches.Rectangle((x1, y1), abs(x2 - x1), abs(y2 - y1),
                                      fill=False,
                                      edgecolor='red',
                                      linewidth=1)
            ax.add_patch(rect)
            ax.annotate("{} {:.2f}%".format(label, score * 100, 2), (x1, y1), fontsize=12)
    plt.show()


def remove_background_predictions(filtered_bboxes, filtered_labels, filtered_scores, label_lookup):
    remove_indices_at = []
    for index, filtered_label in enumerate(filtered_labels):
        if int(filtered_label) == 0:
            remove_indices_at.append(index)
    bboxes_without_background = np.delete(filtered_bboxes, remove_indices_at, axis=0)
    scores_without_background = np.delete(filtered_scores, remove_indices_at, axis=0)
    labels_without_background = np.delete(filtered_labels, remove_indices_at, axis=0)
    labels_as_strings = np.array([label_lookup[label] for label in labels_without_background])
    return bboxes_without_background, labels_as_strings, scores_without_background


def get_ground_truth_info(image, scale, padding):
    gt_bboxes = list(map(lambda line: list(map(lambda string: float(string), re.findall(re.compile(r"\d+.\d+"), line))),
                         open(image_path.replace("jpg", "bboxes.tsv")).readlines()))
    scaled_bboxes = list(map(lambda box: utils.scale_bounding_box(np.array(box), scale, padding), gt_bboxes))
    gt_labels = list(
        map(lambda line: line.strip('\n'), open(image_path.replace("jpg", "bboxes.labels.tsv")).readlines()))
    return gt_labels, image, scaled_bboxes


if __name__ == "__main__":
    count = 0
    total_time = 0
    cntk.device.try_set_default_device(cntk.device.gpu(0))
    base_directory = r"D:\LEGO Vision Datasets\Detection\Faster R-CNN\Natural Data_output 500 samples"
    output_directory = os.path.join(base_directory, "results", "16-11-2018-16-02")
    label_lookup = list(map(lambda x: x.split('\t')[0],
                            open(os.path.join(base_directory, "class_map.txt")).readlines()))
    model = load_model(os.path.join(base_directory, "results", "16-11-2018-16-02",
                                    "faster_rcnn_eval_AlexNet_e2e - 500 samples natural data.model"))
    cfg = EasyDict(json.load(open(os.path.join(base_directory, "results", "16-11-2018-16-02", "settings.json"))))

    for image_path in glob.glob(os.path.join(base_directory, "testImages", "*.jpg"))[:5]:
        start = time.time()
        evaluate_image(image_path, label_lookup, model, cfg, output_directory, plot_bboxes=True, with_map_eval=False)
        total_time += time.time() - start
        count += 1
    print("Average prediction time is: {}".format(total_time / count))
