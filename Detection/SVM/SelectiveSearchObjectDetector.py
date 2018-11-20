import re
from typing import Callable, List

import numpy as np

from Helpers import utils, selectivesearch
from Helpers.utils import w_h_to_x2_y2, x2_y2_to_w_h
from Detection.CNN.FasterRCNN.lib.utils.nms_wrapper import apply_nms_to_single_image_results

"""
Class that has a constructor which needs:
    1. Path to an image.
    2. List of ground truth bounding boxes.
    3. List of ground truth labels.
    4. IoU threshold.
    
and that will:
    1. Guess bounding boxes based on a selective search algorithm.
    2. Will filter bounding boxes based on a max_width/max_height, a IoU threshold.
    3. Output predicted label with bounding box.    
"""


class SelectiveSearchObjectDetector:

    def __init__(self, label_lookup: list, image: np.ndarray, classifier, gt_boxes: np.ndarray, gt_labels: list,
                 preprocessor: Callable, shape, selective_search_config: dict, iou_threshold=0.5):
        self.image = image
        self.preprocessor = preprocessor
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.iou_threshold = iou_threshold
        self.only_numbers = re.compile(r"\d+\.\d+")
        self.classifier = classifier
        self.label_lookup = label_lookup
        self.shape = shape
        self.selective_search_config = selective_search_config

    def predictions(self):
        bboxes = []
        labels = []
        scores = []
        predictions = self.predicted_bounding_boxes()
        for bbox in predictions:
            x1, y1, x2, y2 = utils.reshape_bbox(bbox, self.shape)
            prediction = self.predict(self.image[y1:y2, x1:x2, :])
            bboxes.append((x1, y1, x2, y2))
            scores.append(max(prediction))
            labels.append(prediction.argmax(axis=0))
        print("Before NMS {} Bounding boxes".format(len(bboxes)))
        keep_indices = apply_nms_to_single_image_results(coords=bboxes, labels=labels, scores=scores, use_gpu_nms=True,
                                                         device_id=0, conf_threshold=0.6)
        bboxes = np.array(bboxes)[keep_indices]
        labels = np.array(list(map(lambda x: self.label_lookup[x], labels)))[keep_indices]
        scores = np.array(scores)[keep_indices]

        print("After NMS {} Bounding boxes".format(len(bboxes)))
        return bboxes, scores.tolist(), labels.tolist()

    def predict(self, sub_image):
        preprocessed = self.preprocessor(sub_image, self.shape)
        probability_distribution = self.classifier.predict_proba([preprocessed])[0]
        return probability_distribution

    def predicted_bounding_boxes(self):
        regions = self.selective_search()
        bounding_boxes = list(map(lambda region: w_h_to_x2_y2(region['rect']), regions))
        bounding_boxes = self.remove_too_large(bounding_boxes)
        return bounding_boxes

    def selective_search(self):
        labels, regions = selectivesearch.selective_search(self.image,
                                                           scale=self.selective_search_config['scale'],
                                                           min_size=self.selective_search_config['min_size'],
                                                           sigma=self.selective_search_config['sigma'])
        return regions

    def remove_too_large(self, bounding_boxes: List):
        without_large_bouding_boxes = []
        for bbox in bounding_boxes:
            x1, y1, w, h = x2_y2_to_w_h(bbox)
            if not int(w) > self.image.shape[0] / 2 and not int(h) > self.image.shape[1] / 2:
                without_large_bouding_boxes.append(bbox)
        return without_large_bouding_boxes

    def filter_via_iou(self, bounding_boxes):
        return utils.intersection_over_union_removal(self.gt_boxes, bounding_boxes, self.iou_threshold)
