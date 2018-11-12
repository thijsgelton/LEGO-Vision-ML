import re
from functools import partial
from typing import Iterable, Callable

import cv2
import numpy as np
import selectivesearch
from sklearn import preprocessing
from sklearn.externals import joblib

from Helpers import helpers, utils
from Helpers.utils import resize_and_pad, correct_bounding_boxes, w_h_to_x2_y2, display_bounding_boxes,\
    x2_y2_to_w_h, apk

only_numbers = re.compile(r"\d+\.\d+")

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

    def __init__(self, image: np.ndarray, classifier, gt_boxes: np.ndarray, gt_labels: list, preprocessor: Callable,
                 shape: tuple, iou_threshold=0.5):
        self.image = image
        self.preprocessor = preprocessor
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.iou_threshold = iou_threshold
        self.classifier = classifier
        self.shape = shape

    def predictions(self):
        predictions = []
        for bbox in self.predicted_bounding_boxes():
            x1, y1, x2, y2 = bbox
            prediction = self.predict(self.image[y1:y2, x1:x2, :])
            predictions.append(dict(prediction=prediction, x1=x1, y1=y1, x2=x2, y2=y2))
        return predictions

    def predict(self, sub_image):
        preprocessed = self.preprocessor(sub_image, self.shape)
        probability_distribution = self.classifier.predict_proba([preprocessed])[0]
        return probability_distribution

    def predicted_bounding_boxes(self):
        labels, regions = selectivesearch.selective_search(self.image, scale=500, min_size=600, sigma=1.5)
        bounding_boxes = list(map(lambda region: w_h_to_x2_y2(region['rect']), regions))
        bounding_boxes = self.remove_too_large(bounding_boxes)
        return self.filter_via_iou(bounding_boxes)

    def remove_too_large(self, bounding_boxes: Iterable):
        return list(filter(
            lambda box: x2_y2_to_w_h(box)[2] < self.image.shape[0] / 2 and x2_y2_to_w_h(box)[3] < self.image.shape[
                1] / 2,
            bounding_boxes))

    def filter_via_iou(self, bounding_boxes):
        return utils.intersection_over_union_removal(self.gt_boxes, bounding_boxes, self.iou_threshold)


def preproccesing(image, shape):
    preprocessed = helpers.pipeline(image, shape=shape, smoothing=0.1, with_hog_attached=True,
                                    with_dominant_color_attached=True, pixels_per_cell=(shape[0] / 8, shape[0] / 8),
                                    cells_per_block=(8, 8), debug=False)
    preprocessed = preprocessing.scale(preprocessed, with_mean=False)
    return preprocessed


if __name__ == "__main__":
    image_path = r"D:\LEGO Vision Datasets\Detection\Natural Data_output\testImages\IMG_20181105_092359.jpg"
    bbox_path = r"D:\LEGO Vision Datasets\Detection\Natural Data_output\testImages\IMG_20181105_092359.bboxes.tsv"
    labels_path = r"D:\LEGO Vision Datasets\Detection\Natural Data_output\testImages\IMG_20181105_092359.bboxes.labels.tsv"
    image, scale, padding = resize_and_pad(cv2.cvtColor(cv2.imread(image_path),
                                                        cv2.COLOR_BGR2RGB), width=1024, height=1024)
    gt_bounding_boxes = np.array(list(map(lambda line: np.array(list(map(lambda string: float(string),
                                                                         re.findall(only_numbers, line)))),
                                          open(bbox_path).readlines())))
    gt_bounding_boxes = np.array(
        list(map(lambda box: correct_bounding_boxes(np.array(box), scale, scale, padding), gt_bounding_boxes)))
    gt_labels = list(map(lambda line: line.strip('\n'), open(labels_path).readlines()))
    classifier = joblib.load(r"D:\Projects\LEGO Vision\Classification\Final models\SVM\SVM-3200-256W.joblib")
    shape = (256, 256)
    display_bounding_boxes(image, gt_bounding_boxes, gt_labels)
    detector = SelectiveSearchObjectDetector(image,
                                             classifier=classifier,
                                             gt_boxes=gt_bounding_boxes,
                                             gt_labels=gt_labels,
                                             preprocessor=partial(preproccesing),
                                             shape=shape,
                                             iou_threshold=0.4)
    label_lookup = sorted(map(lambda x: x.split('\t')[0],
                              open(r"D:\LEGO Vision Datasets\Detection\Natural Data_output\class_map.txt").readlines()[1:]))
    predictions = detector.predictions()
    bboxes = []
    labels = []
    scores = []
    for prediction in predictions:
        bboxes.append((prediction['x1'], prediction['y1'], prediction['x2'], prediction['y2']))
        labels.append(label_lookup[prediction['prediction'].argmax(axis=0)])
        scores.append(max(prediction['prediction']))

    display_bounding_boxes(image,
                           bounding_boxes=np.array(bboxes),
                           labels=labels,
                           scores=scores)
