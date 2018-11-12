import cv2

from Helpers.utils import resize_and_pad, scale_bounding_boxes
import numpy as np
import re


class SelectiveSearchEvaluator:

    def __init__(self, image_paths, gt_bbox_paths, gt_labels_path, object_detector, evaluation_dimension=1024):
        self.image_paths = image_paths
        self.gt_bbox_paths = gt_bbox_paths
        self.gt_labels_path = gt_labels_path
        self.object_detector = object_detector
        self.evaluation_dimension = evaluation_dimension
        self.only_numbers = re.compile(r"\d+.\d+")

    def eval(self):
        for image_path, gt_bbox_path, gt_labels_path in zip(self.image_paths, self.gt_bbox_paths, self.gt_labels_path):
            pass

    def read_image(self, image_path):
        image, scale, padding = resize_and_pad(img=cv2.cvtColor(cv2.imread(image_path),
                                                                cv2.COLOR_BGR2RGB),
                                               width=self.evaluation_dimension,
                                               height=self.evaluation_dimension)
        return image, scale, padding

    def read_bboxes(self, gt_bbox_path, scale, padding):
        gt_bounding_boxes = np.array(list(map(lambda line: np.array(list(map(lambda string: float(string),
                                                                             re.findall(self.only_numbers, line)))),
                                              open(gt_bbox_path).readlines())))
        gt_bounding_boxes = np.array(
            list(map(lambda box: scale_bounding_boxes(np.array(box), scale, scale, padding), gt_bounding_boxes)))
        return gt_bounding_boxes
