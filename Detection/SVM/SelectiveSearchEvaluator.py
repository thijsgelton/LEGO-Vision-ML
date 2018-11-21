import os
import re
import uuid
from functools import partial

import cv2
import imageio
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from tqdm import tqdm

from Helpers import helpers, utils
from Detection.SVM.SelectiveSearchObjectDetector import SelectiveSearchObjectDetector


class SelectiveSearchEvaluator:

    def __init__(self, image_paths, gt_bbox_paths, gt_labels_path, classifier, label_lookup,
                 selective_search_config, image_dimension=1024,
                 classifier_dimension=256, plot_every_n_images=None,
                 hard_negative_mining_directory=None, output_directory=None):
        self.image_paths = image_paths
        self.gt_bbox_paths = gt_bbox_paths
        self.gt_labels_path = gt_labels_path
        self.classifier = classifier
        self.image_dimension = image_dimension
        self.classifier_dimension = classifier_dimension
        self.only_numbers = re.compile(r"\d+.\d+")
        self.label_lookup = label_lookup
        self.plot_every_n_images = plot_every_n_images
        self.hard_negative_mining_directory = hard_negative_mining_directory
        self.output_directory = output_directory
        self.selective_search_config = selective_search_config

    def eval(self):
        for index, image_path, in tqdm(enumerate(self.image_paths), total=len(self.image_paths)):
            image, scale, padding = self.read_image(image_path)
            gt_bboxes = self.read_bboxes(self.gt_bbox_paths[index], scale, padding)
            gt_labels = self.read_labels(self.gt_labels_path[index])
            plot = self.plot_every_n_images and index % self.plot_every_n_images == 0
            if plot:
                utils.display_bounding_boxes(image, gt_bboxes, gt_labels)
            detector = SelectiveSearchObjectDetector(image=image,
                                                     label_lookup=self.label_lookup,
                                                     classifier=self.classifier,
                                                     gt_boxes=gt_bboxes,
                                                     gt_labels=gt_labels,
                                                     preprocessor=partial(self.preproccesing),
                                                     shape=[self.classifier_dimension] * 2,
                                                     iou_threshold=0.5,
                                                     selective_search_config=self.selective_search_config)
            bboxes, scores, labels = detector.predictions()
            if plot:
                utils.display_bounding_boxes(image, bboxes, labels, scores)
            actual, predicted, actual_bboxes = self.get_actual_and_predicted_per_bbox(gt_bboxes, gt_labels, bboxes,
                                                                                      labels, scores)
            if self.hard_negative_mining_directory:
                self.hard_negative_mining(image, actual, predicted, actual_bboxes)
            if self.output_directory:
                utils.write_ground_truth(image_path, gt_bboxes, gt_labels, base_directory=self.output_directory)
                utils.write_prediction(image_path, bboxes, scores, labels, base_directory=self.output_directory)

    @staticmethod
    def get_actual_and_predicted_per_bbox(gt_bounding_boxes, gt_labels, pred_bounding_boxes, pred_labels, scores):
        actual = []
        predicted = []
        gt_bboxes = []
        for gt_index, gt_box in enumerate(gt_bounding_boxes):
            for pred_index, predicted_bbox in enumerate(pred_bounding_boxes):
                if utils.calculate_intersection_over_union(gt_box, predicted_bbox) > 0.4:
                    actual.append(gt_labels[gt_index])
                    predicted.append(dict(label=pred_labels[pred_index], score=scores[pred_index]))
                    gt_bboxes.append(gt_box)
        return actual, predicted, gt_bboxes

    def read_image(self, image_path):
        if self.image_dimension != -1:
            image, scale, padding = utils.resize_and_pad(img=cv2.cvtColor(cv2.imread(image_path),
                                                                          cv2.COLOR_BGR2RGB),
                                                         width=self.image_dimension,
                                                         height=self.image_dimension)
        else:
            image = cv2.cvtColor(cv2.imread(image_path),
                                 cv2.COLOR_BGR2RGB)
            scale = 1.0
            padding = (0, 0, 0, 0)
        return image, scale, padding

    def read_bboxes(self, gt_bbox_path, scale, padding):
        bbox_lines = open(gt_bbox_path).readlines()
        gt_bounding_boxes = []
        for bbox_line in bbox_lines:
            bbox = np.array(list(map(lambda string: float(string), re.findall(self.only_numbers, bbox_line))))
            gt_bounding_boxes.append(utils.scale_bounding_box(bbox, scale, padding))
        return np.array(gt_bounding_boxes)

    @staticmethod
    def read_labels(labels_path):
        labels = list(map(lambda line: line.strip('\n'), open(labels_path).readlines()))
        return labels

    @staticmethod
    def preproccesing(image, shape):
        preprocessed = helpers.pipeline(image, shape=shape, smoothing=0.1, with_hog_attached=True,
                                        with_dominant_color_attached=True, pixels_per_cell=(shape[0] / 8, shape[0] / 8),
                                        cells_per_block=(8, 8), debug=False)
        preprocessed = preprocessing.scale(preprocessed, with_mean=False)
        return preprocessed

    def hard_negative_mining(self, image, actual, predicted, actual_bboxes):
        for index, (a, p) in enumerate(zip(actual, list(map(lambda x: x['label'], predicted)))):
            if a != p:
                self.extract_sub_image_for_training(image, actual_bboxes[index], a)

    def extract_sub_image_for_training(self, image, bbox, label):
        label_directory = os.path.join(self.hard_negative_mining_directory, label)
        os.makedirs(label_directory, exist_ok=True)
        imageio.imwrite(os.path.join(label_directory, str(uuid.uuid4()) + '.jpg'),
                        self.cut_out_sub_image(bbox, image, add_padding=False))

    def cut_out_sub_image(self, bbox, image, add_padding=True):
        if add_padding:
            x1, y1, w, h = utils.x2_y2_to_w_h(utils.reshape_bbox(bbox, [self.classifier_dimension] * 2))
            move_up = int(h * 0.50) if y1 - int(h * 0.50) > 0 else 0
            move_left = int(w * 0.50) if x1 - int(w * 0.50) > 0 else 0
            scale_down = int(h * 1.50) if y1 + int(h * 1.50) < image.shape[0] else abs(y1 - image.shape[0])
            scale_right = int(w * 1.50) if x1 + int(w * 1.50) < image.shape[1] else abs(x1 - image.shape[1])
            return image[y1 - move_up:y1 + scale_down, x1 - move_left:x1 + scale_right, :]
        else:
            x1, y1, w, h = utils.x2_y2_to_w_h(utils.reshape_bbox(bbox, [self.classifier_dimension] * 2))
            return image[y1:y1 + h, x1:x1 + w, :]


if __name__ == "__main__":
    from Detection.SVM.config.detection_natural_500 import *

    selective_search_config = dict(
        scale=350,
        min_size=400,
        sigma=0.7
    )

    evaluator = SelectiveSearchEvaluator(
        image_paths=image_paths,
        gt_bbox_paths=gt_bbox_paths,
        gt_labels_path=gt_labels_paths,
        classifier=joblib.load(classifier),
        label_lookup=label_lookup,
        plot_every_n_images=70,
        image_dimension=1024,
        # output_directory=output_directory,
        hard_negative_mining_directory=r"D:\LEGO Vision Datasets\Detection\SVM\Hard Negative Mining 2",
        selective_search_config=selective_search_config
    )
    evaluator.eval()
