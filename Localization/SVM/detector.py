import glob
import logging
import os

import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from Helpers.enum_validation import Validation

from Helpers.utils import cannify_with_opencv


class SVMDetector:
    def __init__(self, shape, model_path=None, validation_method=Validation.K_FOLD, name="SVMDetector",
                 feature_extractor=cannify_with_opencv):
        self.feature_extractor = feature_extractor
        self.images = []
        self.labels = []
        self.classes = []
        self.validation_method = validation_method
        self.name = name
        self.shape = shape
        self.logger = self.active_logger()
        if model_path:
            self.detector = joblib.load(model_path)
        else:
            self.detector = SVC(gamma=0.1, kernel='rbf', C=1.0, probability=True, tol=0.001)

    def train_to_finalize(self):
        if not len(self.images) or not len(self.labels):
            raise ValueError("Can not finalize: The data has not been set yet. Use set_data(images, labels).")
        self.detector.fit(self.images, self.labels)

    def train_to_validate(self):
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        if not len(self.images) or not len(self.labels):
            raise ValueError("Can not validate: The data has not been set yet. Use set_data(images, labels).")
        if self.validation_method:
            f_score = self.validation_method(algorithm=self.detector, images=self.images, labels=self.labels)
            self.log(f"{self.validation_method.name} method with F-score of: {f_score}")

    def add_sample(self, image, label, **kwargs):
        self.images.append(self.pipeline(image, **kwargs))
        self.labels.append(label)

    @staticmethod
    def set_classes(y):
        if len(list(set(y))) > 2:
            raise AssertionError("For a detector to work best, you should consider only using positive and negative"
                                 " classes.")
        return list(set(y))

    def predict(self, image, with_probability=True):
        self.set_classes(self.labels)
        if with_probability:
            return zip(self.classes, self.detector.predict_proba(image)[0])
        else:
            return self.detector.predict(image)

    def log(self, message, level=logging.DEBUG):
        self.logger.log(level=level, msg=message)

    def save(self, path):
        joblib.dump(self.detector, os.path.join(path, f"{self.name}.joblib"))

    @staticmethod
    def active_logger():
        logger = logging.getLogger("SVMDetector")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)
        return logger

    def pipeline(self, image, **kwargs):
        if len(image.shape) > 2 and image.shape[-1] > 1:
            raise AssertionError("Expected grayscale image.")
        if not self.shape:
            raise AssertionError("Set the shape for the detector in order to proceed.")
        image = cv2.resize(image, dsize=self.shape)
        return self.feature_extractor(image, **kwargs)


if __name__ == "__main__":
    shape = (24, 24)
    detector = SVMDetector(shape=shape, validation_method=Validation.STRAT_K_FOLD)
    negative_image_dir = r"D:\LEGO Vision Datasets\Positive and Negative Samples for Linear SVM\neg"
    positive_image_dir = r"D:\LEGO Vision Datasets\classification-natural-data"

    positive_sample_count = 0
    for _class in list(os.walk(positive_image_dir))[0][1]:
        for positive_sample in glob.glob(os.path.join(positive_image_dir, _class, '*.jpg'))[:3200]:
            positive_sample_count += 1
            visualize = False
            if positive_sample_count % 100 == 0:
                print(positive_sample_count)
            detector.add_sample(cv2.imread(positive_sample, 0), label=1, visualize=visualize)

    negative_sample_count = 0
    while negative_sample_count < positive_sample_count:
        for negative_sample in glob.glob(os.path.join(negative_image_dir, '*.jpg'))[:positive_sample_count]:
            negative_sample_count += 1
            visualize = False
            if negative_sample_count % 100 == 0:
                print(negative_sample_count)
            if negative_sample_count == positive_sample_count:
                break
            detector.add_sample(cv2.imread(negative_sample, 0), label=0, visualize=visualize)
    detector.train_to_validate()
