# -*- coding: utf-8 -*-

""""
 Needs some tweaking, doesnt work correctly yet.
"""

from __future__ import (
    division,
    print_function,
)

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import selectivesearch
import sys
import numpy as np
import time

from skimage.feature import hog

from Localization.SVM.detector import SVMDetector
from sklearn import preprocessing
from Helpers import helpers, utils

from sklearn.externals import joblib


def main(classifier, detector):
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # read image
    img = cv2.imread(sys.argv[1])[:, :, ::-1]
    new_height = 500
    new_width = int(img.shape[1] * 500 / img.shape[0])
    img = cv2.resize(img, (new_width, new_height))
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(img, scale=100, sigma=3.5, min_size=200)
    regions = utils.non_max_suppression(np.array([region['rect'] for region in regions]), overlap_thresh=.1)
    candidates = set()
    for x, y, w, h in regions:
        # excluding regions smaller than 2000 pixels
        # x, labels, w, h = r['rect']
        if 400 < w or 400 < h:
            continue
        # distorted rects
        # if w / h > 1.5 or h / w > 1.5:
        #     continue
        candidates.add((x, y, w, h))
    # draw rectangles on the original image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for x, y, w, h in candidates:
        if h > w:
            window_size = h
        else:
            window_size = w
        # window_size = int(window_size * 1.5)
        # red_block = img[labels:labels + window_size, x:x + window_size, :]
        # prediction = detector.predict([convert_to_skimage_hog(red_block)], with_probability=False)
        # if prediction[0]:
        rect = mpatches.Rectangle((x, y), window_size, window_size, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        # classify(ax, classifier, red_block, x, labels)
    plt.show()


def convert_to_hog(image):
    image = cv2.resize(image, (256, 256))
    descriptor = np.squeeze(helpers.hog_descriptor_opencv(helpers.to_gray(image), image.shape[:2])).tolist()
    return descriptor


def convert_to_skimage_hog(image):
    shape = (256, 256)
    image = cv2.resize(image, shape)
    descriptors, image = hog(image, pixels_per_cell=(shape[0] / 8, shape[0] / 8),
                             cells_per_block=(8, 8),
                             orientations=9, multichannel=True, block_norm='L2-Hys', visualize=True)
    plt.imshow(image)
    return descriptors


def classify(ax, classifier, red_block, x, y):
    label_lookup = ['2458-blue', '3003-yellow', '3020-red', '3039-trans-clear', '3298-yellow', '3710-red',
                    '6041-yellow']
    preproccesed = helpers.pipeline(red_block, shape=(256, 256), smoothing=0.1, with_hog_attached=True,
                                    with_dominant_color_attached=True, pixels_per_cell=(32, 32),
                                    cells_per_block=(8, 8), debug=False)
    preproccesed = preprocessing.scale(preproccesed, with_mean=False)
    # prediction = np.asarray(classifier.predict_proba([preproccesed]), dtype=np.float32)
    prediction = classifier.predict_proba([preproccesed])[0]
    # if max(prediction) > 0.95:
    #     print(max(prediction))
    ax.annotate(label_lookup[np.argmax(prediction)], (x, y))


if __name__ == "__main__":
    start = time.time()
    np.set_printoptions(suppress=True)
    detector = SVMDetector(model_path="./SVMDetector-SkimageHog.joblib", y=[0, 1])
    classifier = joblib.load(r"D:\Projects\LEGO Vision\Classification\Final models\SVM\SVM-3200-256W.joblib")
    main(classifier, detector)
    print(f"Time: {time.time() - start}")
