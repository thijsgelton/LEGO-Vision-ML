import time

import os
from collections import Counter

from pymongo import MongoClient
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
from Helpers import helpers


def full(class_counter, number_of_samples):
    full = True
    for class_, count in class_counter.items():
        if count < number_of_samples:
            full = False
    return full


def finalize_model(source, export_path, number_of_samples, shape):
    start = time.time()
    print("Start retrieving data..")
    if isinstance(source, list):
        divided = number_of_samples // len(source)
        X = []
        y = []
        for src in source:
            temp_X, temp_y = helpers.get_data(src, divided, shape, color_insensitive=False)
            X.extend(temp_X.tolist())
            y.extend(temp_y)
        X = np.array(X)
    else:
        X, y = helpers.get_data(source, number_of_samples, shape, color_insensitive=False)
    print("Total time for retrieval: {}".format(time.time() - start))
    classifier = svm.SVC(C=100, gamma=0.001, kernel='rbf', probability=True)
    print("Starting training...")
    start = time.time()
    classifier.fit(X, y)
    print("Total time for training: {}".format(time.time() - start))
    joblib.dump(value=classifier, filename=export_path)


def main():
    client = MongoClient(host="localhost")
    natural_data = client.get_database("lego_vision").get_collection("natural_data_hog_dom")
    natural_data_hard_negative_mined = client.get_database("lego_vision").get_collection(
        "natural_hog_dom_negative_mining")
    export_path = os.path.abspath(os.path.join("..", "..", "Final models", "SVM", "SVM-3200-256W-HNM.joblib"))
    print("Exporting model to {} after training".format(export_path))
    finalize_model([natural_data, natural_data_hard_negative_mined], export_path, number_of_samples=3200, shape=256)


if __name__ == "__main__":
    main()
