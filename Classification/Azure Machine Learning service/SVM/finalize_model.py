import time

import os
from pymongo import MongoClient
from sklearn import svm
from sklearn.externals import joblib

from Helpers import helpers


def finalize_model(natural_data_collection, export_path, number_of_samples, shape):
    start = time.time()
    print("Start retrieving data..")
    X, y = helpers.get_data(natural_data_collection, number_of_samples, shape, color_insensitive=False)
    print(f"Total time for retrieval: {time.time() - start}")
    classifier = svm.SVC(C=100, gamma=0.001, kernel='rbf', probability=True)
    print("Starting training...")
    start = time.time()
    classifier.fit(X, y)
    print(f"Total time for training: {time.time() - start}")
    joblib.dump(value=classifier, filename=export_path)


def main():
    client = MongoClient(host="localhost")
    natural_data = client.get_database("lego_vision").get_collection("natural_data_hog_dom")
    export_path = os.path.abspath(os.path.join("..", "..", "Final models", "SVM", "SVM-3200-64W.joblib"))
    print(f"Exporting model to {export_path} after training")
    finalize_model(natural_data, export_path, number_of_samples=3200, shape=64)


if __name__ == "__main__":
    main()
