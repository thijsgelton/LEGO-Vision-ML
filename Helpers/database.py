import argparse
import glob
import os
import time

import cv2
import numpy as np
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn import preprocessing

from Helpers import helpers


def export_hog_and_color_feature_to_database(image_directory, total_number_of_samples, dimensions, host, database,
                                             collection, color_insensitive):
    client = MongoClient(host=host)
    image_collection = client.get_database(database).get_collection(collection)
    for dimension in [int(dimension.strip()) for dimension in dimensions.split(',')]:
        features_list, labels = images_to_hog_and_color_feature(image_directory, total_number_of_samples,
                                                                (dimension, dimension), color_insensitive)
        duplicates = 0
        for features, label in zip(features_list, labels):
            try:
                image_collection.insert_one(document=dict(
                    _id=hash(features.tostring()),
                    features=features.tolist(),
                    label=label,
                    shape=dimension,
                    color=not bool(color_insensitive))
                )
            except DuplicateKeyError:
                duplicates += 1
        print("Skipped {} duplicates".format(duplicates))


def images_to_hog_and_color_feature(data_dir, number_of_samples, shape, color_insensitive):
    if color_insensitive:
        converter = lambda x: 1 - rgb2gray(x)  # inverting the image makes it process faster
    else:
        converter = None
    X, y = helpers.images_to_dataset(dataset_path=data_dir,
                                     shape=shape,
                                     smoothing=0.1,
                                     denoising=0.0,
                                     with_hog_attached=True,
                                     with_dominant_color_attached=False,
                                     with_mean_color_attached=True,
                                     pixels_per_cell=(shape[0] / 8, shape[0] / 8),
                                     cells_per_block=(8, 8),
                                     orientations=9,
                                     samples=number_of_samples,
                                     converter=converter,
                                     debug=False)
    X = preprocessing.scale(X, with_mean=False)
    return X, y


def export_orb_to_database(image_directory, total_number_of_samples, dimensions, host, database,
                           collection):
    client = MongoClient(host=host)
    image_collection = client.get_database(database).get_collection(collection)
    for dimension in [int(dimension.strip()) for dimension in dimensions.split(',')]:
        features_list, labels = images_to_orb_feature(image_directory, total_number_of_samples)
        documents = []
        for index, (features, label) in enumerate(zip(features_list, labels)):
            print(index)
            documents.append(
                dict(_id=hash(features.tostring()),
                     features=features.tolist(),
                     label=label,
                     shape=dimension,
                     feature_names='orb')
            )
        image_collection.insert_many(documents)


def images_to_orb_feature(image_directory, number_of_samples=3200, shape=256):
    X = []
    y = []
    total_time = 0
    count = 0
    for subdirectory in list(os.walk(image_directory))[0][1]:
        current_image_directory = os.path.join(image_directory, subdirectory)
        for file in glob.glob(os.path.join(current_image_directory, '*.jpg'))[:number_of_samples]:
            if count % 100 == 0:
                print(count)
            start = time.time()
            descriptors = helpers.gen_sift_features(helpers.to_gray(cv2.imread(file)))[-1]
            if not isinstance(descriptors, np.ndarray):
                continue
            X.append(descriptors)
            y.append(subdirectory)
            total_time += time.time() - start
            count += 1
    print("Average time: {}".format(total_time / count))
    X = np.array(X)
    y = np.array(y)
    return X, y


def export_hog_opencv_to_database(image_directory, total_number_of_samples, dimensions, host, database,
                                  collection):
    client = MongoClient(host=host)
    image_collection = client.get_database(database).get_collection(collection)
    for dimension in [int(dimension.strip()) for dimension in dimensions.split(',')]:
        features_list, labels = images_to_hog_opencv_feature(image_directory, total_number_of_samples)
        documents = []
        for index, (features, label) in enumerate(zip(features_list, labels)):
            documents.append(
                dict(_id=hash(features.tostring()),
                     features=features.tolist(),
                     label=label,
                     shape=dimension,
                     feature_names='hog_opencv')
            )
        image_collection.insert_many(documents)


def images_to_hog_opencv_feature(image_directory, number_of_samples=3200, shape=256):
    X = []
    y = []
    total_time = 0
    count = 0
    for subdirectory in list(os.walk(image_directory))[0][1]:
        current_image_directory = os.path.join(image_directory, subdirectory)
        for file in glob.glob(os.path.join(current_image_directory, '*.jpg'))[:number_of_samples]:
            if count % 100 == 0:
                print(count)
            start = time.time()
            descriptors = helpers.hog_descriptor_opencv(helpers.to_gray(cv2.imread(file)), (shape, shape))
            X.append(np.squeeze(descriptors))
            y.append(subdirectory)
            total_time += time.time() - start
            count += 1
    print("Average time: {}".format(total_time / count))
    return np.array(X), np.array(y)


def export_hog_skimage_to_database(image_directory, total_number_of_samples, dimensions, host, database,
                                   collection):
    client = MongoClient(host=host)
    image_collection = client.get_database(database).get_collection(collection)
    for dimension in [int(dimension.strip()) for dimension in dimensions.split(',')]:
        features_list, labels = images_to_hog_skimage_feature(image_directory, total_number_of_samples)
        documents = []
        for index, (features, label) in enumerate(zip(features_list, labels)):
            documents.append(
                dict(_id=hash(features.tostring()),
                     features=features.tolist(),
                     label=label,
                     shape=dimension,
                     feature_names='hog_skimage')
            )
        image_collection.insert_many(documents, bypass_document_validation=True)


def images_to_hog_skimage_feature(image_directory, number_of_samples=3200, shape=256):
    X = []
    y = []
    total_time = 0
    count = 0
    for subdirectory in list(os.walk(image_directory))[0][1]:
        current_image_directory = os.path.join(image_directory, subdirectory)
        for file in glob.glob(os.path.join(current_image_directory, '*.jpg'))[:number_of_samples]:
            if count % 100 == 0:
                print(count)
            start = time.time()
            descriptors = hog(cv2.imread(file), pixels_per_cell=(shape / 8, shape / 8),
                              cells_per_block=(8, 8),
                              orientations=9, multichannel=True, block_norm='L2-Hys')
            X.append(descriptors)
            y.append(subdirectory)
            total_time += time.time() - start
            count += 1
    print("Average time: {}".format(total_time / count))
    return np.array(X), np.array(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_directory', type=str, help='Directory with the images')
    parser.add_argument('--total_number_of_samples', type=int, help='Total amount of samples per class',
                        default=3200)
    parser.add_argument('--comma_separated_dimensions', type=str, help='Pixel by pixel resolutions,'
                                                                       ' comma separated list',
                        default="256")
    parser.add_argument('--database_host', default="localhost")
    parser.add_argument('--database_name', default="lego_vision")
    parser.add_argument('--collection_name', default="natural_data_hog")
    args = parser.parse_args()
    export_hog_and_color_feature_to_database(
        image_directory=r"D:\LEGO Vision Datasets\Detection\SVM\Augmented Hard Negative Mining",
        total_number_of_samples=1600,
        dimensions="256",
        host="localhost",
        database="lego_vision",
        collection="natural_hog_dom_negative_mining_mean_color",
        color_insensitive=False
    )
