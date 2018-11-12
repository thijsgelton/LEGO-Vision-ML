import argparse
import os
import numpy as np
import time
import csv
import sklearn
from datetime import datetime
from sklearn import preprocessing
from skimage.color import rgb2gray
from sklearn import svm
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.externals import joblib
from sklearn.utils import shuffle
from pymongo import MongoClient
import helpers

from azureml.core.run import Run


# get the Azure ML run object


def get_data(data_dir, number_of_samples, shape, color_insensitive):
    if color_insensitive:
        converter = lambda x: 1 - rgb2gray(x)  # inverting the image makes it process faster
    else:
        converter = None
    X, y = helpers.images_to_dataset(dataset_path=data_dir,
                                     shape=shape,
                                     smoothing=0.1,
                                     denoising=0.0,
                                     with_hog_attached=True,
                                     with_dominant_color_attached=True,
                                     pixels_per_cell=(shape[0] / 8, shape[0] / 8),
                                     cells_per_block=(8, 8),
                                     orientations=9,
                                     samples=number_of_samples,
                                     converter=converter,
                                     debug=False)
    X = preprocessing.scale(X, with_mean=False)
    return X, y


def k_fold_crossvalidation(classifier, X, y):
    accuracy_scores = 0
    f_scores = 0
    y_pred_list = []
    y_test_list = []
    for train, test in KFold(n_splits=5).split(X=X, y=y):
        X_train, y_train = X[train], y[train]
        classifier.fit(X_train, y_train)
        X_test, y_test = X[test], y[test]
        y_pred = classifier.predict(X_test)
        y_pred_list.extend(y_pred)
        y_test_list.extend(y_test)
        accuracy_scores += accuracy_score(y_pred=y_pred, y_true=y_test)
        f_scores += fbeta_score(y_test, y_pred, beta=0.5, average='macro')
    return accuracy_scores / 5, f_scores / 5, y_pred_list, y_test_list


def train(X_train, y_train, X_test, y_test, classifier_name, number_of_samples, shape, output_directory,
          beta, color_insensitive, is_local, run):
    classifier = get_classifier(classifier_name)

    start = time.time()
    # classifier.fit(X_train, y_train)
    accuracy, f_score, y_preds, y_tests = k_fold_crossvalidation(classifier, np.concatenate((X_train, X_test)),
                                                                 np.asarray(y_train + y_test))
    training_time = time.time() - start

    # y_pred = classifier.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)

    # fscore = fbeta_score(y_test, y_pred, beta=beta, average='macro')
    color_insensitive = 'color_insensitive' if color_insensitive else 'color_sensitive'
    helpers.plot_confusion_matrix_with_acc_and_fbeta(y_tests, y_preds, classes=set(y_test), normalize=True,
                                                     save_path=f"{output_directory}/{classifier_name}-hog(ppc="
                                                               f"{(shape[0] / 16, shape[0] / 16)},"f"cpb={(16, 16)})"
                                                               f"-{color_insensitive}-{shape}-{number_of_samples}-cm.png",
                                                     fbeta_beta=beta)
    if not is_local and run:
        run.log('classifier_name', np.float(training_time))
        run.log('training_time', np.float(training_time))
        run.log('accuracy', np.float(accuracy))
        run.log('f_score', np.float(f_score))
        run.log('number_of_samples', number_of_samples)
        run.log('shape_and_samples', int(str(number_of_samples) + str(shape[0])))
        run.log('color_insensitive', color_insensitive)

    with open(f'{output_directory}/results-{datetime.now().date()}.csv', 'a', newline='') as csvfile:
        results = {'classifier': classifier_name,
                   'number_of_samples': number_of_samples,
                   'shape': shape,
                   'color_insensitive': color_insensitive,
                   'accuracy': accuracy,
                   'f_score': f_score}
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writerow(results)

    joblib.dump(value=classifier,
                filename=f'{output_directory}/{classifier_name}-hog(ppc={(shape[0] / 16, shape[0] / 16)},'
                         f'cpb={(16, 16)})-{color_insensitive}-{shape}-{number_of_samples}.joblib')


def get_classifier(classifier: str):
    classifiers = {'svm': svm.SVC(C=100, gamma=0.001, kernel='rbf', probability=True),
                   'gaussian': GaussianNB(),
                   'multinomial': MultinomialNB(),
                   'bernoulli': BernoulliNB()}
    return classifiers.get(classifier)


def log(run, metric_name, metric_value):
    if run:
        run.log(metric_name, metric_value)
    else:
        print(metric_name, metric_value)


def main():
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='directory of training data')
    parser.add_argument('--test_dir', type=str, help='directory of test data', default=None)
    parser.add_argument('--output_dir', type=str, help='output directory', default="./outputs")
    parser.add_argument('--classifier', type=str, default='svm')
    parser.add_argument('--number_of_samples', help='the width and height e.g. (128,128)', type=int, default='32')
    parser.add_argument('--fbeta_beta', type=float, default='0.5')
    parser.add_argument('--is_local', type=bool, default=False)
    args = parser.parse_args()
    with open(f'{args.output_dir}/results-{datetime.now().date()}.csv', 'w') as csvfile:
        fieldnames = ['classifier', 'number_of_samples', 'shape', 'color_insensitive', 'accuracy', 'f_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    run = None
    if not args.is_local:
        run = Run.get_submitted_run()
        run.log('data_dir', args.data_dir)
        run.log('test_dir', args.test_dir)
        run.log('fbeta_beta', args.fbeta_beta)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'scikit-learn version: {sklearn.__version__}')
    print("data directory is: " + args.data_dir)

    client = MongoClient(host="localhost", port=27017)
    natural_data = client.get_database("lego_vision").get_collection("natural_data_hog_dom")
    synthetic_data = client.get_database("lego_vision").get_collection("synthetic_data_hog_dom")

    for shape in ['32', '64', '128', '256']:
        shape = (int(shape), int(shape))
        for color_insensitive in [0, 1]:
            start = time.time()
            if args.test_dir:
                X_train_all, y_train_all = get_data(args.data_dir, int(args.number_of_samples * 0.7),
                                                    shape, color_insensitive)
                X_test_all, y_test_all = get_data(args.test_dir, int(args.number_of_samples * 0.3),
                                                  shape, color_insensitive)
            else:
                X, y = get_data(args.data_dir, args.number_of_samples, shape, color_insensitive)

            loading_time = time.time() - start
            log(run, 'data_loading_time', loading_time)
            for classifier in ['svm', 'multinomial']:
                for number_of_samples in [args.number_of_samples / 8, args.number_of_samples / 4,
                                          args.number_of_samples / 2, args.number_of_samples]:
                    print(number_of_samples, shape, color_insensitive)
                    if args.test_dir:
                        X_train_sliced, y_train_sliced = shuffle(X_train_all, y_train_all,
                                                                 n_samples=int(number_of_samples * .7))
                        X_test_sliced, y_test_sliced = shuffle(X_test_all, y_test_all,
                                                               n_samples=int(number_of_samples * .3))
                    else:
                        X_sliced, y_sliced = shuffle(X, y, n_samples=int(number_of_samples))
                        X_train_sliced, X_test_sliced, y_train_sliced, y_test_sliced = train_test_split(X_sliced,
                                                                                                        y_sliced,
                                                                                                        test_size=0.3,
                                                                                                        random_state=42)
                    train(X_train_sliced, y_train_sliced, X_test_sliced, y_test_sliced, classifier,
                          number_of_samples=int(number_of_samples), shape=shape, output_directory=args.output_dir,
                          beta=args.fbeta_beta, color_insensitive=color_insensitive, is_local=args.is_local, run=run)


if __name__ == "__main__":
    main()
