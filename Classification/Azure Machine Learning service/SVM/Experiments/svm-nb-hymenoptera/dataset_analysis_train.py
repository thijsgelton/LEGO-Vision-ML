import argparse
import os
import numpy as np
import time

from sklearn import svm
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

import helpers

from azureml.core.run import Run

# get the Azure ML run object
run = Run.get_submitted_run()


def get_data(data_dir, number_of_samples, shape):
    X, y = helpers.images_to_dataset(dataset_path=data_dir, to_grayscale=False, shape=shape,
                                     smoothing=0.0, to_hog=False, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
                                     orientations=9,
                                     denoising=0.0, samples=number_of_samples)
    return X, y


def train(X, y, classifier_name, number_of_samples, shape, output_directory, beta):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    classifier = get_classifier(classifier_name)

    start = time.time()
    classifier.fit(X_train, y_train)
    endtime = time.time() - start
    run.log('training_time', np.float(endtime))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    run.log('accuracy', np.float(accuracy))

    fscore = fbeta_score(y_test, y_pred, beta=beta, average='macro')
    run.log('f_score', np.float(fscore))
    helpers.plot_confusion_matrix_with_acc_and_fbeta(y_test, y_pred, classes=set(y_test), normalize=True,
                                                     save_path=f"{output_directory}/"
                                                               f"{classifier_name}-{number_of_samples}-{shape}-cm.png",
                                                     fbeta_beta=beta)


def get_classifier(classifier: str):
    classifiers = {'svm': svm.SVC(C=100, gamma=0.001, kernel='rbf'),
                   'gaussian': GaussianNB(),
                   'multinomial': MultinomialNB(),
                   'bernoulli': BernoulliNB()}
    return classifiers.get(classifier)


def main():
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='directory of training data')
    parser.add_argument('--output_dir', type=str, help='output directory', default="./outputs")
    parser.add_argument('--classifier', type=str, default='svm')
    parser.add_argument('--number_of_samples', type=int, help='amount of training samples', default="400")
    parser.add_argument('--shape', help='the width and height e.g. (128,128)', type=int, default='64')
    parser.add_argument('--fbeta_beta', type=float, default='0.5')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("data directory is: " + args.data_dir)
    shape = (args.shape, args.shape)
    run.log('shape', args.shape)
    run.log('number_of_samples', args.number_of_samples)
    run.log('fbeta_beta', args.fbeta_beta)
    run.log('classifier', args.classifier)

    X, y = get_data(args.data_dir, args.number_of_samples, shape=shape)
    train(X, y, args.classifier, args.number_of_samples, shape=shape, output_directory=args.output_dir,
          beta=args.fbeta_beta)


if __name__ == "__main__":
    main()
