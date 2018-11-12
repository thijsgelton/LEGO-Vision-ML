import argparse
import os
import numpy as np
import time
import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.externals import joblib
from sklearn.utils import shuffle
import helpers

from azureml.core.run import Run

# get the Azure ML run object
run = Run.get_submitted_run()


def get_data(data_dir, number_of_samples, shape, with_dominant_color_attached):
    X, y = helpers.images_to_dataset(dataset_path=data_dir, shape=shape,
                                     smoothing=0.1, denoising=0.1, with_hog_attached=True,
                                     with_dominant_color_attached=with_dominant_color_attached,
                                     pixels_per_cell=(shape[0] / 16, shape[0] / 16),
                                     cells_per_block=(16, 16),
                                     orientations=9, samples=number_of_samples)
    return X, y


def train(data_dir, test_dir, classifier_name, number_of_samples, shape, output_directory, beta,
          with_dominant_color_attached):
    start = time.time()
    if test_dir:
        X_train, y_train = get_data(data_dir, int(number_of_samples * 0.7), shape, with_dominant_color_attached)
        X_test, y_test = get_data(test_dir, int(number_of_samples * 0.3), shape, with_dominant_color_attached)
        X_train, y_train, = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)
    else:
        X, y = get_data(data_dir, number_of_samples, shape, with_dominant_color_attached)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    endtime = time.time() - start
    run.log('data_loading_time', np.float(endtime))

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
    rgb_or_dom = 'dom' if with_dominant_color_attached else 'rgb'
    helpers.plot_confusion_matrix_with_acc_and_fbeta(y_test, y_pred, classes=set(y_test), normalize=True,
                                                     save_path=f"{output_directory}/{classifier_name}-hog(ppc="
                                                               f"{(shape[0] / 16, shape[0] / 16)},"f"cpb={(16, 16)})"
                                                               f"-{rgb_or_dom}-{shape}-{number_of_samples}-cm.png",
                                                     fbeta_beta=beta)

    joblib.dump(value=classifier,
                filename=f'{output_directory}/{classifier_name}-hog(ppc={(shape[0] / 16, shape[0] / 16)},'
                         f'cpb={(16, 16)})'f'-{rgb_or_dom}-{shape}-{number_of_samples}.joblib')


def get_classifier(classifier: str):
    classifiers = {'svm': svm.SVC(C=100, gamma=0.001, kernel='rbf', probability=True),
                   'gaussian': GaussianNB(),
                   'multinomial': MultinomialNB(),
                   'bernoulli': BernoulliNB()}
    return classifiers.get(classifier)


def main():
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='directory of training data')
    parser.add_argument('--test_dir', type=str, help='directory of test data', default=None)
    parser.add_argument('--output_dir', type=str, help='output directory', default="./outputs")
    parser.add_argument('--classifier', type=str, default='multinomial')
    parser.add_argument('--number_of_samples', type=int, help='amount of training samples', default="400")
    parser.add_argument('--dominant_color', type=bool, help='with the dominantcolor attached', default=False)
    parser.add_argument('--shape', help='the width and height e.g. (128,128)', type=int, default='64')
    parser.add_argument('--fbeta_beta', type=float, default='0.5')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'scikit-learn version: {sklearn.__version__}')
    print("data directory is: " + args.data_dir)
    run.log('fbeta_beta', args.fbeta_beta)
    run.log('classifier', args.classifier)
    shape = (args.shape, args.shape)
    run.log('shape', args.shape)
    run.log('number_of_samples', args.number_of_samples)
    run.log('with_dominant_color', args.dominant_color)
    run.log('data_dir', args.data_dir)
    run.log('test_dir', args.test_dir)
    train(args.data_dir, args.test_dir, args.classifier, args.number_of_samples, shape=shape,
          output_directory=args.output_dir, beta=args.fbeta_beta, with_dominant_color_attached=args.dominant_color)


if __name__ == "__main__":
    main()
