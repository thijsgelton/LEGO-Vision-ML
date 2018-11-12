import argparse
import os
import numpy as np
import time
import sklearn
from skimage.color import rgb2gray
from sklearn import svm
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.externals import joblib
from sklearn.utils import shuffle
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
                                     denoising=0.1,
                                     with_hog_attached=True,
                                     with_dominant_color_attached=True,
                                     pixels_per_cell=(shape[0] / 16, shape[0] / 16),
                                     cells_per_block=(16, 16),
                                     orientations=9,
                                     samples=number_of_samples,
                                     converter=converter)
    return X, y


def train(data_dir, test_dir, classifier_name, number_of_samples, shape, output_directory, beta, color_insensitive,
          is_local, run):
    start = time.time()
    if test_dir:
        X_train, y_train = get_data(data_dir, int(number_of_samples * 0.7), shape, color_insensitive)
        X_test, y_test = get_data(test_dir, int(number_of_samples * 0.3), shape, color_insensitive)
        X_train, y_train, = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)
    else:
        X, y = get_data(data_dir, number_of_samples, shape, color_insensitive)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    endtime = time.time() - start

    classifier = get_classifier(classifier_name)

    start = time.time()
    classifier.fit(X_train, y_train)
    endtime = time.time() - start

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    fscore = fbeta_score(y_test, y_pred, beta=beta, average='macro')
    training_data = data_dir.split("\\")[-1]
    output_dir = f'{output_directory}/{training_data}'
    color_insensitive = 'color_insensitive' if color_insensitive else 'color_sensitive'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    helpers.plot_confusion_matrix_with_acc_and_fbeta(y_test, y_pred, classes=set(y_test), normalize=True,
                                                     save_path=f"{output_dir}/{classifier_name}-hog(ppc="
                                                               f"{(shape[0] / 16, shape[0] / 16)},"f"cpb={(16, 16)})"
                                                               f"-{color_insensitive}-{shape}-{number_of_samples}-cm.png",
                                                     fbeta_beta=beta)
    if not is_local and run:
        run.log('data_loading_time', np.float(endtime))
        run.log('training_time', np.float(endtime))
        run.log('accuracy', np.float(accuracy))
        run.log('f_score', np.float(fscore))

    joblib.dump(value=classifier,
                filename=f'{output_dir}/{classifier_name}-hog(ppc={(shape[0] / 16, shape[0] / 16)},'
                         f'cpb={(16, 16)})-{color_insensitive}-{shape}-{number_of_samples}.joblib')


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
    parser.add_argument('--classifier', type=str, default='svm')
    parser.add_argument('--number_of_samples', type=int, help='Amount of training samples', default="400")
    parser.add_argument('--color_insensitive', type=int, help='True if color may not be a feature', default=0)
    parser.add_argument('--shape', help='the width and height e.g. (128,128)', type=int, default='64')
    parser.add_argument('--fbeta_beta', type=float, default='0.5')
    parser.add_argument('--is_local', type=bool, default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'scikit-learn version: {sklearn.__version__}')
    print("data directory is: " + args.data_dir)
    shape = (args.shape, args.shape)
    run = None
    if not args.is_local:
        run = Run.get_submitted_run()
        run.log('fbeta_beta', args.fbeta_beta)
        run.log('classifier', args.classifier)
        run.log('shape', args.shape)
        run.log('number_of_samples', args.number_of_samples)
        run.log('shape_and_samples', int(str(args.number_of_samples) + str(args.shape)))
        run.tag('color_insensitive', str(args.color_insensitive))
        run.log('color_insensitive', args.color_insensitive)
        run.log('data_dir', args.data_dir)
        run.log('test_dir', args.test_dir)
    train(args.data_dir, args.test_dir, args.classifier, args.number_of_samples, shape=shape,
          output_directory=args.output_dir, beta=args.fbeta_beta, color_insensitive=args.color_insensitive,
          is_local=args.is_local, run=run)


if __name__ == "__main__":
    main()
