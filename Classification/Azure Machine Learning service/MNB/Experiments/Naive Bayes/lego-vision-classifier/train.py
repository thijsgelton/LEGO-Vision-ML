
import argparse
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import classification_report,accuracy_score, fbeta_score

from sklearn.model_selection import train_test_split
import numpy as np

from azureml.core import Run
import helpers

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

classifiers = {'gaussian': GaussianNB(), 'bernoulli': BernoulliNB(), 'multinomial': MultinomialNB()}


# let user feed in 2 parameters, the location of the data files (from datastore), and the regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument('--classifier', type=str, dest='classifier', help='What Naive Bayes classifier to use. gaussian, bernoulli, multinomial')
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()

data_folder = os.path.join(args.data_folder, 'lego-vision-classification')
print('Data folder:', data_folder)

X, y = helpers.images_to_dataset(dataset_path=data_folder, to_grayscale=False, shape=(128,128), 
                                 smoothing=0.1, to_hog=True, pixels_per_cell=(8,8), cells_per_block=(8,8), orientations=9,
                                 denoising=0.2, samples=600)

X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# get hold of the current run
run = Run.get_submitted_run()

print('Train a naive bayes model, classifier: ', args.classifier)
clf = classifiers.get(args.classifier)
clf.fit(X_train, y_train)

print('Predict the test set')
y_hat = clf.predict(X_test)

# calculate accuracy on the prediction
acc = accuracy_score(y_test, y_hat)
print("F-Beta with fbeta_score(y_test, y_pred, beta=0.6, 'macro')")
fbeta = fbeta_score(y_test, y_hat, beta=0.6, average='macro')

print(f"Accuracy: {acc}")
print(classification_report(y_test, y_hat))

run.log('classifier', args.classifier)
run.log('accuracy', np.float(acc))
run.log('fbeta-score', np.float(fbeta))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=clf, filename='outputs/nb-classifier.pkl')