import glob
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, accuracy_score, fbeta_score


def pipeline(image, to_grayscale: bool = False, shape: tuple = None, smoothing: float = 0.0, denoising: float = 0.0,
             to_hog: bool = False,
             pixels_per_cell: tuple = (3, 3), cells_per_block: tuple = (5, 5), orientations: int = 9):
    if shape and len(shape) > 1:
        image = resize(image, shape, anti_aliasing=True)
    if smoothing:
        image = gaussian(image, sigma=smoothing)
    if denoising:
        image = denoise_tv_chambolle(image, weight=denoising, multichannel=True)
    if to_grayscale:
        image = grayscale(image)
    if to_hog:
        image, show = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, visualize=True, multichannel=True, block_norm="L2-Hys")
        plt.imshow(show)
    else:
        image = image.flatten()
    return image


def grayscale(image):
    return [rgb2gray(p) for p in image]


def images_to_dataset(dataset_path, to_grayscale, shape, smoothing, denoising, to_hog, pixels_per_cell, cells_per_block,
                      orientations, samples=300):
    X = []
    y = []

    for path, subfolders, files in os.walk(dataset_path):
        for subfolder in subfolders:
            for file in glob.glob(os.path.join(dataset_path, subfolder, "*.png"))[:int(samples)]:
                image = np.array(Image.open(os.path.abspath(file)))
                image = pipeline(image, to_grayscale, shape, smoothing, to_hog, denoising,
                                 pixels_per_cell, cells_per_block, orientations)
                X.append(image)
                y.append(subfolder)
    return X, y


def plot_confusion_matrix_with_acc_and_fbeta(y_true, y_pred, classes,
                                             normalize=False,
                                             title='Confusion matrix',
                                             cmap=plt.cm.Blues,
                                             save_path=None,
                                             fbeta_beta=0.5):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)

    fbeta = fbeta_score(y_true, y_pred, average='macro', beta=fbeta_beta)

    accuracy = accuracy_score(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.figtext(0.2, -0.05, f'Accuracy: {accuracy:0.2f}', horizontalalignment='left', fontsize=12)
    plt.figtext(0.2, -0.10, f'F-Score (beta = {fbeta_beta:0.1f}): {fbeta:0.2f}', horizontalalignment='left',
                fontsize=12)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
