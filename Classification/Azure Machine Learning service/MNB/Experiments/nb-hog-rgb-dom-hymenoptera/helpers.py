import glob
import itertools as it
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def images_to_dataset(dataset_path, shape, smoothing, denoising, with_hog_attached, with_dominant_color_attached,
                      pixels_per_cell, cells_per_block, orientations, samples=300, converter=None, debug=False):
    X = []
    y = []
    for path, subfolders, files in os.walk(dataset_path):
        for subfolder in subfolders:
            for file in shuffle(glob.glob(os.path.join(dataset_path, subfolder, '*')))[:int(samples)]:
                if file.endswith(('png', 'jpg')):
                    image = np.array(Image.open(os.path.join(file)))
                    image = pipeline(image, shape=shape, smoothing=smoothing, denoising=denoising,
                                     with_hog_attached=with_hog_attached,
                                     with_dominant_color_attached=with_dominant_color_attached,
                                     pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                                     orientations=orientations, converter=converter, debug=debug)
                    X.append(image)
                    y.append(subfolder)
    return X, y


def pipeline(image, shape: tuple = None, smoothing: float = 0.0, denoising: float = 0.0,
             with_hog_attached: bool = False, with_dominant_color_attached: bool = False,
             pixels_per_cell: tuple = (3, 3), cells_per_block: tuple = (5, 5), orientations: int = 9, converter=None,
             debug=False):
    if shape and len(shape) > 1:
        image = resize(image, shape, anti_aliasing=True, mode='reflect')
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    if smoothing:
        image = gaussian(image, sigma=smoothing, multichannel=False)
    if denoising:
        image = denoise_tv_chambolle(image, weight=denoising, multichannel=True)
    if callable(converter):
        image = converter(image)
    if with_dominant_color_attached:
        fd, show = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, visualize=True, multichannel=True, block_norm="L2-Hys")
        dom_color = dominant_color(image)
        if debug:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

            ax1.axis('off')
            ax1.imshow(show, cmap=plt.cm.gray)
            ax1.set_title('HOG')

            color_array = np.ndarray(shape=(shape[0], shape[1], 3))
            print(dom_color)
            color_array[:, :, :] = dom_color

            ax2.axis('off')
            ax2.imshow(color_array)
            ax2.set_title('Dominant color')
            plt.show()
        image = np.concatenate((dom_color, fd), axis=None)
    elif with_hog_attached:
        fd, show = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, visualize=True, multichannel=True, block_norm="L2-Hys")
        if debug:
            plt.imshow(show)
        image = np.concatenate((image.flatten(), fd), axis=None)
    image = image.flatten()
    return image


def grayscale(image):
    return [rgb2gray(p) for p in image]


def dominant_color(image):
    counter = Counter()
    kmeans = quantize(np.asarray(image), 3)
    onedar = []
    for row in kmeans:
        for rgb in row:
            counter[tuple(rgb)] += 1

    return np.asarray(counter.most_common()[-1][0])


def quantize(raster, n_colors):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))

    model = KMeans(n_clusters=n_colors)
    labels = model.fit_predict(reshaped_raster)
    palette = model.cluster_centers_

    quantized_raster = np.reshape(palette[labels], (width, height, palette.shape[1]))

    return quantized_raster


import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, fbeta_score


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
