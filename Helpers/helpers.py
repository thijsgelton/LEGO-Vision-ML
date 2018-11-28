import glob
import os
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, fbeta_score
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from pymongo.collection import Collection


def get_data(data_source, number_of_samples, shape, color_insensitive):
    if isinstance(data_source, str):
        X, y = from_disk(color_insensitive, data_source, number_of_samples, shape)
    elif isinstance(data_source, Collection):
        X, y = from_database(color_insensitive, data_source, number_of_samples, shape)
    else:
        raise ValueError("data_source must be of type(str) or str(pymongo.collection.Collection), but is of type:"
                         + str(type(data_source)))
    return X, y


def from_database(color_insensitive, collection, number_of_samples, shape):
    X = []
    y = []
    number_of_samples_per_class = Counter()
    images = collection.find({'shape': shape, 'color': not bool(color_insensitive)})
    for image in images:
        if number_of_samples_per_class[image['label']] >= number_of_samples:
            continue
        X.append(np.array(image['features']))
        y.append(image['label'])
        number_of_samples_per_class.update({image['label']: 1})
    X = np.array(X)
    return X, y


def from_disk(color_insensitive, data_dir, number_of_samples, shape):
    if color_insensitive:
        converter = lambda x: 1 - rgb2gray(x)  # inverting the image makes it process faster
    else:
        converter = None
    X, y = images_to_dataset(dataset_path=data_dir,
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
                             debug=True)
    X = preprocessing.scale(X, with_mean=False)
    return X, y


def images_to_dataset(dataset_path, shape, smoothing, denoising, with_hog_attached, with_mean_color_attached,
                      pixels_per_cell, cells_per_block, orientations, samples=300, converter=None, debug=False,
                      with_dominant_color_attached=False):
    X = []
    y = []
    for subfolder in list(os.walk(dataset_path))[0][1]:
        for i, file in enumerate(shuffle(glob.glob(os.path.join(dataset_path, subfolder, '*')))[:int(samples)]):
            if i > 0 and int(i) % 100 == 0:
                print(subfolder, i)
            if file.endswith(('png', 'jpg')):
                image = np.array(Image.open(os.path.join(file)))
                image = pipeline(image, shape=shape, smoothing=smoothing, denoising=denoising,
                                 with_hog_attached=with_hog_attached,
                                 with_mean_color_attached=with_mean_color_attached,
                                 with_dominant_color_attached=with_dominant_color_attached,
                                 pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                                 orientations=orientations, converter=converter, debug=debug)
                X.append(image)
                y.append(subfolder)
    return X, y


def pipeline(image, shape: tuple = None, smoothing: float = 0.0, denoising: float = 0.0,
             with_hog_attached: bool = False, with_dominant_color_attached: bool = False,
             with_mean_color_attached: bool = False,
             pixels_per_cell: tuple = (3, 3), cells_per_block: tuple = (5, 5), orientations: int = 9, converter=None,
             debug=False):
    if shape and len(shape) > 1:
        image = resize(image, shape, anti_aliasing=True, mode='reflect')
    original = image
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    if smoothing:
        image = gaussian(image, sigma=smoothing, multichannel=True)
    if denoising:
        image = denoise_tv_chambolle(image, weight=denoising, multichannel=True)
    converted = None
    if callable(converter):
        image = converter(image)
        converted = image
    if with_dominant_color_attached or with_mean_color_attached:
        multichannel = True if len(image.shape) == 3 and image.shape[-1] > 1 else False
        fd, show = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, visualize=True,
                       multichannel=multichannel,
                       block_norm="L2-Hys")
        if with_mean_color_attached:
            mean = mean_color(image)
            if debug:
                plot_debug(converted, converter, mean, original, shape, show)
            image = np.concatenate((mean, fd), axis=None)
        elif with_dominant_color_attached:
            dom_color = dominant_color(image, k=2)
            if debug:
                plot_debug(converted, converter, dom_color, original, shape, show)
            image = np.concatenate((dom_color, fd), axis=None)
    elif with_hog_attached:
        fd, show = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, visualize=True, multichannel=True, block_norm="L2-Hys")
        if debug:
            plt.imshow(show)
        image = np.concatenate((image.flatten(), fd), axis=None)
    image = image.flatten()
    return image


def mean_color(image):
    return np.mean(image, axis=(0, 1))


def plot_debug(converted, converter, dom_color, original, shape, show):
    fig, axs = plt.subplots(1, 4 if converter else 3, figsize=(8, 4), sharex=True, sharey=True)
    axs[0].axis('off')
    axs[0].imshow(show, cmap=plt.cm.gray)
    axs[0].set_title('HOG')
    color_array = np.ndarray(shape=(shape[0], shape[1], 3))
    color_array[:, :, :] = dom_color
    axs[1].axis('off')
    axs[1].imshow(color_array)
    axs[1].set_title('Dominant color')
    axs[2].axis('off')
    axs[2].imshow(original)
    axs[2].set_title('Original')
    if converter:
        axs[3].axis('off')
        axs[3].imshow(converted)
        axs[3].set_title('Converted')
    plt.show()


def grayscale(image):
    return [rgb2gray(p) for p in image]


def dominant_color(image, k=3, nth_color=-1):
    counter = Counter()
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    kmeans = quantize(np.asarray(image), k)
    for row in kmeans:
        for rgb in row:
            counter[tuple(rgb)] += 1

    return np.asarray(counter.most_common()[nth_color][0])


def quantize(raster, n_colors):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))

    model = KMeans(n_clusters=n_colors)
    labels = model.fit_predict(reshaped_raster)
    palette = model.cluster_centers_

    quantized_raster = np.reshape(palette[labels], (width, height, palette.shape[1]))

    return quantized_raster


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

    plt.figtext(0.2, -0.05, 'Accuracy: {:0.2f}'.format(accuracy), horizontalalignment='left', fontsize=12)
    plt.figtext(0.2, -0.10, 'F-Score (beta = {:0.1f}): {:0.2f}'.format(fbeta_beta, fbeta), horizontalalignment='left',
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


def gen_sift_features(gray_img):
    orb = cv2.ORB_create(nfeatures=21)
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = orb.detectAndCompute(gray_img, None)
    return kp, desc


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def hog_descriptor_opencv(image, shape):
    win_size = shape
    block_size = (int(shape[0] / 8), int(shape[0] / 8))
    block_stride = (int(shape[0] / 8), int(shape[0] / 8))
    cell_size = (int(shape[0] / 8), int(shape[0] / 8))
    orientation_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, orientation_bins)
    return hog.compute(image)


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
