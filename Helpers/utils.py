import itertools
import os
from shutil import copy
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import average_precision_score

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', 'pgm')


def rgb2gray(rgb_image):
    """
    Convert RGB image to grayscale

    Parameters:
        rgb_image : RGB image

    Returns:
        gray : grayscale image

    """
    return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.144])


def pyramid(image, downscale=1.5, min_size=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / downscale)
        image = resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, step_size, window_size):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


# Malisiewicz et al.
def non_max_suppression(boxes, overlap_thresh=0.7):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right labels-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, labels) coordinates for the start of
        # the bounding box and the smallest (x, labels) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def is_image_file(file_name):
    ext = file_name[file_name.rfind('.'):].lower()
    return ext in IMAGE_EXTENSIONS


def list_images(base_path, contains=None):
    # return the set of files that are valid
    return list_files(base_path, valid_exts=IMAGE_EXTENSIONS, contains=contains)


def list_files(base_path, valid_exts=IMAGE_EXTENSIONS, contains=None):
    # loop over the directory structure
    for (root_cir, dir_names, file_names) in os.walk(base_path):
        # loop over the file names in the current directory
        for file_name in file_names:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and file_name.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = file_name[file_name.rfind('.'):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(valid_exts):
                # construct the path to the image and yield it
                image_path = os.path.join(root_cir, file_name).replace(" ", "\\ ")
                yield image_path


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
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


def cannify_with_opencv(image, visualize=False):
    edges = cv2.Canny(image, 10, 50)
    if visualize:
        fig, ax = plt.subplots()
        ax.imshow(edges, cmap=plt.cm.gray)
        plt.show()
    return edges.flatten()


def calculate_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def intersection_over_union_removal(ground_truth_bounding_boxes, predicted_bounding_boxes, threshold=0.5):
    above_threshold = []
    for gt_bbox in ground_truth_bounding_boxes:
        for predicted_box in predicted_bounding_boxes:
            iou = calculate_intersection_over_union(gt_bbox, predicted_box)
            if iou > threshold:
                above_threshold.append(predicted_box)
        # above_threshold.extend(list(filter(lambda boxB: calculate_intersection_over_union(gt_bbox, boxB) > threshold,
        #                                    predicted_bounding_boxes)))
    return above_threshold


def resize_and_pad(img, width, height, pad_value=114):
    img_width = len(img[0])
    img_height = len(img)
    scale_w = img_width > img_height
    target_w = width
    target_h = height
    scale = None
    if scale_w:
        scale = float(width) / float(img_width)
        target_h = int(np.round(img_height * scale))
    else:
        scale = float(height) / float(img_height)
        target_w = int(np.round(img_width * scale))

    resized = cv2.resize(img, (target_w, target_h), 0, 0, interpolation=cv2.INTER_NEAREST)

    top = int(max(0, np.round((height - target_h) / 2)))
    left = int(max(0, np.round((width - target_w) / 2)))
    bottom = height - top - target_h
    right = width - left - target_w
    resized_with_pad = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])
    return resized_with_pad, scale, (left, top, right, bottom)


def display_bounding_boxes(image, bounding_boxes, labels=None, scores=None):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)
    for index, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle((x1, y1), abs(x2 - x1), abs(y2 - y1), fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        if labels and scores:
            ax.annotate("{} {:.2f}%".format(labels[index], scores[index] * 100), (x1, y1))
        elif labels:
            ax.annotate("{}".format(labels[index]), (x1, y1))
    plt.show()


def reshape_bbox(bbox, shape):
    x1, y1, w, h = x2_y2_to_w_h(bbox)
    w_ratio = shape[0] / (shape[0] + shape[1])
    h_ratio = shape[1] / (shape[0] + shape[1])
    new_width = (w + h) * w_ratio
    new_height = (w + h) * h_ratio
    return w_h_to_x2_y2((int(x1), int(y1), int(new_width), int(new_height)))


def w_h_to_x2_y2(box):
    box = np.array(box)
    x1, y1 = box[:2]
    x2, y2 = box[:2] + box[2:]
    return x1, y1, x2, y2


def x2_y2_to_w_h(box):
    box = np.array(box)
    x1, y1 = box[:2]
    w, h = box[2:] - box[:2]
    return x1, y1, w, h


def scale_bounding_box(box, scale=None, padding=None):
    assert isinstance(box, np.ndarray)
    if scale:
        box[np.array([0, 2])] = np.multiply(box[np.array([0, 2])], scale)
        box[np.array([1, 3])] = np.multiply(box[np.array([1, 3])], scale)
    left, top, right, bottom = padding
    box[0] = box[0] + left
    box[1] = box[1] + top
    box[2] = box[2] + right
    box[3] = box[3] + bottom
    return box


def average_precision_per_class(actual, predicted):
    average_precision_per_class = dict()
    for _class, actual_predicted_list in itertools.groupby(zip(actual, predicted), lambda x: x[0]):
        actual, predicted = sorted(actual_predicted_list, key=lambda x: x[1]['score'])
        average_precision_per_class[_class] = average_precision_score(actual, [pred['label'] for pred in predicted])
    return average_precision_per_class


def write_ground_truth(image_path, gt_boxes, gt_labels, base_directory, with_animation=False):
    image_name = image_path.split("\\")[-1]
    os.makedirs(os.path.join(base_directory, "ground-truth"), exist_ok=True)
    if with_animation:
        os.makedirs(os.path.join(base_directory, "images"), exist_ok=True)
        copy(image_path, os.path.join(base_directory, "images", image_name))
    with open(os.path.join(base_directory, "ground-truth", image_name.replace("jpg", "txt")), mode="w") as gt:
        for gt_label, gt_box in zip(gt_labels, gt_boxes):
            x1, y1, x2, y2 = gt_box
            gt.write("{} {} {} {} {}\n".format(gt_label, int(x1), int(y1), int(x2), int(y2)))


def write_prediction(image_path, bboxes, scores, labels, base_directory):
    image_name = image_path.split("\\")[-1]
    os.makedirs(os.path.join(base_directory, "predicted"), exist_ok=True)
    with open(os.path.join(base_directory, "predicted", image_name.replace("jpg", "txt")), mode="w") as gt:
        for pred_label, confidence, pred_bbox in zip(labels, scores, bboxes):
            x1, y1, x2, y2 = pred_bbox
            gt.write("{} {} {} {} {} {}\n".format(pred_label, confidence, int(x1), int(y1), int(x2), int(y2)))
