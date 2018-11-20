import cv2
import glob
import os
import sys
import time
import numpy as np
import re

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# Replace with a valid key
from msrest.exceptions import HttpOperationError

from Helpers import utils

SUBSCRIPTION_KEY_ENV_NAME = "04bcd5b3634c4ca5ada3cd1d57ab8d76"
PREDICTION_KEY_ENV_NAME = "f2a1f096849242dfbe50583990b197fe"

ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com"

# Add this directory to the path so that custom_vision_training_samples can be found
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))

BASE_DIRECTORY = r"D:\LEGO Vision Datasets\Detection\Faster R-CNN\Natural Data_output 500 samples"
IMAGES_FOLDER = os.path.join(BASE_DIRECTORY, "positive")
ONLY_NUMBERS = re.compile(r"\d+.\d+")


def run_sample(subscription_key):
    try:
        prediction_key = os.environ[PREDICTION_KEY_ENV_NAME]
    except KeyError:
        raise KeyError("You need to set the {} env variable.".format(PREDICTION_KEY_ENV_NAME))

    project, iteration = train_project(subscription_key)
    predict_project(prediction_key, project, iteration)


def read_bboxes(gt_bbox_path, scale, padding):
    bbox_lines = open(gt_bbox_path).readlines()
    gt_bounding_boxes = []
    for bbox_line in bbox_lines:
        bbox = np.array(list(map(lambda string: float(string), re.findall(ONLY_NUMBERS, bbox_line))))
        gt_bounding_boxes.append(utils.scale_bounding_box(bbox, scale, padding))
    return np.array(gt_bounding_boxes)


def read_labels(labels_path):
    labels = list(map(lambda line: line.strip('\n'), open(labels_path).readlines()))
    return labels


# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def train_project(training_key):
    trainer = CustomVisionTrainingClient(training_key, endpoint=ENDPOINT)

    # Find the object detection domain
    obj_detection_domain = next(domain for domain in trainer.get_domains() if domain.type == "ObjectDetection")

    print("Creating project...")
    try:
        project = trainer.create_project("LEGO Vision", domain_id=obj_detection_domain.id)
    except HttpOperationError:
        print("Project already exists. Using this one.")
        project = trainer.get_project(project_id="71548120-925d-4e59-ba7e-32f99de50240")

    classes = os.path.join(BASE_DIRECTORY, "class_map.txt")
    tags = dict()
    # Make two tags in the new project
    for _class in list(map(lambda line: line.split('\t')[0], open(classes).readlines())):
        try:
            tags[_class] = trainer.create_tag(project.id, _class)
        except HttpOperationError:
            print("Tag already created, continuing...")
            for tag in trainer.get_tags(project_id="71548120-925d-4e59-ba7e-32f99de50240"):
                tags[tag.name] = tag

        # Go through the data table above and create the images
    print("Adding images...")
    tagged_images_with_regions = []

    for image_path in glob.glob(os.path.join(IMAGES_FOLDER, "*.jpg")):
        file_id, extension = image_path.split(".", 1)
        # image, scale, padding = utils.resize_and_pad(img=cv2.cvtColor(cv2.imread(image_path),
        #                                                               cv2.COLOR_BGR2RGB),
        #                                              width=1024,
        #                                              height=1024)
        # utils.display_bounding_boxes(image, bounding_boxes=[])
        # succes, encoded_image = cv2.imencode('.'+extension, image)
        bboxes = read_bboxes(os.path.join(IMAGES_FOLDER, file_id + ".bboxes.tsv"), scale=1, padding=(0, 0, 0, 0))
        labels = read_labels(os.path.join(IMAGES_FOLDER, file_id + ".bboxes.labels.tsv"))
        regions = [Region(tag_id=tags[_class].id, left=bbox[0], top=bbox[1],
                          width=abs(bbox[0] - bbox[2]), height=abs(bbox[1] - bbox[3])) for _class, bbox in zip(labels,
                                                                                                               bboxes)]
        with open(image_path, mode="rb") as image_contents:
            tagged_images_with_regions.append(ImageFileCreateEntry(name=file_id, contents=image_contents.read(),
                                                                   regions=regions))

    for batch in chunks(tagged_images_with_regions, 64):
        trainer.create_images_from_files(project.id, images=batch, tag_ids=[tag.id for tag in tags.values()])

    print("Training...")
    iteration = trainer.train_project(project.id)
    while iteration.status != "Completed" or iteration.status != "Failed":
        iteration = trainer.get_iteration(project.id, iteration.id)
        print("Training status: " + iteration.status)
        time.sleep(1)

    # The iteration is now trained. Make it the default project endpoint
    trainer.update_iteration(project.id, iteration.id, is_default=True)
    print("Done!")
    return project, iteration


def predict_project(prediction_key, project, iteration):
    predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

    # Open the sample image and get back the prediction results.
    with open(os.path.join(IMAGES_FOLDER, "Test", "test_od_image.jpg"), mode="rb") as test_data:
        results = predictor.predict_image(project.id, test_data, iteration.id)

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100),
              prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width,
              prediction.bounding_box.height)


if __name__ == "__main__":
    train_project(SUBSCRIPTION_KEY_ENV_NAME)
