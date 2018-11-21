import glob
import time

import requests

"""
File created to evaluate the Custom Vision model. This way the average inference time can be measured. This
was important for the research.
"""

PREDICTION_ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction" \
                      "/71548120-925d-4e59-ba7e-32f99de50240/image?iterationId=a4a287bc-8a33-4a81-9c9d-068b06780616"


def make_prediction(image_path):
    with open(image_path, mode="rb") as image:
        r = requests.post(PREDICTION_ENDPOINT, data=image.read(),
                          headers={"Prediction-Key": "f2a1f096849242dfbe50583990b197fe",
                                   "Content-Type": "application/octet-stream"})


if __name__ == "__main__":
    total_time = 0
    count = 0
    for image_path in glob.glob(r"D:\LEGO Vision Datasets\Detection\Faster R-CNN"
                                r"\Natural Data_output 500 samples\testImages\*.jpg"):
        start = time.time()
        make_prediction(image_path)
        total_time += time.time() - start
        count += 1
    print("Average Inference Time: {:.2f} seconde per afbeelding".format(total_time//count))
