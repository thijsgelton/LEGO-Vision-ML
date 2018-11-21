import glob
import os

base_directory = r"D:\LEGO Vision Datasets\Localization\Faster R-CNN\Natural Data"
image_paths = glob.glob(os.path.join(base_directory, "testImages", "*.jpg"))[29:30]
gt_bbox_paths = glob.glob(os.path.join(base_directory, "testImages", "*.bboxes.tsv"))[29:30]
gt_labels_paths = glob.glob(os.path.join(base_directory, "testImages", "*.bboxes.labels.tsv"))[29:30]
classes = os.path.join(base_directory, "class_map.txt")
output_directory = r"D:\LEGO Vision Datasets\Localization\SVM"
classifier = r"D:\Projects\LEGO Vision\Classification\Final models\SVM\SVM-3200-256W.joblib"
label_lookup = sorted([line.split('\t')[0] for line in open(classes).readlines()[1:]])