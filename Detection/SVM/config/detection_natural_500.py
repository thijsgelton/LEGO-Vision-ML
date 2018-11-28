import glob
import os

base_directory = r"D:\LEGO Vision Datasets\Detection\Faster R-CNN\Natural Data_output 500 samples"
image_paths = glob.glob(os.path.join(base_directory, "testImages", "No cut off bricks", "*.jpg"))
gt_bbox_paths = glob.glob(os.path.join(base_directory, "testImages", "No cut off bricks", "*.bboxes.tsv"))
gt_labels_paths = glob.glob(os.path.join(base_directory, "testImages", "No cut off bricks", "*.bboxes.labels.tsv"))
classes = os.path.join(base_directory, "class_map.txt")
output_directory = os.path.join("D:\LEGO Vision Datasets\Detection", "SVM", "results", "500 samples")
classifier = r"D:\Projects\LEGO Vision\Classification\Final models\SVM\SVM-3200-256W-HNM-MC.joblib"
label_lookup = sorted([line.split('\t')[0] for line in open(classes).readlines()[1:]])