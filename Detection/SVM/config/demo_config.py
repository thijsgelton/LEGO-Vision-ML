import glob
import os

base_directory = r"D:\LEGO Vision Datasets\Detection\Faster R-CNN\Natural Data_output 500 samples"
image_paths = glob.glob(os.path.join(r"C:\Users\Thijs\OneDrive\LEGO Vision\Demo", "*.jpg"))
gt_bbox_paths = []
gt_labels_paths = []
classes = os.path.join(base_directory, "class_map.txt")
classifier = r"D:\Projects\LEGO Vision\Classification\Final models\SVM\SVM-3200-256W-HNM.joblib"
label_lookup = sorted([line.split('\t')[0] for line in open(classes).readlines()[1:]])
