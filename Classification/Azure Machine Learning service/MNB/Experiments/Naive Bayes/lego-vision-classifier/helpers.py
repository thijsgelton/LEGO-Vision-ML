import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
from PIL import Image
from skimage.transform import resize
from skimage.feature import hog
from skimage import data, exposure
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle


def pipeline(image, to_grayscale: bool = False, shape: tuple = None, smoothing: float = 0.0, denoising: float = 0.0, to_hog: bool = False,
             pixels_per_cell: tuple = (3,3), cells_per_block: tuple = (5,5), orientations: int = 9):
    preprocessed = []
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


def images_to_dataset(dataset_path, to_grayscale, shape, smoothing, denoising, to_hog, pixels_per_cell, cells_per_block, orientations, samples=300):
    X = []
    y = []

    for path, subfolders, files in os.walk(dataset_path):
        for subfolder in subfolders:
            for file in glob.glob(os.path.join(dataset_path, subfolder, "*.png"))[:int(samples)]:
                image = np.array(Image.open(os.path.join(dataset_path, subfolder, file))) 
                image = pipeline(image, to_grayscale, shape, smoothing, to_hog, denoising,
                                 pixels_per_cell, cells_per_block, orientations)        
                X.append(image)
                y.append(subfolder)
    return X,y
