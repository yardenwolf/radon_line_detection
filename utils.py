import numpy as np
from skimage import io as sk_io


def add_contrast(image: np.ndarray):
    median = np.mean(image)
    image = np.where(image > median, 2 * image, image)
    return image


def normalize_matrix(matrix: np.ndarray):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val == min_val:
        min_val = 0
    return (((matrix - min_val)) / (max_val - min_val)) * 255


def save_as_tif(first_image, orig_images, path):
    output = np.concatenate([np.expand_dims(first_image, axis=0), orig_images], axis=0)
    sk_io.imsave(path, output, photometric='minisblack')
