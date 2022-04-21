import numpy as np

def add_contrast(image: np.ndarray):
    median = np.mean(image)
    image = np.where(image > median, 2 * image, image)
    return image