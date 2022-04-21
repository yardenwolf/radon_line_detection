from typing import Callable

import cv2
import numpy as np
from nd2reader import ND2Reader


def load_image(dir: str, processing_func: Callable = None):
    data = None
    with ND2Reader(dir) as images:
        if processing_func:
            data = processing_func(images)
        else:
            data = images[0]
    return data


def process_func(images: ND2Reader.ndim, threshold_func: Callable, agg_func: Callable):
    agg_image = np.ndarray(shape=images[0].shape, dtype=images[0].dtype)
    for image in images:
        threshold = threshold_func(image)
        image_copy = image.copy()
        image_copy[image_copy < threshold] = 0
        agg_image = agg_func(agg_image, image_copy)
    return agg_image


def process_or_percentile(images):
    percentile_thresh_func = lambda x: np.percentile(x, 30)
    or_agg = cv2.bitwise_or
    return process_func(images, percentile_thresh_func, or_agg)


def load_image_for_labeling(dir: str, processing_func: Callable = None):
    all_images = []
    with ND2Reader(dir) as images:
        if processing_func:
            all_images.append(np.expand_dims(processing_func(images), 2))
        for image in images:
            all_images.append(np.expand_dims(image, 2))
    all_images = np.stack(all_images, axis=0)
    return all_images
