from skimage import transform
import cv2
import numpy as np
from typing import List, Tuple, Callable
import torch
from skimage.feature.peak import peak_local_max
from nd2reader import ND2Reader
import matplotlib.pyplot as plt


def radon_exmaple():
    image = load_image("C:/Users/yarde/Documents/sample-data/2022-02-10/1824.nd2", process_or_percentile)
    # blured_image = cv2.medianBlur(image, 5)
    blured_image = image
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    res = transform.radon(image=blured_image, theta=theta)
    kernel = np.ones((3, 7), np.uint8)
    peaks = peak_local_max(res, min_distance=5, num_peaks=30, footprint=kernel)
    peaks = filter_peaks(peaks)
    zero_image = np.zeros(res.shape)
    for i, j in peaks:
        zero_image[i, j] = res[i, j]
    reconstruction = transform.iradon(zero_image, theta=theta)
    reconstruction = normalize_matrix(reconstruction)
    image[reconstruction > 0.4] = 20000
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(reconstruction, cmap=plt.get_cmap('gray'))
    axarr[1].imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()
    # plt.savefig('example.pdf', dpi=1200)
    print("123")


def filter_peaks(peaks: np.ndarray):
    """
    filtering peaks based on interval around the median of theta from the radon results
    :param peaks:
    :return:
    """
    thetas = peaks[0:, 1]
    median = np.median(thetas)
    q = 0.05  # quantile distance from median
    a = median * q
    elements = np.logical_and((median - a) < thetas, thetas < (median + a))
    filtered_peaks = peaks[elements]
    return filtered_peaks


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)


def find_max_n_elements(image: np.ndarray, n: int) -> List[Tuple]:
    new_image = np.copy(image)
    max_list = []
    for i in range(n):
        index = np.unravel_index(np.argmax(new_image), new_image.shape)
        max_list.append(index)
        new_image[index] = 0
    return max_list


def find_and_pool(image: np.ndarray, n: int) -> List[Tuple]:
    matrix = torch.tensor(image)
    matrix = torch.unsqueeze(matrix, 0)
    max_pool = torch.nn.MaxPool2d(kernel_size=(9, 9), return_indices=True)
    vals, indices = max_pool(matrix)
    # check = vals.squeeze(dim=0)
    # new_im = check.numpy()
    indices = torch.flatten(indices)
    j_indices = (indices % image.shape[1])
    i_indices = torch.floor(indices / image.shape[1]).to(torch.int)
    max_vals_list = list(
        map(lambda val, i, j: (val, i, j), vals.flatten().numpy(), i_indices.numpy(), j_indices.numpy()))
    max_vals_list.sort(key=lambda x: x[0], reverse=True)
    return max_vals_list[:n]


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


def test_images(images: ND2Reader.ndim):
    mean_thresh_func = np.mean
    percentile_thresh_func = lambda x: np.percentile(x, 30)
    or_agg = cv2.bitwise_or
    sum_agg = lambda x, y: x + y
    sample_1 = process_func(images, mean_thresh_func, or_agg)
    sample_2 = process_func(images, mean_thresh_func, sum_agg)
    sample_3 = process_func(images, percentile_thresh_func, or_agg)
    sample_4 = process_func(images, percentile_thresh_func, sum_agg)
    f, axarr = plt.subplots(2, 2)
    plt.figure(dpi=1200)
    axarr[0, 0].imshow(sample_1, cmap=plt.get_cmap('gray'))
    axarr[0, 1].imshow(sample_2, cmap=plt.get_cmap('gray'))
    axarr[1, 0].imshow(sample_3, cmap=plt.get_cmap('gray'))
    axarr[1, 1].imshow(sample_4, cmap=plt.get_cmap('gray'))
    plt.show()
    return "1234"


def load_image(dir: str, processing_func: Callable = None):
    data = None
    with ND2Reader(dir) as images:
        if processing_func:
            data = processing_func(images)
        else:
            data = images[0]
    return data


if __name__ == "__main__":
    radon_exmaple()
    # load_image()
