from typing import List, Tuple

from skimage import transform
import cv2
import numpy as np
import torch
from skimage.feature.peak import peak_local_max
from nd2reader import ND2Reader
import matplotlib.pyplot as plt

from utils import add_contrast
from image_loader import load_image, process_or_percentile, process_func

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale


def radon_exmaple():
    # image = load_image("C:/Users/yarde/Documents/sample-data/2022-02-10/1824.nd2", process_or_percentile)
    image = np.zeros((512, 512))
    diagonal = [i for i in range(0, 512)]
    image[diagonal, diagonal] = 256
    # blured_image = cv2.medianBlur(image, 5)
    blured_image = image
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    res = transform.radon(image=blured_image, theta=theta)
    kernel = np.ones((3, 7), np.uint8)
    peaks = peak_local_max(res, min_distance=5, num_peaks=30, footprint=kernel)
    peaks = filter_peaks(peaks)
    zero_image = np.zeros(res.shape)
    # for i, j in peaks:
    #    zero_image[i, j] = res[i, j]
    i, j = peaks[0]
    zero_image[i, j] = res[i, j]
    reconstruction = transform.iradon(zero_image, theta=theta)
    reconstruction = normalize_matrix(reconstruction)
    image = add_contrast(image)  # just for displaying the dots better
    image[reconstruction > 0.4] = 20000
    # f, axarr = plt.subplots(1, 2, constrained_layout=True)
    # axarr[0].imshow(reconstruction, cmap=plt.get_cmap('gray'))
    # axarr[0].set_title("lines")
    # axarr[1].imshow(image, cmap=plt.get_cmap('gray'))
    # axarr[1].set_title("lines on dots")
    # plt.show()
    # plt.savefig('example.pdf', dpi=1200)
    # plt.savefig('./pictures/after_peak_filter.png', dpi=1200)
    # print("123")
    return peaks


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
    return ((matrix - min_val)) / (max_val - min_val)


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


def test_images(images: ND2Reader.ndim):
    mean_thresh_func = np.mean
    percentile_thresh_func = lambda x: np.percentile(x, 30)
    or_agg = cv2.bitwise_or
    sum_agg = lambda x, y: x + y
    sample_1 = process_func(images, mean_thresh_func, or_agg)
    sample_1 = add_contrast(sample_1)
    sample_2 = process_func(images, mean_thresh_func, sum_agg)
    sample_3 = process_func(images, percentile_thresh_func, or_agg)
    sample_3 = add_contrast(sample_3)
    sample_4 = process_func(images, percentile_thresh_func, sum_agg)
    fig, axarr = plt.subplots(2, 2, constrained_layout=True)
    plt.figure(dpi=1200)
    axarr[0, 0].imshow(sample_1, cmap=plt.get_cmap('gray'))
    axarr[0, 0].set_title("mean and or")

    axarr[0, 1].imshow(sample_2, cmap=plt.get_cmap('gray'))
    axarr[0, 1].set_title("mean and sum")

    axarr[1, 0].imshow(sample_3, cmap=plt.get_cmap('gray'))
    axarr[1, 0].set_title("percentile and or")

    axarr[1, 1].imshow(sample_4, cmap=plt.get_cmap('gray'))
    axarr[1, 1].set_title("percentile and sum", pad=2)
    fig.savefig('./pictures/thresh_agg_1.png', dpi=1200)
    return "1234"


def extract_r_theta_from_peaks(peaks: list, sinogram_size: Tuple):
    exmp_list = []
    for i, j in peaks:
        # i is the distance of the intersection of the normal with the line
        # j is the angle according to the line space
        theta = 180 * (j / sinogram_size[1])
        r = i / np.cos(np.deg2rad(theta))
        exmp_list.append((theta, r))
    return exmp_list


if __name__ == "__main__":
    res = radon_exmaple()
    example = extract_r_theta_from_peaks(res, (512, 512))
