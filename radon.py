from typing import List, Tuple

from collections import namedtuple
from skimage import transform
import cv2
import numpy as np
import torch
from skimage.feature.peak import peak_local_max
from nd2reader import ND2Reader
import matplotlib.pyplot as plt

from labeler import Line
from utils import add_contrast
from image_loader import load_image, process_or_percentile, process_func, process_or_mean

from skimage.transform import radon, rescale


def radon_exmaple():
    # image = load_image("C:/Users/yarde/Documents/sample-data/2022-02-10/1824.nd2", process_or_mean)
    image = np.zeros((512, 512))
    # blured_image = cv2.medianBlur(image, 5)
    # cv2.line(image, (0,216), (512,512), color=(255,0,0))
    #cv2.line(image, (0, 0), (512, 512), color=(255, 0, 0))
    cv2.line(image, (0, 0), (296, 512), color=(255, 0, 0))
    blured_image = image
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    res = transform.radon(image=blured_image, theta=theta)
    kernel = np.ones((3, 7), np.uint8)
    #peaks = peak_local_max(res, min_distance=2, num_peaks=20, footprint=kernel)
    #peaks = filter_peaks(peaks)
    peak = np.unravel_index(np.argmax(res, axis=None), res.shape)
    peaks = [peak]
    #unfiltered_new_image = np.zeros(image.shape)
    #filtered_new_image = np.zeros(image.shape)
    #image_3 = cv2.cvtColor(normalize_matrix(blured_image), cv2.COLOR_GRAY2RGB)
    lines = find_lines(peaks=peaks, image_shape=blured_image.shape, image=blured_image)
    for line in lines:
        prof = get_line_profile(m=line.m, n=line.n, pixel_dist=5, image=blured_image)
        print("12345")
    #f, axarr = plt.subplots(2, 2, constrained_layout=True)
    #axarr[0, 0].imshow(blured_image, cmap=plt.get_cmap('gray'))
    #axarr[0, 0].set_title("processed original")
    #axarr[1, 0].imshow(unfiltered_new_image, cmap=plt.get_cmap('gray'))
    #axarr[1, 0].set_title("unfiltered")
    #axarr[1, 1].imshow(filtered_new_image, cmap=plt.get_cmap('gray'))
    #axarr[1, 1].set_title("filtered")
    #plt.savefig('pictures/noise_reduction_demo.png', dpi=1200)


def find_lines(peaks: List, image_shape: List, image=None) -> List[Line]:
    lines: List[Line] = []
    for peak in peaks:
        m, n = get_m_n_line(r_cent=peak[0], theta=(peak[1] / image_shape[1]) * 180, image_size=image_shape)
        lines.append(Line(m=m, n=n))
        if image is not None:
            get_ij_line_by_equation(m=m, n=n, image=image, color=50)
    return lines


def get_m_n_line(r_cent, theta, image_size):
    dist = r_cent - image_size[0] // 2
    theta_rad = np.deg2rad(theta)
    intersec_y = image_size[1] // 2 + np.sin(theta_rad) * dist
    intersect_x = image_size[0] // 2 + np.cos(theta_rad) * dist
    m = -np.tan(np.pi / 2 - theta_rad)
    n = intersec_y - m * intersect_x
    return m, n


def get_line_profile(m: float, n: float, pixel_dist: int, image):
    (_, _), line_pixel_size = get_points_on_line_by_equation(m, n, image)
    (first_0, last_0),_ = get_points_on_line_by_equation(m, n + pixel_dist, image)
    (first_1, last_1),_ = get_points_on_line_by_equation(m, n - pixel_dist, image)
    pts1 = np.float32([first_0,
                       last_0,
                       first_1])
    pts2 = np.float32([[0, line_pixel_size - 1],
                       [0, 0],
                       [pixel_dist - 1, line_pixel_size - 1]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, (pixel_dist, line_pixel_size))
    return dst


def draw_example():
    image = np.zeros((512, 512))
    for n in range(-50, -29):
        get_ij_line_by_equation(2, n, image)
    image_border = np.zeros((512, 512))
    # for n in [-60,-20]:
    #    get_ij_line_by_equation(2, n, image_border)
    (first_0, last_0), _ = get_points_on_line_by_equation(2, -30, image)
    (first_1, last_1), _ = get_points_on_line_by_equation(2, -50, image)
    pts1 = np.float32([first_0,
                       last_0,
                       first_1])

    pts2 = np.float32([[0, 511],
                       [0, 0],
                       [19, 511]])
    # image_border = np.zeros((512, 40))
    M = cv2.getAffineTransform(pts1, pts2)
    rows, cols = image.shape
    dst = cv2.warpAffine(image, M, (20, 512))
    # rot_mat = cv2.getRotationMatrix2D([(first_0[0]-first_1[0])/2, (first_0[1]-first_1[1])/2], angle = 30,scale=1)
    # warp_rotate_dst = cv2.warpAffine(image, rot_mat, )
    print("1234")
    # pts_1 = [np.where()]
    # pts_2 = []


def get_points_on_line_by_equation(m: float, n: float, image, color=255):
    y = lambda x: m * x + n
    index_list = []
    pixel_line_size = 0
    image_height = image.shape[0]
    for j in range(image.shape[1]):
        i = int(image_height - y(j))
        if i >= 0 and i < image_height:
            index_list.append([j, i])
            pixel_line_size += 1
    return [index_list[0], index_list[-1]], pixel_line_size


def get_ij_line_by_equation(m: float, n: float, image, color=255):
    y = lambda x: m * x + n
    index_list = []
    image_height = image.shape[0]
    for j in range(image.shape[1]):
        i = int(image_height - y(j))
        if i >= 0 and i < image_height:
            image[i, j] = color
    return index_list


def filter_peaks(peaks: np.ndarray):
    """
    filtering peaks based on interval around the median of theta from the radon results
    :param peaks:
    :return:
    """
    thetas = peaks[0:, 1]
    median = np.median(thetas)
    q = 0.1  # quantile distance from median
    a = median * q
    elements = np.logical_and((median - a) < thetas, thetas < (median + a))
    filtered_peaks = peaks[elements]
    return filtered_peaks


def normalize_matrix(matrix: np.ndarray):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (((matrix - min_val)) / (max_val - min_val)) * 255


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


if __name__ == "__main__":
    res = radon_exmaple()
    #draw_example()
