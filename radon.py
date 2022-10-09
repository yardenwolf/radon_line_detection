import copy
from typing import List, Tuple
import pickle
from collections import deque

from skimage import transform
import cv2
import numpy as np
from skimage.feature.peak import peak_local_max
import math
from skimage.transform import warp_coords
from uuid import uuid4
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from itertools import product
from pathlib import Path
import torch

from utils import normalize_matrix
from labeler import Line
from image_loader import load_image, process_or_mean, process_or_percentile
from om_data.exported import ImageMetadataExported, ImageSegmentExported, SegmentedImageExported


def segment_lines_with_radon(file_name: str, output_prefix: str):
    current_path = Path(file_name)
    image, orig_images = load_image(file_name, process_or_percentile)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    res = transform.radon(image=image, theta=theta)
    kernel = np.ones((3, 7), np.uint8)
    peaks = peak_local_max(res, min_distance=5, num_peaks=30, footprint=kernel)
    theta_min, theta_max = find_most_frequent_theta_in_peaks(peaks, res, window_size=5)
    peaks = filter_peaks_by_range(theta_min, theta_max, peaks)
    lines = find_lines(peaks=peaks, image_shape=image.shape)
    first_image = normalize_matrix(copy.deepcopy(orig_images[0]))
    segments = []
    for index, line in enumerate(lines):
        # r, theta = get_r_theta_by_m_n(line.m, line.n, image_size=image.shape)
        get_ij_line_by_equation(line.m, line.n, first_image, color=255)
        prof, line_image, image_segment = get_line_profile_new(m=line.m, n=line.n, pixel_dist=10,
                                                               image=image,
                                                               image_sequence=orig_images)
        segments.append(image_segment)

    uuid = str(uuid4())
    plt.imshow(first_image, cmap=plt.get_cmap('gray'))
    plt.savefig(f"{output_prefix}/{current_path.stem}_{uuid}.svg", dpi=1200)
    image_metadata = ImageMetadataExported(file=file_name, processed_image_uuid=0, timestamps_sec=[], stage_x_um=0,
                                           stage_y_um=0, image_read_metadata={})

    exported_segments = SegmentedImageExported(image_metadata, segments)
    with open(f"{output_prefix}/{current_path.stem}_{uuid}.pickle", "wb") as f:
        pickle.dump(exported_segments, f)


def find_most_frequent_theta_in_peaks(peaks: np.ndarray, image, window_size: int):
    peaks_list = peaks.tolist()
    vals_matrix = np.zeros(image.shape)
    rows, cols = zip(*peaks_list)
    vals_matrix[rows, cols] = image[rows, cols]
    res = sliding_window_sum(np.sum(vals_matrix, axis=0), size=window_size)
    max_index = np.argmax(res)
    return (max_index, max_index + window_size - 1)


def sliding_window_sum(a, size):
    out = []
    the_sum = 0
    q = deque()
    for i in a:
        if len(q) == size:
            the_sum -= q[0]
            q.popleft()
        q.append(i)
        the_sum += i
        if len(q) == size:
            out.append(the_sum)
    return out


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


def get_line_profile_new(m: float, n: float, pixel_dist: int, image, image_sequence, len=100):
    """
    returns image with line on it and the straightened profile along the line y=mx+n
    :param m: the gradient
    :param n: the intercept
    :param pixel_dist:
    :param image: This is the processed image we use to work on without affecting the originals.
    :param image_sequence: the sequence of images (the 3d image). We'll extract the segments across these images to obtain segments.
    :return:
    """
    new_image = copy.deepcopy(normalize_matrix(image))
    curve_indices = get_ij_line_by_equation(m, n, new_image, color=255)
    pt_0, pt_1 = get_points_on_line_by_equation(m, n, image)
    line_length = int(math.dist(pt_0, pt_1))
    first_0, last_0 = get_points_on_line_by_equation(m, n + pixel_dist, image)
    first_1, last_1 = get_points_on_line_by_equation(m, n - pixel_dist, image)
    pts1 = np.float32([first_0,
                       last_0,
                       first_1])
    pts2 = np.float32([[0, pixel_dist - 1],
                       [line_length - 1, pixel_dist - 1],
                       [0, 0]])
    M = cv2.getAffineTransform(pts1, pts2)
    iM = cv2.invertAffineTransform(M=M)
    dst = cv2.warpAffine(new_image, M, (line_length, pixel_dist))
    com_i, com_j = center_of_mass(dst)

    profile_list = []
    for frame in range(0, image_sequence.shape[0]):
        affine_res = cv2.warpAffine(image_sequence[frame], M, (line_length, pixel_dist))
        profile_list.append(affine_res[:, max(int(com_j) - len, 0):min(int(com_j) + len, affine_res.shape[1])])
    profile_sequence = np.stack(profile_list, axis=0)

    inverse_func = get_inverse_affine(iM)
    coorods_map = warp_coords(inverse_func, dst.shape, int)
    coorods_map = coorods_map[:, :, max(int(com_j) - len, 0):min(int(com_j) + len, coorods_map.shape[2])]
    image_segment = ImageSegmentExported(int(uuid4()), profile_image_seq=profile_sequence,
                                         profile_coords_map=coorods_map,
                                         profile_curve_coords=np.stack(curve_indices))
    return dst, new_image, image_segment


def get_line_profile_using_com(m: float, n: float, pixel_dist: int, image, image_sequence, profile_len: int):
    """
    returns image with line on it and the straightened profile. older version of this function. not used any longer.
    :param m:
    :param n:
    :param pixel_dist:
    :param image:
    :param index:
    :return:
    """
    new_image = copy.deepcopy(normalize_matrix(image))
    curve_indices = get_ij_line_by_equation(m, n, new_image, color=255)
    com_i, com_j = find_center_of_mass_for_line(m, n, pixel_dist, image)
    p_0, p_1, p_2 = get_points_around_com(m, pixel_dist, com_i, com_j, profile_len=200, image_shape=image.shape)
    pts1 = np.float32([p_0,
                       p_1,
                       p_2])
    pts2 = np.float32([[0, 0],
                       [2 * pixel_dist - 1, 0],
                       [2 * pixel_dist - 1, profile_len - 1]])
    M = cv2.getAffineTransform(pts1, pts2)
    iM = cv2.invertAffineTransform(M=M)
    profile_list = []
    for frame in range(0, image_sequence.shape[0]):
        profile_list.append(cv2.warpAffine(image_sequence[frame], M, (pixel_dist, profile_len)))
    # profile_sequence = np.concatenate(profile_list, axis=0)
    profile_sequence = np.stack(profile_list, axis=0)
    dst = cv2.warpAffine(image, M, (pixel_dist, profile_len))
    inverse_func = get_inverse_affine(iM)
    coorods_map = warp_coords(inverse_func, dst.shape, int)
    image_segment = ImageSegmentExported(int(uuid4()), profile_image_seq=profile_sequence,
                                         profile_coords_map=coorods_map,
                                         profile_curve_coords=np.stack(curve_indices))
    return normalize_matrix(dst), new_image, image_segment


def find_center_of_mass_for_line(m: float, n: float, pixel_dist: int, image):
    """
    finding com across line y=mx+n
    :param m:
    :param n:
    :param pixel_dist:
    :param image:
    :return:
    """
    norm_image = normalize_matrix(image)
    for i, j in product(range(0, image.shape[0]), range(0, image.shape[1])):
        y_0 = m * j + n + pixel_dist
        y_1 = m * j + n - pixel_dist
        if image.shape[0] - i > y_0 or image.shape[0] - i < y_1:
            norm_image[i, j] = 0
    return center_of_mass(norm_image)


def get_points_around_com(m: float, pixel_dist: int, com_i: int, com_j: int, profile_len: int, image_shape: Tuple):
    alpha = np.arctan(m)
    vertical_dist = np.sin(alpha) * (profile_len / 2)
    horizantal_dist = np.cos(alpha) * (profile_len / 2)
    p_0 = (max(int(com_j - pixel_dist + horizantal_dist), 0), max(int(com_i - vertical_dist), 0))
    p_1 = (min(int(com_j + pixel_dist + horizantal_dist), image_shape[1]), max(int(com_i - vertical_dist), 0))
    p_2 = (max(int(com_j + pixel_dist - horizantal_dist), 0), min(int(com_i + vertical_dist), image_shape[0]))
    return p_0, p_1, p_2


def get_inverse_affine(iM):
    def apply(xy):
        ones = np.ones((xy.shape[0], 1))
        return np.matmul(np.append(xy, ones, axis=1), np.transpose(iM))

    return apply


def get_line_profile(m: float, n: float, pixel_dist: int, image):
    """
    returns image with line on it and the straightened profile
    :param m:
    :param n:
    :param pixel_dist:
    :param image:
    :param index:
    :return:
    """
    new_image = copy.deepcopy(normalize_matrix(image))
    curve_indices = get_ij_line_by_equation(m, n, new_image, color=255)
    pt_0, pt_1 = get_points_on_line_by_equation(m, n, image)
    line_length = int(math.dist(pt_0, pt_1))
    first_0, last_0 = get_points_on_line_by_equation(m, n + pixel_dist, image)
    first_1, last_1 = get_points_on_line_by_equation(m, n - pixel_dist, image)
    pts1 = np.float32([first_0,
                       last_0,
                       first_1])
    pts2 = np.float32([[0, line_length - 1],
                       [0, 0],
                       [pixel_dist - 1, line_length - 1]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, (pixel_dist, line_length))
    return normalize_matrix(dst), new_image, curve_indices


def get_points_on_line_by_equation(m: float, n: float, image, color=255):
    """
    returns two extreme points on line. Return format in (x,y) format.
    :param m:
    :param n:
    :param image:
    :param color:
    :return:
    """
    y = lambda x: m * x + n
    index_list = []
    image_height = image.shape[0]
    for j in range(image.shape[1]):
        i = int(image_height - y(j))
        if i >= 0 and i < image_height:
            index_list.append([j, i])
    return [index_list[0], index_list[-1]]


def get_ij_line_by_equation(m: float, n: float, image, color=255):
    """
    calculates the indices of the line in the image. if color is specified it
    colors the line according to those indices.
    :param m:
    :param n:
    :param image:
    :param color:
    :return:
    """
    y = lambda x: m * x + n
    index_list = []
    image_height = image.shape[0]
    for j in range(image.shape[1]):
        i = int(image_height - y(j))
        if i >= 0 and i < image_height:
            index_list.append([j, i])
            image[i, j] = color
    return index_list


def filter_peaks_by_range(min: int, max: int, peaks):
    """
    returns peaks from peaks list that fall within the (min,max) range.
    :param min:
    :param max:
    :param peaks:
    :return:
    """
    thetas = peaks[0:, 1]
    elements = np.logical_and(min <= thetas, thetas <= max)
    filtered_peaks = peaks[elements]
    return filtered_peaks


def filter_peaks(peaks: np.ndarray, q: float = 0.05):
    """
    filtering peaks based on interval around the median of theta from the radon results. Not used anymore.
    :param peaks:
    :param q: qunatile distance from median from both sides (negative and positive).
    :return:
    """
    thetas = peaks[0:, 1]
    median = np.mean(thetas)
    std = np.std(thetas)
    elements = np.logical_and((median - std) < thetas, thetas < (median + std))
    filtered_peaks = peaks[elements]
    return filtered_peaks


def find_max_n_elements(image: np.ndarray, n: int) -> List[Tuple]:
    new_image = np.copy(image)
    max_list = []
    for i in range(n):
        index = np.unravel_index(np.argmax(new_image), new_image.shape)
        max_list.append(index)
        new_image[index] = 0
    return max_list


def find_and_pool(image: np.ndarray, n: int) -> List[Tuple]:
    """
    peak selection function. Not used because we found the peak_local_max
    :param image:
    :param n:
    :return:
    """
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


def get_r_theta_by_m_n(m: float, n: float, image_size: Tuple):
    """
    get r,theta of radon transform by m,n of line equation
    :param m:
    :param n:
    :param image_size:
    :return:
    """
    theta_rad = np.pi / 2 - np.arctan(-m)
    dist = (n + m * (image_size[0] // 2) - image_size[1] // 2) / (np.sin(theta_rad) - m * np.cos(theta_rad))
    r_cent = dist + image_size[0] // 2
    theta_index = (np.rad2deg(theta_rad) / 180) * image_size[1]
    return int(r_cent), int(theta_index)


if __name__ == "__main__":
    res = segment_lines_with_radon(file_name="C:/Users/yarde/Documents/sample-data/2022-04-26/1008.nd2",
                                   output_prefix="./extracted")
