import copy
from typing import List, Tuple
import pickle

from labeler import Line
from image_loader import load_image, process_or_mean
from om_data.exported import ImageMetadataExported, ImageSegmentExported, SegmentedImageExported

from skimage import transform
import cv2
import numpy as np
import torch
from skimage.feature.peak import peak_local_max
import math
import imageio
from skimage import io as sk_io
from skimage.transform import warp_coords
from uuid import uuid4
import potrace
import matplotlib.pyplot as plt


def radon_exmaple(file_name):
    image, orig_images = load_image(file_name, process_or_mean)
    # image = np.zeros((512, 512))
    # blurred_image = cv2.medianBlur(image, 5)
    # cv2.line(image, (0,216), (512,512), color=(255,0,0))
    # cv2.line(image, (0, 0), (512, 512), color=(255, 0, 0))
    # cv2.line(image, (0, 0), (296, 512), color=(255, 0, 0))
    blurred_image = image
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    res = transform.radon(image=blurred_image, theta=theta)
    kernel = np.ones((3, 7), np.uint8)
    peaks = peak_local_max(res, min_distance=2, num_peaks=20, footprint=kernel)
    peaks = filter_peaks(peaks, q=0.01)
    # peak = np.unravel_index(np.argmax(res, axis=None), res.shape)
    # peaks = [peak]
    lines = find_lines(peaks=peaks, image_shape=blurred_image.shape)
    first_image = normalize_matrix(copy.deepcopy(orig_images[0]))
    segments = []
    for index, line in enumerate(lines):
        # r, theta = get_r_theta_by_m_n(line.m, line.n, image_size=image.shape)
        prof, line_image, image_segment = get_line_profile_new(m=line.m, n=line.n, pixel_dist=10, image=blurred_image,
                                                               image_sequence=orig_images)
        segments.append(image_segment)
        get_ij_line_by_equation(line.m, line.n, first_image, color=255)
        imageio.imwrite(f"./extracted/{index}_profile.png", prof)
        imageio.imwrite(f"./extracted/{index}_origianl.jpeg", line_image)
    # output = np.concatenate([np.expand_dims(first_image, axis=0), orig_images], axis=0)
    # sk_io.imsave('./extracted/output.svg', first_image, photometric='minisblack')
    # image_data = im.fromarray(first_image)
    # image_data.save('./extracted/output.svg')
    plt.imshow(first_image, cmap=plt.get_cmap('gray'))
    plt.savefig('./extracted/output.svg', dpi=1200)
    image_metadata = ImageMetadataExported(file=file_name, processed_image_uuid=0, timestamps_sec=[], stage_x_um=0,
                                           stage_y_um=0, image_read_metadata={})
    exported_segments = SegmentedImageExported(image_metadata, segments)
    with open("example.pickle", "wb") as f:
        pickle.dump(exported_segments, f)
    # f, axarr = plt.subplots(2, 2, constrained_layout=True)
    # axarr[0, 0].imshow(blurred_image, cmap=plt.get_cmap('gray'))
    # axarr[0, 0].set_title("processed original")
    # axarr[1, 0].imshow(unfiltered_new_image, cmap=plt.get_cmap('gray'))
    # axarr[1, 0].set_title("unfiltered")
    # axarr[1, 1].imshow(filtered_new_image, cmap=plt.get_cmap('gray'))
    # axarr[1, 1].set_title("filtered")
    # plt.savefig('pictures/noise_reduction_demo.png', dpi=1200)


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


def get_line_profile_new(m: float, n: float, pixel_dist: int, image, image_sequence):
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
    iM = cv2.invertAffineTransform(M=M)
    profile_list = []
    for frame in range(0, image_sequence.shape[0]):
        profile_list.append(np.rot90(cv2.warpAffine(image_sequence[frame], M, (pixel_dist, line_length))))
    profile_sequence = np.concatenate(profile_list, axis=0)
    dst = cv2.warpAffine(image, M, (pixel_dist, line_length))
    inverse_func = get_inverse_affine(iM)
    coorods_map = warp_coords(inverse_func, dst.shape, int)
    image_segment = ImageSegmentExported(int(uuid4()), profile_image_seq=profile_sequence,
                                         profile_coords_map=coorods_map,
                                         profile_curve_coords=np.stack(curve_indices))
    return normalize_matrix(dst), new_image, image_segment


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
    returns it in (x,y) format
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


def get_ij_line_by_equation(m: float, n: float, image, color=None):
    """
    calculates the indices of the line in the image. if color is specified it
    colors in the line according to those indices.
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


def filter_peaks(peaks: np.ndarray, q: float = 0.05):
    """
    filtering peaks based on interval around the median of theta from the radon results.
    :param peaks:
    :param q: qunatile distance from median from both sides (negative and positive).
    :return:
    """
    thetas = peaks[0:, 1]
    median = np.median(thetas)
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
    res = radon_exmaple(file_name="C:/Users/yarde/Documents/sample-data/2022-04-26/1001.nd2")
    # draw_example()
