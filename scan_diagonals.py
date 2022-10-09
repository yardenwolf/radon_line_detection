from typing import List

import numpy as np

from labeler import Line
from utils import normalize_matrix


def scan_all_diagonals(lines: List[Line], image, num_to_return: int, step: int):
    """
    scanning all diagonals at a specific angle. The main idea was to check for lines that we missed.
    Was very noisy so we didn't use this in the end.
    :param lines:
    :param image:
    :param num_to_return:
    :param step:
    :return:
    """
    if not is_theta_significant(lines, 0.5):
        return
    mean_m = np.mean(list(map(lambda x: x.m, lines)))
    normalized_image = normalize_matrix(image)
    sums = []
    n_range = range(int(-mean_m * 511), 511) if mean_m > 0 else range(0, int(511 - mean_m * 511))
    for n in n_range:
        sums.append([n, sum_diagonal(mean_m, n, normalized_image)])
    sums.sort(key=lambda x: x[1], reverse=True)
    filtered_sums = [sums[0]]
    prev = sums[0]
    for curr_sum in sums:
        if abs(prev[0] - curr_sum[0]) >= step:
            filtered_sums.append(curr_sum)
            prev = curr_sum
    return filtered_sums[0:num_to_return + 1], mean_m


def sum_diagonal(m, n, image):
    y = lambda x: m * x + n
    sum = 0
    image_height = image.shape[0]
    counter = 0
    for j in range(image.shape[1]):
        i = int(image_height - y(j))
        if i >= 0 and i < image_height:
            sum += image[j, i]
            counter += 1
    return sum / counter


def is_theta_significant(lines: List[Line], epsilon: float) -> bool:
    std = np.std(list(map(lambda x: x.m, lines)))
    return std <= epsilon
