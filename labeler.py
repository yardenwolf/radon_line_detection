from collections import namedtuple
from typing import List

from skimage import data
import napari
from image_loader import load_image, process_or_percentile, load_image_for_labeling
import numpy as np
import pandas as pd

from utils import add_contrast

Line = namedtuple('Line', ['m','n'])


def view_dataset_in_napari(data_path: str):
    all_images = load_image_for_labeling(data_path, process_or_percentile)
    viewer = napari.view_image(all_images, rgb=True)
    napari.run()


def center_image_around_0(coordinates_df: pd.DataFrame, length: int, width: int):
    coordinates_df["axis-2"] = coordinates_df["axis-2"] - width // 2
    coordinates_df["axis-1"] = coordinates_df["axis-1"] - length // 2
    return coordinates_df


def extract_angles_from_lables(lable_path: str) -> List[Line]:
    labeled_data: pd.DataFrame = pd.read_csv(lable_path)
    # centered_labled_data = center_image_around_0(labeled_data, length=512, width=512)
    lines_data = labeled_data.groupby("index")
    lines = []
    for line_number, line_df in lines_data:
        base_index = line_number * 2
        x_0 = line_df["axis-2"][base_index]
        y_0 = 512 - line_df["axis-1"][base_index]
        x_1 = line_df["axis-2"][base_index + 1]
        y_1 = 512 - line_df["axis-1"][base_index + 1]
        dy = y_1 - y_0
        dx = x_1 - x_0
        m = dy / dx
        # theta is the angle between the origin and closest point from the origin to the line
        theta_radians = np.pi / 2 - np.arctan(np.abs(m))
        cos_theta = np.cos(theta_radians)
        # y_star = y_0 - m * x_0
        # r = cos_theta * y_star
        x_star = x_0 - y_0 / m
        r = ((cos_theta) ** 2) * x_star
        # calculating the intersection between the line and x axis
        # r = x_1 * np.cos(theta_radians) + y_1 * np.sin(theta_radians)
        lines.append(Line(theta=np.rad2deg(theta_radians), r=r))
    return lines


if __name__ == "__main__":
    # view_dataset_in_napari("C:/Users/yarde/Documents/sample-data/2022-02-10/1824.nd2")
    extract_angles_from_lables("C:/Users/yarde/Documents/test_lines.csv")
