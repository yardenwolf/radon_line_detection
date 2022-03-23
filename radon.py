from skimage import transform
import cv2
import numpy as np
from typing import List, Tuple


def radon_exmaple():
    image = cv2.imread("road-220058__480.jpg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Sobel(edges, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    res = transform.radon(image=edges, theta=theta)
    indices = find_max_n_elements(image=res, n=2)
    zero_image = np.zeros(res.shape)
    for index in indices:
        zero_image[index] = res[index]
    print("1234")
    reconstruction = transform.iradon(zero_image, theta=theta)
    print("123")


def find_max_n_elements(image: np.ndarray, n: int) -> List[Tuple]:
    new_image = np.copy(image)
    max_list = []
    for i in range(n):
        index = np.unravel_index(np.argmax(new_image), new_image.shape)
        max_list.append(index)
        new_image[index] = 0
    return max_list


if __name__ == "__main__":
    radon_exmaple()
