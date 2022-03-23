from skimage import transform
import cv2
import numpy as np
from typing import List, Tuple
import torch

def radon_exmaple():
    image = cv2.imread("road-220058__480.jpg")
    blured_image = cv2.GaussianBlur(image,(5,5),0)
    gray_image = cv2.cvtColor(blured_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    res = transform.radon(image=edges, theta=theta)
    #indices = find_max_n_elements(image=res, n=200)
    val_and_index_list = find_and_pool(res, n=5)
    zero_image = np.zeros(res.shape)
    for val, i, j in val_and_index_list:
        zero_image[i,j] = val
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

def find_and_pool(image: np.ndarray, n: int) -> List[Tuple]:
    matrix = torch.tensor(image)
    matrix = torch.unsqueeze(matrix, 0)
    max_pool = torch.nn.MaxPool2d(kernel_size=(3,3), return_indices=True)
    vals, indices = max_pool(matrix)
    indices = torch.flatten(indices)
    j_indices = (indices % image.shape[1])
    i_indices = torch.floor(indices / image.shape[1]).to(torch.int)
    max_vals_list = list(map(lambda val, i, j: (val,i,j) ,vals.flatten().numpy(), i_indices.numpy(), j_indices.numpy()))
    max_vals_list.sort(key=lambda x: x[0], reverse=True)
    return max_vals_list[:n]


if __name__ == "__main__":
    radon_exmaple()
