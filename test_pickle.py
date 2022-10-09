import pickle
from om_data import *
from skimage import io as sk_io

def load_pickle(path:str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    path_for_saving_segments = "./extracted/1008-segments/"
    for i,segment in enumerate(data.image_segments):
        sk_io.imsave(f'{path_for_saving_segments}/{i}.tiff', segment.profile_image_seq, photometric='minisblack')
    print("1234")

if __name__ == "__main__":
    load_pickle("./extracted/1008_d3d60056-414c-494e-88e3-c412504e83a1.pickle")
