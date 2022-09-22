import pickle
from om_data import *

def load_pickle(path:str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("1234")



if __name__ == "__main__":
    load_pickle("./extracted/1006_1d3bb264-73a8-4b00-8bd1-86a704ba0946.pickle")
