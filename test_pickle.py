import pickle
from om_data import *

def load_pickle(path:str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("1234")



if __name__ == "__main__":
    load_pickle("./extracted/1006_305c2b02-a6e7-4bd5-b283-b4a6e7d77bac.pickle")
