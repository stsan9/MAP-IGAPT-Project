import torch
from . import utils

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.autograd.set_detect_anomaly(False)
    
    torch.manual_seed(0)
    data = utils.SimpleData(jet_type="g", data_dir="./data", batch_size=0) #TODO: implement reading settings
    X_train = data.train
    X_test = data.test
    