import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ------------- Initialize PyTorch and GPU -------------
print(f"pytorch version: {torch.__version__}")
device = ''

if not torch.cuda.is_available():
    print("CUDA is not enabled on your system. Please enable it to train the model on the GPU.")
    print("Download the right version of pytorch here https://download.pytorch.org/whl/cu118")
    device = 'cpu'
else:
    print(f"CUDA is enabled on this system.")
    print(f"CUDA version: {torch.version.cuda}")
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
    device = 'cuda'

# ------------- Data Loading and preparing functions -------------
# TODO: Data preperation functions

# ------------- Define the model architecture -------------
# TODO: Define model

# ------------- Define loss function, and optimizer -------------
# TODO: loss function, optimizer

# ------------- Training and Testing loop -------------
# TODO: training and testing loop
# TODO: save model with generated name, and load model with Name
# TODO: visualize loss function curve, and keep a log of epochs and accuracy

# ------------- API -------------
# TODO: implement useful API functions like predict(image)

