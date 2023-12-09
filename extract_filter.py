import torch
import io
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from PIL import Image
import os
import time
import matplotlib.pyplot as plt

def load_model_script(model_path: str) -> nn.Module:
    print(f"Loading model: {model_path}")
    return torch.jit.load(model_path)

model = load_model_script("./models/3-64x64-CPUModel-89.pts").to('cpu')

# JIT compile the model
example_input = torch.rand(1, 3, 64, 64)

first_conv_layer = None
for name, layer in model.named_modules():
    if name == "conv_block_3.0":
        first_conv_layer = layer
        break

if first_conv_layer is not None:
    filters = first_conv_layer.weight.data.cpu().numpy()
    num_filters = filters.shape[0]
    num_cols = 12
    num_rows = num_filters // num_cols + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4,4))
    for i, ax in enumerate(axes.flatten()):
        if i < num_filters:
            filter_img = filters[i]
            for j in range(3):
                ax.imshow(filter_img[j], cmap='viridis')
            ax.axis('off')
        else:
            ax.axis('off')
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
