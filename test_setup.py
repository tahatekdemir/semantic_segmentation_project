import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("numpy version:", np.__version__)

if torch.backends.mps.is_available():
    print("device: mps")
elif torch.cuda.is_available():
    print("device: cuda")
else:
    print("device: cpu")