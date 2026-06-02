import torch
from model import unet_model


model = unet_model(in_channels=3, out_channels=1)

x = torch.randn(1, 3, 128, 128)

y = model(x)

print("input shape:", x.shape)
print("output shape:", y.shape)