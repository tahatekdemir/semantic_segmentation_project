import os
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

image_dir = "data/oxford-iiit-pet/images"
mask_dir = "data/oxford-iiit-pet/annotations/trimaps"

model = models.segmentation.fcn_resnet50(weights=None, weights_backbone=None)
model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
model.load_state_dict(torch.load("segmentation_model.pth", map_location=device))
model = model.to(device)
model.eval()

images = sorted([
    f for f in os.listdir(image_dir)
    if f.endswith(".jpg")
])

image_name = images[10]

image_path = os.path.join(image_dir, image_name)
mask_path = os.path.join(mask_dir, image_name.replace(".jpg", ".png"))

image = Image.open(image_path).convert("RGB")
real_mask = Image.open(mask_path).convert("L")

resized_image = TF.resize(image, (128, 128))
input_tensor = TF.to_tensor(resized_image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)["out"]
    prediction = torch.argmax(output, dim=1).squeeze().cpu()

real_mask = TF.resize(real_mask, (128, 128), interpolation=Image.NEAREST)
real_mask_tensor = torch.tensor(list(real_mask.getdata())).reshape(128, 128)
real_mask_tensor = (real_mask_tensor == 1).long()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(resized_image)
plt.title("original image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(real_mask_tensor, cmap="gray")
plt.title("real mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(prediction, cmap="gray")
plt.title("predicted mask")
plt.axis("off")

plt.savefig("prediction_result.png")
plt.show()