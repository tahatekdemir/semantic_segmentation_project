import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms.functional as TF
from torchvision import models
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)


image_dir = "data/oxford-iiit-pet/images"
mask_dir = "data/oxford-iiit-pet/annotations/trimaps"


class PetSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)

        mask_name = image_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)

        image = TF.to_tensor(image)

        mask = torch.tensor(list(mask.getdata()), dtype=torch.long).reshape(128, 128)

        # Oxford-IIIT Pet trimap:
        # 1 = pet, 2 = background, 3 = border
        # We convert it to binary segmentation:
        # pet = 1, everything else = 0
        mask = (mask == 1).long()

        return image, mask


dataset = PetSegmentationDataset(image_dir, mask_dir)

print("total images:", len(dataset))

# CPU yavaş olduğu için ilk denemede küçük subset kullanıyoruz
subset_size = min(6000, len(dataset))
dataset = Subset(dataset, range(subset_size))

print("used images:", len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


model = models.segmentation.fcn_resnet50(weights=None, weights_backbone=None)
model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 50

train_losses = []
val_accuracies = []


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)["out"]
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval()
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            predictions = torch.argmax(outputs, dim=1)

            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()

    pixel_accuracy = 100 * correct_pixels / total_pixels
    val_accuracies.append(pixel_accuracy)

    print(f"epoch [{epoch+1}/{epochs}] loss: {avg_loss:.4f} validation pixel accuracy: {pixel_accuracy:.2f}%")


torch.save(model.state_dict(), "segmentation_model.pth")
print("model saved as segmentation_model.pth")


plt.figure()
plt.plot(train_losses)
plt.title("training loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("training_loss.png")
plt.show()


plt.figure()
plt.plot(val_accuracies)
plt.title("validation pixel accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("validation_accuracy.png")
plt.show()