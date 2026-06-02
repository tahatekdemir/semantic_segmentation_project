from dataset import pet_segmentation_dataset
from torch.utils.data import DataLoader


dataset = pet_segmentation_dataset(root="data", split="trainval", image_size=128)

print("dataset size:", len(dataset))

image, mask = dataset[0]

print("image shape:", image.shape)
print("mask shape:", mask.shape)
print("image min:", image.min().item())
print("image max:", image.max().item())
print("mask unique values:", mask.unique())