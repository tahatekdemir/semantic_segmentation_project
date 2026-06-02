import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import numpy as np


class pet_segmentation_dataset(Dataset):
    def __init__(self, root="data", split="trainval", image_size=128):
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True
        )

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = np.array(mask)

        # Oxford mask values:
        # 1 = pet
        # 2 = border
        # 3 = background
        # Biz pet ve border kısımlarını 1 yapıyoruz, background 0 oluyor.
        mask = np.where(mask == 3, 0, 1)

        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask