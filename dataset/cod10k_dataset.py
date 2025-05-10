import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
import torch

class COD10KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment

        self.image_filenames = []
        for f in sorted(os.listdir(image_dir)):
            if not f.lower().endswith(".jpg"):
                continue
            mask_name = f.replace(".jpg", ".png")
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.image_filenames.append((f, mask_name))
            else:
                print(f"⚠️ Mask missing for: {f}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name, mask_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        if self.augment and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask
