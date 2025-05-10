# test_single_image.py

import torch
import cv2
import numpy as np
from models.unet import UNet
from torchvision import transforms
from torchvision.utils import save_image
import os

# GraphCut refinement function
def graphcut_refinement(image_tensor, mask_tensor):
    image = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    mask = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, fg = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
    _, bg = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV)
    gc_mask = np.zeros(mask.shape, np.uint8)
    gc_mask[bg == 255] = 0
    gc_mask[fg == 255] = 1
    gc_mask[(gc_mask != 0) & (gc_mask != 1)] = 2

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    refined = np.where((gc_mask == 1) | (gc_mask == 3), 1, 0).astype('float32')
    return torch.tensor(refined).unsqueeze(0)  # (1, H, W)

# Paths
MODEL_PATH = "checkpoints/best_model.pth"
IMAGE_PATH = "sample_inputs/your_image.jpg"   # <- Replace with your test image path
SAVE_PATH = "outputs/test_single_output.png"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load image
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))

# Transform
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    pred = model(image_tensor)
    pred_bin = (pred > 0.5).float()

# Apply GraphCut refinement (optional)
refined = graphcut_refinement(image_tensor[0].cpu(), pred_bin[0].cpu())
save_image(refined, SAVE_PATH)
print(f"Saved refined mask to: {SAVE_PATH}")
