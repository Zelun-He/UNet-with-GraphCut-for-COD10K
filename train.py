import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet
from dataset.cod10k_dataset import COD10KDataset
import torch.optim as optim
from tqdm import tqdm

# Save path
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset loading
train_dataset = COD10KDataset(
    image_dir="camouflage_detection/data/COD10K/train/Images",
    mask_dir="camouflage_detection/data/COD10K/train/Masks",
    augment=True
)

val_dataset = COD10KDataset(
    image_dir="camouflage_detection/data/COD10K/val/Images",
    mask_dir="camouflage_detection/data/COD10K/val/Masks"
)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model
model = UNet().to(device)

# Loss functions
bce = nn.BCELoss()

def dice_loss(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def combined_loss(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
NUM_EPOCHS = 15
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = combined_loss(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print("Saved new best model.")

print("Training complete.")
