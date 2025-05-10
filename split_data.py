import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Correct archive root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_ROOT = os.path.join(SCRIPT_DIR, '..', 'data', 'COD10K-v3')
TRAIN_SRC = os.path.join(ARCHIVE_ROOT, 'Train')
TEST_SRC = os.path.join(ARCHIVE_ROOT, 'Test')

# Output directory
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, '..', 'data', 'COD10K')

SPLITS = ['train', 'val', 'test']
SUBFOLDERS = ['Image', 'Masks']

def create_dirs():
    for split in SPLITS:
        for sub in SUBFOLDERS:
            os.makedirs(os.path.join(OUTPUT_ROOT, split, sub), exist_ok=True)

def get_file_pairs(image_dir, mask_dir):
    image_files = sorted(os.listdir(image_dir))
    pairs = []
    for f in image_files:
        if not f.startswith("COD10K-CAM"):
            continue
        mask_name = f.replace(".jpg", ".png")
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            pairs.append((os.path.join(image_dir, f), mask_path))
        else:
            print(f"⚠️ No matching mask for: {f}")
    return pairs

def copy_files(pairs, split):
    img_dst = os.path.join(OUTPUT_ROOT, split, "Images")
    mask_dst = os.path.join(OUTPUT_ROOT, split, "Masks")
    for img_path, mask_path in tqdm(pairs, desc=f"Copying {split} set"):
        shutil.copy(img_path, os.path.join(img_dst, os.path.basename(img_path)))
        shutil.copy(mask_path, os.path.join(mask_dst, os.path.basename(mask_path)))

def split_and_copy(val_ratio=0.15):
    print("Creating directories...")
    create_dirs()

    print("Processing training set...")
    train_images = os.path.join(TRAIN_SRC, 'Image')
    train_masks = os.path.join(TRAIN_SRC, 'GT_Object')
    print("Checking path:", train_images)
    print("Exists:", os.path.exists(train_images))

    train_pairs = get_file_pairs(train_images, train_masks)
    random.shuffle(train_pairs)
    val_size = int(len(train_pairs) * val_ratio)
    val_pairs = train_pairs[:val_size]
    train_pairs = train_pairs[val_size:]

    copy_files(train_pairs, 'train')
    copy_files(val_pairs, 'val')

    print("Processing test set...")
    test_images = os.path.join(TEST_SRC, 'Image')
    test_masks = os.path.join(TEST_SRC, 'GT_Object')
    test_pairs = get_file_pairs(test_images, test_masks)
    copy_files(test_pairs, 'test')

    print(f"✅ Done. Files copied to {OUTPUT_ROOT}/train, val, test")

if __name__ == "__main__":
    random.seed(42)
    split_and_copy()
