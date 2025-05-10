import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from models.unet import UNet
from dataset.cod10k_dataset import COD10KDataset
from torchvision.utils import save_image
from tqdm import tqdm
import signal
from functools import wraps
import errno

# Timeout decorator for GraphCut
class TimeoutError(Exception):
    pass

def timeout(seconds=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

def _handle_timeout(signum, frame):
    raise TimeoutError("Operation timed out")

# Graphcut refinement with safety features
@timeout(10)  # 10 second timeout
def safe_grabcut(image, mask, rect=None):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    if rect is None:
        return cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    else:
        return cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

def graphcut_refinement(image_tensor, mask_tensor):
    try:
        image = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Pre-processing
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Adaptive thresholding with fallback
        _, fg = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
        _, bg = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Ensure minimum regions
        if np.count_nonzero(fg) < 100 or np.count_nonzero(bg) < 100:
            print("⚠️ Not enough FG/BG pixels - using raw mask")
            return mask_tensor

        gc_mask = np.zeros(mask.shape, np.uint8)
        gc_mask[bg == 255] = cv2.GC_BGD
        gc_mask[fg == 255] = cv2.GC_FGD
        gc_mask[(gc_mask != cv2.GC_BGD) & (gc_mask != cv2.GC_FGD)] = cv2.GC_PR_FGD

        # Run GrabCut with timeout protection
        try:
            safe_grabcut(image, gc_mask)
            refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype('float32')
            return torch.tensor(refined).unsqueeze(0)
        except TimeoutError:
            print("⚠️ GraphCut timed out - using raw mask")
            return mask_tensor
            
    except Exception as e:
        print(f"⚠️ Refinement failed: {str(e)} - using raw mask")
        return mask_tensor

# Directories
MODEL_PATH = "checkpoints/best_model.pth"
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = UNet().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    exit(1)

# Dataset
try:
    test_dataset = COD10KDataset(
        image_dir="data/COD10K/test/Images",
        mask_dir="data/COD10K/test/Masks"
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
except Exception as e:
    print(f"❌ Failed to load dataset: {str(e)}")
    exit(1)

# Evaluation
def evaluate():
    total_mae = 0
    processed_count = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for idx, (img, mask) in enumerate(pbar):
            try:
                # Memory monitoring
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    alloc = torch.cuda.memory_allocated()/1024**2
                    cached = torch.cuda.memory_reserved()/1024**2
                    pbar.set_postfix_str(f"GPU Mem: {alloc:.1f}/{cached:.1f} MB")

                img = img.to(device)
                mask = mask.to(device)

                # Prediction
                pred = model(img)
                pred_bin = (pred > 0.5).float()

                # Refinement
                refined = graphcut_refinement(img[0], pred_bin[0])
                refined = refined.to(device)

                # MAE
                mae = torch.abs(refined - mask[0]).mean().item()
                total_mae += mae
                processed_count += 1

                # Save every 10 samples
                if idx % 10 == 0:
                    save_path = os.path.join(SAVE_DIR, f"sample_{idx}.png")
                    save_image(refined, save_path)

            except Exception as e:
                print(f"\n⚠️ Error processing sample {idx}: {str(e)}")
                continue

    if processed_count > 0:
        print(f"\n✅ Completed {processed_count}/{len(test_loader)} samples")
        print(f"Average MAE: {total_mae / processed_count:.4f}")
    else:
        print("\n❌ No samples processed successfully")

if __name__ == "__main__":
    evaluate()