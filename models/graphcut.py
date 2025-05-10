import cv2
import numpy as np

def apply_graphcut(image, mask, iter_count=5):
    """
    Applies GraphCut refinement to a U-Net output mask.

    Args:
        image (numpy.ndarray): Original RGB image (H, W, 3).
        mask (numpy.ndarray): Predicted mask (H, W), values in [0, 1].
        iter_count (int): Number of iterations for GraphCut.

    Returns:
        numpy.ndarray: Refined mask after GraphCut (binary).
    """
    mask = (mask * 255).astype(np.uint8)
    mask[mask > 0] = cv2.GC_PR_FGD
    mask[mask == 0] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply grabCut
    cv2.grabCut(image, mask, None, bgd_model, fgd_model, iter_count, mode=cv2.GC_INIT_WITH_MASK)

    # Convert mask to binary
    refined_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    return refined_mask
