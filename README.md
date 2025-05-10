# Camouflaged Object Detection with U-Net + GraphCut

This project uses deep learning to detect camouflaged objects that blend into their backgrounds using a custom U-Net model. Trained on the COD10K dataset, it outputs binary segmentation masks that highlight hidden subjects. Bonus: includes optional GraphCut refinement to sharpen predictions!

---

## How It's Made

**Tech used:** Python, PyTorch, torchvision, PIL, OpenCV

This model was built using a custom U-Net architecture with skip connections. It was trained on the COD10K dataset, filtering for images with valid segmentation masks. The loss function combines **Binary Cross Entropy** and **Dice Loss** to maximize overlap and handle class imbalance. After prediction, we apply **GraphCut** refinement for sharper object boundaries.

---

## Optimizations

- Used Dice loss for class imbalance in foreground/background.
- Filtered only CAM images with matching masks for clean training data.
- GraphCut used post-inference to refine predicted edges.
- Data resized to 256×256 to reduce memory without sacrificing accuracy.

---

## Lessons Learned

- Building a full segmentation pipeline—from data cleaning to evaluation—requires attention to both model design and dataset quirks.
- The COD10K dataset has inconsistencies (missing masks), so filtering and preprocessing are just as important as model architecture.
- Learned why **MSE** is a poor loss choice for binary segmentation and how **Dice loss** better captures shape/structure.

---

## Examples

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [COD10K Dataset](https://sites.google.com/view/ltnghia/projects/camouflage)

