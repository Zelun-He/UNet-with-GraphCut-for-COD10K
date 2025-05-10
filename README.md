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
##How to Set Up and Run the Project

Clone or download the repository to your computer.

Make sure you have Python 3.8 or later installed. Then install the required libraries using the list provided in the requirements.txt file. You can do this with a Python package manager.

Download the COD10K-v3 dataset from the official repository hosted on GitHub (search for "LiuYingzheng/COD10K"). After downloading, extract the archive so that the Train and Test folders are placed inside the camouflage_detection/data/COD10K-v3/ directory in your project.

To organize the data into training, validation, and testing folders, run the split_data.py script located in the camouflage_detection/data/ directory.

After the data is prepared, you can start training the model by running the train.py script from the main project directory.

Once training is complete, you can evaluate the model and generate output masks by running the evaluate.py script.


