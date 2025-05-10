# Camouflaged Object Detection with U-Net + GraphCut

This project uses deep learning to detect camouflaged objects that blend into their backgrounds using a custom U-Net model. Trained on the COD10K dataset, it outputs binary segmentation masks that highlight hidden subjects. Bonus: includes optional GraphCut refinement to sharpen predictions!

---

## How It's Made

**Tech used:** Python, PyTorch, torchvision, PIL, OpenCV

This model was built using a custom U-Net architecture with skip connections. It was trained on the COD10K dataset, filtering for images with valid segmentation masks. The loss function combines **Binary Cross Entropy** and **Dice Loss** to maximize overlap and handle class imbalance. After prediction, we apply **GraphCut** refinement for sharper object boundaries.

---

## Run i

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
##If you want to try it out
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/zelun-he/UNet-with-GraphCut-for-COD10K.git
cd UNet-with-GraphCut-for-COD10K
2. Create Virtual Environment and Install Dependencies
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
If you are using GitHub Codespaces, the environment is already active. Just run:

nginx
Copy
Edit
pip install -r requirements.txt
Download and Prepare the COD10K Dataset
Download the COD10K-v3 dataset from the official source or mirror:

Official repo: https://github.com/CODAI/COD10K

Google Drive: https://drive.google.com/file/d/1FxbyulYohQ1c7D7ZbFYoiK4vWaBzvYz5/view

Extract the dataset into:

bash
Copy
Edit
camouflage_detection/data/COD10K-v3/
The folder structure should look like this:

swift
Copy
Edit
camouflage_detection/data/COD10K-v3/Train/Images
camouflage_detection/data/COD10K-v3/Train/GT_Object
camouflage_detection/data/COD10K-v3/Test/Images
camouflage_detection/data/COD10K-v3/Test/GT_Object
Run the data splitter script to organize the dataset into train, val, and test folders:

bash
Copy
Edit
python camouflage_detection/data/split_data.py
Train the Model
Run the following to train the U-Net model:

bash
Copy
Edit
python camouflage_detection/train.py
This will:

Train the model on the COD10K dataset

Save the best model to camouflage_detection/checkpoints/best_model.pth

Evaluate the Model
Once training is complete, you can evaluate the model and apply GraphCut refinement:

bash
Copy
Edit
python camouflage_detection/evaluate.py
This will:

Run the model on the test set

Save predictions to camouflage_detection/outputs/

Print evaluation metrics (e.g., MAE)



## Examples

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [COD10K Dataset](https://sites.google.com/view/ltnghia/projects/camouflage)

