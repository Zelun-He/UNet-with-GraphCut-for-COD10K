import matplotlib.pyplot as plt
import torch

def show_prediction(image, mask, pred, save_path=None):
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[0].set_title("Image")
    ax[1].imshow(mask.squeeze().cpu(), cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred.squeeze().cpu().detach(), cmap="gray")
    ax[2].set_title("Prediction")
    for a in ax: a.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
