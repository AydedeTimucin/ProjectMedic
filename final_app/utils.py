import model

import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from scipy.ndimage import gaussian_filter


# Increases the contrast of tissue regions
def adaptive_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


# Picks an appropriate read level for specified stride and patch number
def pick_best_level(slide, stride, max_patch_num, min_patch_num):

    for idx, dims in enumerate(slide.level_dimensions):
        height = dims[0]
        width = dims[0]

        num_patches = (width // stride) * (height // stride)

        if num_patches > max_patch_num or min_patch_num < 20:
            continue

        else:
            level = idx
            print("Chosen level:", level)
            print("#Patches:", num_patches)
            return level


# Used to create binary masks
def mask_patch(patch: np.array, prediction_model: model.UNet(), device: torch.device, treshold = 0.5) -> torch.Tensor():
    patch_tensor = torchvision.transforms.ToTensor()(patch)
    patch_tensor = patch_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = prediction_model(patch_tensor)
        threshold = treshold
        binary_prediction = (prediction > threshold).int()
        binary_prediction_np = binary_prediction.cpu().numpy()[0, 0]

    return binary_prediction_np


# Used in order to get more uniform masks
def soften_binary_prediction(mask, sigma=1.0):
    # Apply Gaussian smoothing
    smoothed_mask = gaussian_filter(mask.astype(float), sigma=sigma)

    # Convert back to binary
    softened_mask = (smoothed_mask > 0.5).astype(int)

    return softened_mask



