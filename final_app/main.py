import utils
import model

import os
import sys

import argparse
import openslide
import torch
import math
import numpy as np
import time
from PIL import Image
from tqdm import tqdm

start_time = time.time()

parser = argparse.ArgumentParser(description="Deep Learning Tissue Segmentation")

# Required arguments
parser.add_argument(
    "--wsi_paths",
    type=str,
    required=True,
    help="Path to the text file containing the list of WSI slides to segment (e.g., external_wsi_paths.txt)",
    )

# Optional arguments
parser.add_argument(
    "--output_folder",
    type=str,
    default="desired_path",
    help="Folder to save the predicted tissue masks (default: ./predicted_tissue_masks)",)

args = parser.parse_args()

wsi_paths_file = args.wsi_paths
output_folder = args.output_folder


with open(wsi_paths_file, "r") as f:
    wsi_paths = f.readlines()
    

for i in range(len(wsi_paths)):
    wsi_paths[i] = wsi_paths[i].strip()
    slide = openslide.OpenSlide(wsi_paths[i])

    stride = 512

    level = utils.pick_best_level(slide=slide, stride=stride, max_patch_num=400, min_patch_num=20)

    level_dim = level - slide.level_count

    prediction_model = model.UNet()
    saved_model_path = r"ProjectMedic/final_app/unet_best_dice(512).pth"
    prediction_model.load_state_dict(torch.load(saved_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print("Using device:", device)

    prediction_model.to(device)

    prediction_model.eval()

    # I edited this line -Burak
    downscale = int(slide.level_downsamples[level_dim])

    slide_width, slide_height = slide.level_dimensions[level]

    mask_width = math.ceil(slide_width / stride) * stride
    mask_height = math.ceil(slide_height / stride) * stride

    big_mask = np.zeros((mask_height, mask_width), dtype=np.float32)

    # Loop over the slide in 512x512 frames
    for y in tqdm(range(0, slide_height, stride), desc="Processing rows"):
        for x in range(0, slide_width, stride):
            # Extract a patch from the slide
            x_0 = int(x * downscale) 
            y_0 = int(y * downscale)

            patch = slide.read_region((x_0, y_0), level, (stride, stride))
            patch_np = np.array(patch)

            # Preprocess the patch
            patch_processed = utils.adaptive_histogram_equalization(image=patch_np)

            # Get the prediction
            prediction_mask = utils.mask_patch(patch=patch_processed, prediction_model=prediction_model, device=device, treshold=0.42)

            # Construct the mask
            big_mask[y:y + stride, x:x + stride] = prediction_mask

    # post_process the mask and save
    final_mask = utils.soften_binary_prediction(mask=big_mask, sigma=20)
    final_mask_image = (final_mask * 255).astype(np.uint8)
    image = Image.fromarray(final_mask_image)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    slide_name = os.path.splitext(wsi_paths[i])[0].split("\\")[-1].split("/")[-1]

    output_path = os.path.join(output_folder, f"{slide_name}_unet.png")
    image.save(output_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")
