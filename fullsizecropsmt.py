import os
import argparse
import cv2
import openslide
import numpy as np
import pandas as pd
import CroppingTools as ct
from PIL import Image






tissue_maks_folder_dir = r'D:/AIN3007_project/outputs_train/tissue_masks'
tissue_maks_info_dir = r"D:/AIN3007_project/outputs_train/total_mask_info.csv"

save_dir = r'D:/AIN3007_project/outputs_train/'


infos_df = pd.read_csv(tissue_maks_info_dir)
infos_df['mask_path'] = infos_df.apply(lambda row: ct.get_maskpath_for_df(tissue_maks_folder_dir, row, folder_name="binary_masks"), axis=1)

cropsize = 512
stride = 512

wanted_rlength = 16 # base rlenght is 0.25

all_patches_df = pd.DataFrame(columns=['patch_path', 'mask_patch_path', 'slide_id', 'source', 'patch_x', 'patch_y', ])
num_patches = 0
num_mask_patches = 0
for row in range(len(infos_df)):
    slide_path = infos_df.iloc[row]['slide_path']
    slide_id = infos_df.iloc[row]['slide_id']
    source = infos_df.iloc[row]['source']
    print("slide_id: ", slide_id)
    patches_slide_dir = os.path.join(save_dir, "patches", slide_id)
    os.makedirs(patches_slide_dir, exist_ok=True)
    
    slide = openslide.OpenSlide(slide_path)
    print("Level0 shape: ", slide.level_dimensions[0])
    mask_im = Image.open(infos_df.iloc[row]['mask_path']).convert('RGB')
    slide_im, downscale = ct.read_slide_to_level(slide, wanted_rlength)
    if mask_im.size != slide_im.size:
        ValueError("slide and mask size not equal")
    print("slide_im width: ", slide_im.width, "slide_im height: ", slide_im.height)
    
    patches, patch_coordinates = ct.crop_for_predict(slide_im, cropsize=cropsize, stride=stride)
    for (patch_x, patch_y), patch in zip(patch_coordinates, patches):
        patch_dir = os.path.join(save_dir, "patches", slide_id, str(patch_x)+"_"+str(patch_y)+".png")
        patch = Image.fromarray(patch)
        patch.save(patch_dir)
        all_patches_df.loc[num_patches] = [patch_dir, None, slide_id, source, patch_x, patch_y]
        num_patches += 1
        
    mask_slide_dir = os.path.join(save_dir, "mask_patches", slide_id)
    os.makedirs(mask_slide_dir, exist_ok=True)
    
    mask_patches, mask_patch_coordinates = ct.crop_for_predict(mask_im, cropsize=cropsize, stride=stride)
    for (mask_patch_x, mask_patch_y), mask_patch in zip(mask_patch_coordinates, mask_patches):
        mask_patch_dir = os.path.join(save_dir, "mask_patches", slide_id, str(mask_patch_x)+"_"+str(mask_patch_y)+".png")
        mask_patch = Image.fromarray(mask_patch)
        mask_patch.save(mask_patch_dir)
        all_patches_df.loc[num_mask_patches, 'mask_patch_path'] = mask_patch_dir
        num_mask_patches += 1
        
    print("------------------")

    #print("slide and mask equal pathes: ", len(patch_coordinates) == len(mask_patch_coordinates))
all_patches_df.to_csv(os.path.join(save_dir, f"all_patches_rlenght{wanted_rlength}.csv"), index=False)
print("Done")