import os
import argparse
import cv2
import openslide
import pandas as pd
import CroppingTools as ct






tissue_maks_folder_dir = r'D:/AIN3007_project/outputs_test/tissue_masks'
tissue_maks_info_dir = r"D:/AIN3007_project/outputs_test/total_mask_info.csv"

save_dir = r'D:/AIN3007_project/outputs_test'


infos_df = pd.read_csv(tissue_maks_info_dir)
infos_df['mask_path'] = infos_df.apply(lambda row: ct.get_maskpath_for_df(tissue_maks_folder_dir, row, folder_name="binary_full_masks"), axis=1)

sizes = [512]

for row in range(len(infos_df)):
    slide_path = infos_df.iloc[row]['slide_path']
    slide_id = infos_df.iloc[row]['slide_id']
    
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.level_dimensions[2]
    print(slide.level_downsamples[2])
    Wrange, Hrange = ct.calculate_ranges((512,512), (width, height))
    print(width, height)
    print(Wrange * Hrange)
    
    
    
    break
    