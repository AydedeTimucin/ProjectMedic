import os
import argparse
import pandas as pd
import CroppingTools as ct






tissue_maks_folder_dir = r'D:/AIN3007_project/outputs_test/tissue_masks'
tissue_maks_info_dir = r"D:/AIN3007_project/outputs_test/total_mask_info.csv"

save_dir = r'D:/AIN3007_project/outputs_test'


infos_df = pd.read_csv(tissue_maks_info_dir)
infos_df['mask_path'] = infos_df.apply(lambda row: ct.get_maskpath_for_df(tissue_maks_folder_dir, row), axis=1)

levels = [5]
sizes = [64,32]

for row in range(len(infos_df)):
    croppable = ct.Croppable(infos_df.iloc[row])
    ct.crop_and_save(croppable, levels, sizes, save_dir)