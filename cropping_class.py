import os
from PIL import Image
import numpy as np
import pandas as pd
import openslide
import imageio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
#Currently works with: TCGA, Camelyon17, HEROHE, HER2

class Croppable:
    
    def __init__(self,df_row) -> None:
        self.slide_loaded : bool = False
        self.mask_path = str(df_row.iloc[0])
        self.slide_path = str(df_row.iloc[1])
        
        self.mask_Rlength = float(df_row.iloc[3])
        self.mask_level = np.log2(round(self.mask_Rlength/0.25)) #For making sure, to get from text use: int(float(df_row.iloc[2]))
        self.patient_id = str(df_row.iloc[4])
        self.slide_id = str(df_row.iloc[5])
        self.source = df_row.iloc[6]
        self.roi = [int(i) for i in df_row.iloc[7].split("-")]
        
        self.mpp = None
        self.slide_base_level = None
        self.slide_shape = None
        
        
        self.__mask_downscale = int(2 ** self.mask_level)
        self.__slide_downscale = None
        
    
    def Load_Slide(self):
        """
        Loads the slide at the specified level.
        
        Returns: OpenSlide object
        """
        
        slide = openslide.OpenSlide(self.slide_path)
        try:
            mpp = slide.properties["openslide.mpp-x"]
        except KeyError:
            try:
                tiff_res = slide.properties["tiff.ResolutionUnit"]
                if tiff_res == "centimeter":
                    mpp = 10000 / float(slide.properties["tiff.XResolution"])
                elif tiff_res == "inch":
                    mpp = 25400 / float(slide.properties["tiff.XResolution"])
                else:
                    raise ValueError("Unknown tiff.ResolutionUnit: {}".format(tiff_res))
            except KeyError:
                raise KeyError("No Resolution property in slide properties")
        mpp = float(mpp)
        if mpp < 0.1:
            raise ValueError("mpp is too small: {}".format(mpp))
        elif mpp < 0.3:
            self.mpp = 0.25
            self.slide_base_level = 0
        elif mpp < 0.6:
            self.mpp = 0.5
            self.slide_base_level = 1
        else:
            raise ValueError("mpp is too large: {}".format(mpp))
        self.__slide_downscale = int(2 ** self.slide_base_level)
        
        self.slide_shape = slide.level_dimensions[0]
        self.__level_downsamples = slide.level_downsamples
        
        self.slide_loaded = True
        return slide
    
    
    def get_mask_downscale(self):
        self.__mask_downscale = int(2 ** self.mask_level)
        return self.__mask_downscale
    
    def get_slide_downscale(self):
        if not self.slide_loaded:
            raise ValueError("Slide must be loaded using Load_Slide() first.")
        self.__slide_downscale = int(2 ** self.slide_base_level)
        return self.__slide_downscale
    
    
    def calculate_mask_crops(self, crop_level:int, crop_size:int, stride:None or int = None):
        """
        Calculate the arguments for cropping the mask.
        
        Returns: (mask_cropsize, mask_stride)
         
        Mask cropsize and stride must be integers.
        """
        if not self.slide_loaded:
            raise ValueError("Slide must be loaded using Load_Slide() first.")
        
        if stride is None:
            stride = crop_size
        crop_downscale = 2**crop_level
    
        
        mask_cropsize = crop_size * (crop_downscale / self.__mask_downscale)
        mask_stride = stride * (crop_downscale / self.__mask_downscale)
        
        if not (mask_cropsize % 1 == 0 and mask_stride % 1 == 0):
            raise ValueError("Mask cropsize and stride must be integers")
        
        return (int(mask_cropsize), int(mask_stride))

    
    def calculate_baselevel_crops(self, crop_level:int, crop_size:int, stride:None or int = None, mode:str = "base"):
        """
        Calculate the arguments for cropping the slide and mask.
        
        if the mode == "base"
            Returns: (base_level_cropsize, base_level_stride), base_openslide_level
        
        if the mode == "best"
            Returns: (best_level_cropsize, best_level_stride), best_openslide_level
        """
        if not self.slide_loaded:
            raise ValueError("Slide must be loaded using Load_Slide() first.")
        
        if stride is None:
            stride = crop_size
        crop_downscale = 2**crop_level
        if mode == "base":
            
            base_level_cropsize = int(crop_size * (crop_downscale / self.__slide_downscale))
            base_level_stride = int(stride * (crop_downscale / self.__slide_downscale))
        
            return (base_level_cropsize, base_level_stride), 0
        
        elif mode == "best":
            best_level = 0
            best_downsample = 1
            for i in range(len(self.__level_downsamples)-1,0 ,-1):
                if round(self.__level_downsamples[i]) < crop_downscale:
                    best_level = i
                    best_downsample = self.__level_downsamples[i]
                    break
                
            best_cropsize = int(crop_size * (crop_downscale / best_downsample))
            best_stride = int(stride * (crop_downscale / best_downsample))
            
            return (best_cropsize, best_stride), best_level
        else:
            raise ValueError("Mode must be 'base' or 'best'")
        
    def calculate_roi_for_level(self, wanted_level:int):
        """
        Calculates the ROI for the given level.
        Calculated ROI could be slightly different than the actual ROI.
        Returns: (X,Y,Width,Height)
        """
        if not self.slide_loaded:
            raise ValueError("Slide must be loaded using Load_Slide() first.")
        
        if wanted_level == self.mask_level:
            return self.roi
        
        downsample = 2 ** (wanted_level - self.mask_level)
        
        roi = [int(np.ceil(i / downsample)) for i in self.roi]
        
        return roi 

def calculate_ranges(strides, shape):
        # ROI(X-Y-Width-Height) 
        if len(shape) != len(strides):
            raise ValueError("Shape and strides must have the same shape.")
        dim_ranges = []
        for i in range(len(shape)):
            dim_ranges.append(int(np.ceil(shape[i] / strides[i])))  

        return dim_ranges

def pad_crop_right(patch, size, remaining_current_width, pad_with = None):
    "pad_with: None, 'white' or 'black'"
    if pad_with == None:
        return patch
    elif pad_with == "white":
        patch[:,size - remaining_current_width:size,:] = 255
        return patch
    elif pad_with == "black":
        patch[:,size - remaining_current_width:size,:] = 0
        return patch
    else:
        raise ValueError("pad_with must be None, 'white' or 'black'")

def pad_crop_bottom(patch, size, remaining_current_height, pad_with = None):
    "pad_with: None, 'white' or 'black'"
    if pad_with == None:
        return patch
    elif pad_with == "white":
        patch[size - remaining_current_height:size,:,:] = 255
        return patch
    elif pad_with == "black":
        patch[size - remaining_current_height:size,:,:] = 0
        return patch
    else:
        raise ValueError("pad_with must be None, 'white' or 'black'")





def crop_and_save(croppable: Croppable, crop_level: int, crop_size: int ):
    try:
        slide = croppable.Load_Slide()


        # Current Level Variables
        (_, _, CL_w, CL_h) = croppable.calculate_roi_for_level(crop_level)
        CL_Wrange, CL_Hrange = calculate_ranges((crop_size,crop_size),(CL_w,CL_h))
        remaining_current_width = (crop_size * CL_Wrange) - CL_w
        remaining_current_height = (crop_size * CL_Hrange) - CL_h


        # Mask Variables
        mask_lvl = croppable.mask_level
        mask_im = np.array(Image.open(croppable.mask_path))[:,:]/255
        mask_im = mask_im.astype(bool)
        mask_im = np.invert(mask_im) #False = bg, True = tissue
        (mask_x, mask_y, _, _) = croppable.calculate_roi_for_level(croppable.mask_level)
        (mask_cropsize, mask_stride) = croppable.calculate_mask_crops(crop_level,crop_size)


        # Base Level Variables
        slide_base_level = croppable.slide_base_level
        (base_x, base_y, _, _) = croppable.calculate_roi_for_level(0)
        (base_cropsize, base_stride), read_level = croppable.calculate_baselevel_crops(crop_level,crop_size)


        # Accepted Patch Mask
        accepted_patch_mask = np.ones(mask_im.shape)*255

        # Numbers to keep track
        num_patches = 0
        num_tissue_patches = 0
        num_bg_patches = 0

        # Crop and save
        for row in range(CL_Hrange): #height
            for col in range(CL_Wrange): #width
                if col != CL_Wrange - 1:
                    continue
                    
                
                patch = slide.read_region((base_x + col*base_stride,base_y + row*base_stride),read_level,(base_cropsize,base_cropsize))
                
                if crop_level / slide_base_level > 1:
                    patch = patch.resize((crop_size, crop_size), Image.LANCZOS)
                    
                patch = np.array(patch)[:,:,:3]
                
                if row == CL_Hrange - 1:
                    patch = pad_crop_bottom(patch, crop_size, remaining_current_height, pad_with = "white")
                if col == CL_Wrange - 1:
                    patch = pad_crop_right(patch, crop_size, remaining_current_width, pad_with = "white")
                
                patch_mask = mask_im[mask_y + row*mask_stride:mask_y + row*mask_stride + mask_cropsize,mask_x + col*mask_stride:mask_x + col*mask_stride + mask_cropsize]
                
                tissue_ratio = np.sum(patch_mask) / (mask_cropsize**2)
                if tissue_ratio > 0.1:
                    accepted_patch_mask[mask_y + row*mask_stride:mask_y + row*mask_stride + mask_cropsize,mask_x + col*mask_stride:mask_x + col*mask_stride + mask_cropsize] = 0
                    lbl = "tissue"
                    num_tissue_patches += 1
                else:
                    lbl = "background"
                    num_bg_patches
                
                patch = Image.fromarray(patch).show()
                
    except Exception as e:
        print(e)
        print("Error occured at: ", croppable.slide_id)
        return 0

df = pd.read_csv(r"Burak_code\outputs_test\Tissue_Masks\TCGA_masks\test_tissue_mask_info.txt")

level = 6
size = 32

#load class and slide
croppable = Croppable(df.iloc[0])
crop_and_save(croppable, level, size)
