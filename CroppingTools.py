import os
from PIL import Image
import numpy as np
import pandas as pd
import openslide
import imageio
from tqdm.auto import tqdm


def get_maskpath_for_df(root, df_row, folder_name = "binary_masks"):
    slide_id = df_row['slide_id']
    source = df_row['source']
    mask_path = os.path.join(root, source, folder_name , slide_id + '.png')
    return mask_path

class Croppable:
    
    def __init__(self,df_row) -> None:
        self.slide_loaded : bool = False
        self.mask_path = str(df_row.iloc[5])
        self.slide_path = str(df_row.iloc[2])
        
        self.mask_Rlength = float(df_row.iloc[3])
        self.mask_level = np.log2(round(self.mask_Rlength/0.25)) #For making sure, to get from text use: int(float(df_row.iloc[2]))
        self.slide_id = str(df_row.iloc[0])
        self.source = df_row.iloc[1]
        self.roi = [int(i) for i in df_row.iloc[4].split("-")]
        
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





def crop_and_save(croppable: Croppable, crop_level_list: list, crop_size_list: list, out_dir):
    
    def patch_n_mask_division(patch_im, patch_num:int, patch_coords, mask_coords:tuple, mask_arr, level_0_size:int, wanted_level:int, wanted_size:int):
        """
        returns list of patches, patches_info, accepted_mask_patch, num_tissue_patches
        
        patch_im: patch image
        patch_num: number of the original patch
        patch_coords: base level coordinates(x/width, y/height) of the patch (top left corner)
        mask_coords: mask coordinates(x/width, y/height) of the patch (top left corner)
        mask_arr: mask np.array
        level_0_size: size of the patch at level 0 (2**croplevel x cropsize)
        wanted_level: level of the wanted patch
        wanted_size: size of the wanted patch
        """
        
        wanted_0_size = (2**wanted_level) * wanted_size
        
        if level_0_size < wanted_0_size:
            raise ValueError("Cannot crop to a level/size larger than the given level/size.")
        
        original_to_small = level_0_size / wanted_0_size
        if not original_to_small % 1 == 0:
            raise ValueError("level_0_size must be a multiple of wanted_0_size")
        
        original_to_small = int(original_to_small)
        patch_arr = np.array(patch_im)[:,:,:3]
        
        small_patch_size = patch_arr.shape[0] / original_to_small
        small_mask_size = mask_arr.shape[0] / original_to_small
        accepted_mask = np.ones(mask_arr.shape)*255
        
        patches = []
        patch_infos = []
        num_patch = original_to_small**2 * patch_num
        num_tissue = 0
        for row in range(original_to_small):
            for col in range(original_to_small):
                patch = patch_arr[int(row*small_patch_size):int((row+1)*small_patch_size),int(col*small_patch_size):int((col+1)*small_patch_size),:]
                patch = Image.fromarray(patch)
                patch = patch.resize((wanted_size, wanted_size), Image.LANCZOS)
                patches.append(patch)

                mask_patch = mask_arr[int(row*small_mask_size):int((row+1)*small_mask_size),int(col*small_mask_size):int((col+1)*small_mask_size)]
                tissue_ratio = np.sum(mask_patch) / (small_mask_size**2)
                if tissue_ratio > 0.1:
                    accepted_mask[int(row*small_mask_size):int((row+1)*small_mask_size),int(col*small_mask_size):int((col+1)*small_mask_size)] = 0
                    lbl = "tissue"
                    num_tissue += 1
                else:
                    lbl = "background"
                #                   patch_num, lbl, tissue_ratio, bg_ratio, base_level_coords(x,y), mask_coords(x,y), mask_level
                patch_infos.append((num_patch,
                                    lbl,
                                    round(tissue_ratio, 3),
                                    round(1- tissue_ratio, 3),
                                    (int(patch_coords[0] + col*small_patch_size), int(patch_coords[1] + row*small_patch_size)),
                                    (int(mask_coords[0] + col*small_mask_size), int(mask_coords[1] + row*small_mask_size))
                                    ))
                num_patch += 1

        return patches, patch_infos, accepted_mask, num_tissue
    
    
    
    
    
    
    
    
    
    try:
        slide = croppable.Load_Slide()

        crop_level = max(crop_level_list)
        crop_size = max(crop_size_list)
        level_0_size = (2**crop_level) * crop_size
        

        # Current Level Variables
        (_, _, CL_w, CL_h) = croppable.calculate_roi_for_level(crop_level)
        CL_Wrange, CL_Hrange = calculate_ranges((crop_size,crop_size),(CL_w,CL_h))

        # Mask Variables
        mask_lvl = croppable.mask_level
        mask_im = np.array(Image.open(croppable.mask_path))[:,:]/255
        mask_im = mask_im.astype(bool)
        mask_im = np.invert(mask_im) #False = bg, True = tissue
        (mask_x, mask_y, _, _) = croppable.calculate_roi_for_level(croppable.mask_level)
        (mask_cropsize, mask_stride) = croppable.calculate_mask_crops(crop_level,crop_size)
        
        if mask_cropsize / 2**(min(crop_size_list) - crop_size) < 1:
            raise ValueError("Some levels are too small for the mask.")
        
        # Base Level Variables
        slide_base_level = croppable.slide_base_level
        (base_x, base_y, base_width, base_height) = croppable.calculate_roi_for_level(slide_base_level)
        (base_cropsize, base_stride), read_level = croppable.calculate_baselevel_crops(crop_level,crop_size)
        base_Wrange, base_Hrange = calculate_ranges((base_cropsize,base_cropsize),(base_width,base_height))
        remaining_base_width = (base_cropsize * base_Wrange) - base_width
        remaining_base_height = (base_cropsize * base_Hrange) - base_height


        # Directory Variables
        out_dir = os.path.join(out_dir,"Cropped_slides", croppable.source)
        slide_dirs_dict = {}
        patch_dirs_dict = {}
        info_dirs_dict = {}
        accepted_patch_mask_dict = {}

        for level in crop_level_list:
            for size in crop_size_list:
                size = int(size * 2**(crop_level - level))
                slide_dir = os.path.join(out_dir, f"level_{level}_size_{size}", croppable.slide_id)
                slide_dirs_dict[(level,size)] = slide_dir
                patch_dirs_dict[(level,size)] = os.path.join(slide_dir, "patches")
                info_dirs_dict[(level,size)] = (os.path.join(slide_dir, "patch_info.txt"), os.path.join(slide_dir, "general_info.txt"))
                accepted_patch_mask_dict[(level,size)] = np.ones(mask_im.shape)*255
                os.makedirs(patch_dirs_dict[(level,size)], exist_ok = True)
                with open(info_dirs_dict[(level,size)][0], "w") as f:
                    f.write("patch_num,label,tissue_ration,bg_ratio,base_coords,mask_coords,mask_lvl\n")
                    
        # Numbers to keep track
        num_patches = 0
        num_tissue_patches = {key:0 for key in accepted_patch_mask_dict.keys()}
        print("Slide: ", croppable.slide_id)
        print("Mpp: ", croppable.mpp)
        prog_bar = tqdm(total = CL_Hrange * CL_Wrange, desc = "Cropping patches")
        # Crop and save
        for row in range(CL_Hrange): #height
            for col in range(CL_Wrange): #width
                    
                base_coords = (base_x + col*base_stride,base_y + row*base_stride)
                patch = slide.read_region(base_coords,read_level,(base_cropsize,base_cropsize))
                
                patch = np.array(patch)[:,:,:3]
                if row == CL_Hrange - 1:
                    patch = pad_crop_bottom(patch, base_cropsize, remaining_base_height, pad_with = "white")
                if col == CL_Wrange - 1:
                    patch = pad_crop_right(patch, base_cropsize, remaining_base_width, pad_with = "white")
                    
                    
                #do mask calculation and patch division here using the function you define above
                mask_x_start, mask_x_end = mask_x + col*mask_stride, mask_x + col*mask_stride + mask_cropsize
                mask_y_start, mask_y_end = mask_y + row*mask_stride, mask_y + row*mask_stride + mask_cropsize
                patch_mask = mask_im[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                
                for wanted_crop_level in crop_level_list:
                    for wanted_size in crop_size_list:
                        wanted_size = int(wanted_size * 2**(crop_level - wanted_crop_level))
                        patches, patch_infos, accepted_mask, num_tissue = patch_n_mask_division(patch, num_patches, base_coords, (mask_x_start, mask_y_start), patch_mask, level_0_size, wanted_crop_level, wanted_size)
                        for i in range(len(patches)):
                            patch_temp = patches[i]
                            patch_temp_dir = os.path.join(patch_dirs_dict[(wanted_crop_level,wanted_size)], f"{patch_infos[i][0]}_{patch_infos[i][1]}.jpeg")
                            patch_temp.save(patch_temp_dir)
                            with open(info_dirs_dict[(wanted_crop_level,wanted_size)][0], "a") as f: #[0] = patch_info, [1] = general_info
                                        #patch_num, lbl, tissue_ratio, base_level_coords(x,y), mask_coords(x,y), created_mask_level
                                f.write(f"{patch_infos[i][0]},{patch_infos[i][1]},{patch_infos[i][2]},{patch_infos[i][3]},{patch_infos[i][4]},{patch_infos[i][5]},{mask_lvl}\n")
                        
                        accepted_patch_mask_dict[(wanted_crop_level,wanted_size)][mask_y_start:mask_y_end, mask_x_start:mask_x_end] = accepted_mask
                        num_tissue_patches[(wanted_crop_level,wanted_size)] += num_tissue
                
                prog_bar.update(1)
                num_patches += 1
                
        for key in accepted_patch_mask_dict: #one of the dicts is enough because they all have the same keys
            accepted_patch_mask_dict[key] = accepted_patch_mask_dict[key][0:mask_im.shape[0], 0:mask_im.shape[1]]
            accepted_patch_mask_dict[key] = accepted_patch_mask_dict[key].astype(np.uint8)
            imageio.imwrite(os.path.join(slide_dirs_dict[key], "accepted_mask.PNG"), accepted_patch_mask_dict[key])                
            with open(info_dirs_dict[key][1], "w") as f: #[0] = patch_info, [1] = general_info
                f.write(f"slide_path: {croppable.slide_path}\n")
                f.write(f"mask_path: {croppable.mask_path}\n")
                f.write(f"slide_id: {croppable.slide_id}\n")
                f.write(f"source: {croppable.source}\n")
                f.write(f"mpp: {croppable.mpp}\n")
                f.write(f"num_patches: {num_patches}\n")
                f.write(f"num_tissue_patches: {num_tissue_patches[key]}\n")
                f.write(f"tissue_ratio: {num_tissue_patches[key]/num_patches*((level_0_size/((2**key[0]) * key[1]))**2)}\n")
                
    except Exception as e:
        raise RuntimeError("Error occured at: ", croppable.slide_id) from e


def generate_allpatches(input_dir, level_size_index = -3, save_index = -3):
    
    for root, dirs, files in os.walk(input_dir):
        
        if "patch_info.txt" in files:
            #Burak_code/outputs_test/Cropped_slides/TCGA/test/level_4_size_128/6a2dd5e8-75f8-4fc6-be5b-184443740f36/TCGA-AQ-A54N-01Z-00-DX1\patch_info.txt
            new_root = root.replace("\\","/")
            
            patient_id = new_root.split("/")[save_index + 1]
            slide_id = new_root.split("/")[save_index + 2]
            level = new_root.split("/")[level_size_index].split("_")[1]
            size = new_root.split("/")[level_size_index].split("_")[3]
            txt_path = os.path.join(new_root, "patch_info.txt")
            
            allpatches_dir = "/".join(new_root.split("/")[:save_index])
            with open(allpatches_dir + f"/allpatches__level_{level}_size_{size}.txt", "w") as f:
                f.write("patient_id,slide_id,patch_id,patch_label,patch_path,tissue_ratio,bg_ratio,base_coords,mask_coords,mask_lvl\n")
            
            with open(allpatches_dir + f"/allpatches__level_{level}_size_{size}.txt", "a") as allpatches_f:
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                    #skip the header
                    # txt format: patch_num, lbl, tissue_ratio, bg_ratio, base_level_coords(x,y), mask_coords(x,y), mask_level
                    for line in lines[1:]:
                        line = line.strip("\n").strip()
                        patch_id = line.split(",")[0]
                        patch_label = line.split(",")[1]
                        tissue_ratio = float(line.split(",")[2])
                        bg_ratio = float(line.split(",")[3])
                        patch_path = os.path.join(new_root,"patches" ,patch_id + "_" + patch_label + ".jpeg")
                        base_x = line.split(",")[4].strip("(").strip(")").strip()
                        base_y = line.split(",")[5].strip("(").strip(")").strip()
                        mask_x = line.split(",")[6].strip("(").strip(")").strip()
                        mask_y = line.split(",")[7].strip("(").strip(")").strip()
                        mask_level = line.split(",")[8]
                        allpatches_f.write(f"{patient_id},{slide_id},{patch_id},{patch_label},{patch_path},{tissue_ratio},{bg_ratio},({base_x}-{base_y}),({mask_x}-{mask_y}),{mask_level}\n")
            
                
def crop_for_predict(img, cropsize: int, stride = None):
    """
    Takes an RGB slide image at the specified croplevel and crops it into a list of images of size cropsize.
    
    returns a list of cropped image arrays and their coordinates in the original image as a tuple.(top left corner)
    (x, y) = (width, height)
    """
        
    if cropsize == 0:
        print("Cropsize can't be 0")
    if stride == 0:
        print("Stride can't be 0")
    
    img = np.array(img)
    cropslist = []
    coordinates_list = []
    height, width, _ = img.shape
    cropsize = min(cropsize, height, width)
    if stride == None:
        stride = cropsize
    stride = min(stride, height, width)
        
    n_crops_height = int(np.ceil(height/stride)) # +0.4 to make sure we get the last crop
    n_crops_width = int(np.ceil(width/stride))# if the remaining pixels are less than stride
    
    for row in range(n_crops_height): # we start cropping from one cropsize above the image
        for col in range(n_crops_width): # we start cropping from one cropsize right of the image
            patch = img[row*stride:min(row*stride + cropsize, height), col*stride:min(col*stride + cropsize, width), :]
            padded_patch = np.pad(patch, ((0, cropsize - patch.shape[0]), (0, cropsize - patch.shape[1]), (0, 0)), mode='constant', constant_values=255 ) #may cause multiprocessing problems.
            cropslist.append(padded_patch)
            coordinates_list.append(((col*stride), (row*stride))) # top left corner coordinates
    
    del img, patch, padded_patch
    return cropslist , coordinates_list

def crop_for_predict_padded(img, cropsize: int, stride = None):
    """
    Takes an RGB slide image at the specified croplevel and crops it into a list of images of size cropsize.
    
    returns a list of cropped image arrays and their coordinates in the original image as a tuple.(top left corner)
    (x, y) = (width, height)
    """
        
    if cropsize == 0:
        print("Cropsize can't be 0")
    if stride == 0:
        print("Stride can't be 0")
    
    img = np.array(img)
    cropslist = []
    coordinates_list = []
    height, width, _ = img.shape
    cropsize = min(cropsize, height, width)
    if stride == None:
        stride = cropsize
    stride = min(stride, height, width)
        
    n_crops_height = int(np.ceil(height/stride)) # +0.4 to make sure we get the last crop
    n_crops_width = int(np.ceil(width/stride))# if the remaining pixels are less than stride
    img = np.pad(img, ((cropsize,cropsize), (cropsize,cropsize), (0, 0)), mode='constant', constant_values=255 ) #may cause multiprocessing problems.
    
    for row in range(n_crops_height + 2*(cropsize//stride)): # we start cropping from one cropsize above the image
        for col in range(n_crops_width + 2*(cropsize//stride)): # we start cropping from one cropsize right of the image
            patch = img[row*stride:min(row*stride + cropsize, height + cropsize), col*stride:min(col*stride + cropsize, width + cropsize), :]
            padded_patch = np.pad(patch, ((0, cropsize - patch.shape[0]), (0, cropsize - patch.shape[1]), (0, 0)), mode='constant', constant_values=255 ) #may cause multiprocessing problems.
            cropslist.append(padded_patch)
            coordinates_list.append(((col*stride - cropsize), (row*stride - cropsize))) # top left corner coordinates
    
    del img, patch, padded_patch
    return cropslist , coordinates_list

def read_slide_to_level(slide, rlenght) -> tuple[Image.Image, int]:
    """
    returns an image at the specified level from an openslide object.
    """
    try:
        current_res = float(slide.properties.get('openslide.mpp-x'))
    except:
        try:
            res_type = slide.properties.get("tiff.ResolutionUnit")
            if res_type == "centimeter":
                numerator = 10000
            elif res_type == "inch":
                numerator = 25400
            current_res = numerator / float(slide.properties.get("tiff.XResolution"))
        except:
            raise Exception('Unknown Val_x')
        
    if current_res < 0.3:  # resolution:0.25um/pixel
        current_res = 0.25
    elif current_res < 0.6:  # resolution:0.5um/pixel
        current_res = 0.5

    for i in range(0, len(slide.level_downsamples)):
        downsample = np.round(slide.level_downsamples[i])
        ratio = (current_res * downsample)/ rlenght
        #print("Ratio: ",ratio)
        if ratio <= 1:
            openslide_downsample = downsample
            openslide_level = i
            best_ratio = ratio
            
            
    slide_im = slide.read_region((0, 0), openslide_level, slide.level_dimensions[openslide_level])
    #print("imsize: ",slide_im.size)
    slide_im = np.array(slide_im)[:,:,:3]
    slide_im = Image.fromarray(slide_im)
    slide_im = slide_im.resize((int(slide_im.size[0] * best_ratio), int(slide_im.size[1] * best_ratio)))
    return slide_im, openslide_downsample/best_ratio





def GoRforgiveme(croppable: Croppable, crop_level_list: list, crop_size_list: list, out_dir):
    
    def patch_n_mask_division(patch_im, patch_num:int, patch_coords, mask_coords:tuple, mask_arr, level_0_size:int, wanted_level:int, wanted_size:int):
        """
        returns list of patches, patches_info, accepted_mask_patch, num_tissue_patches
        
        patch_im: patch image
        patch_num: number of the original patch
        patch_coords: base level coordinates(x/width, y/height) of the patch (top left corner)
        mask_coords: mask coordinates(x/width, y/height) of the patch (top left corner)
        mask_arr: mask np.array
        level_0_size: size of the patch at level 0 (2**croplevel x cropsize)
        wanted_level: level of the wanted patch
        wanted_size: size of the wanted patch
        """
        
        wanted_0_size = (2**wanted_level) * wanted_size
        
        if level_0_size < wanted_0_size:
            raise ValueError("Cannot crop to a level/size larger than the given level/size.")
        
        original_to_small = level_0_size / wanted_0_size
        if not original_to_small % 1 == 0:
            raise ValueError("level_0_size must be a multiple of wanted_0_size")
        
        original_to_small = int(original_to_small)
        patch_arr = np.array(patch_im)[:,:,:3]
        
        small_patch_size = patch_arr.shape[0] / original_to_small
        small_mask_size = mask_arr.shape[0] / original_to_small
        accepted_mask = np.ones(mask_arr.shape)*255
        
        patches = []
        patch_infos = []
        num_patch = original_to_small**2 * patch_num
        num_tissue = 0
        for row in range(original_to_small):
            for col in range(original_to_small):
                patch = patch_arr[int(row*small_patch_size):int((row+1)*small_patch_size),int(col*small_patch_size):int((col+1)*small_patch_size),:]
                patch = Image.fromarray(patch)
                patch = patch.resize((wanted_size, wanted_size), Image.LANCZOS)
                patches.append(patch)

                mask_patch = mask_arr[int(row*small_mask_size):int((row+1)*small_mask_size),int(col*small_mask_size):int((col+1)*small_mask_size)]
                tissue_ratio = np.sum(mask_patch) / (small_mask_size**2)
                if tissue_ratio > 0.1:
                    accepted_mask[int(row*small_mask_size):int((row+1)*small_mask_size),int(col*small_mask_size):int((col+1)*small_mask_size)] = 0
                    lbl = "tissue"
                    num_tissue += 1
                else:
                    lbl = "background"
                #                   patch_num, lbl, tissue_ratio, bg_ratio, base_level_coords(x,y), mask_coords(x,y), mask_level
                patch_infos.append((num_patch,
                                    lbl,
                                    round(tissue_ratio, 3),
                                    round(1- tissue_ratio, 3),
                                    (int(patch_coords[0] + col*small_patch_size), int(patch_coords[1] + row*small_patch_size)),
                                    (int(mask_coords[0] + col*small_mask_size), int(mask_coords[1] + row*small_mask_size))
                                    ))
                num_patch += 1

        return patches, patch_infos, accepted_mask, num_tissue
    
    
    
    
    
    
    
    
    
    try:
        slide = croppable.Load_Slide()

        crop_level = max(crop_level_list)
        crop_size = max(crop_size_list)
        level_0_size = (2**crop_level) * crop_size
        

        # Current Level Variables
        (_, _, CL_w, CL_h) = croppable.calculate_roi_for_level(crop_level)
        CL_Wrange, CL_Hrange = calculate_ranges((crop_size,crop_size),(CL_w,CL_h))

        # Mask Variables
        mask_lvl = croppable.mask_level
        mask_im = np.array(Image.open(croppable.mask_path))[:,:]/255
        mask_im = mask_im.astype(bool)
        mask_im = np.invert(mask_im) #False = bg, True = tissue
        (mask_x, mask_y, _, _) = croppable.calculate_roi_for_level(croppable.mask_level)
        (mask_cropsize, mask_stride) = croppable.calculate_mask_crops(crop_level,crop_size)
        
        if mask_cropsize / 2**(min(crop_size_list) - crop_size) < 1:
            raise ValueError("Some levels are too small for the mask.")
        
        # Base Level Variables
        slide_base_level = croppable.slide_base_level
        (base_x, base_y, base_width, base_height) = croppable.calculate_roi_for_level(slide_base_level)
        (base_cropsize, base_stride), read_level = croppable.calculate_baselevel_crops(crop_level,crop_size)
        base_Wrange, base_Hrange = calculate_ranges((base_cropsize,base_cropsize),(base_width,base_height))
        remaining_base_width = (base_cropsize * base_Wrange) - base_width
        remaining_base_height = (base_cropsize * base_Hrange) - base_height


        # Directory Variables
        out_dir = os.path.join(out_dir,"Cropped_slides", croppable.source)
        slide_dirs_dict = {}
        patch_dirs_dict = {}
        info_dirs_dict = {}
        accepted_patch_mask_dict = {}

        for level in crop_level_list:
            for size in crop_size_list:
                size = int(size * 2**(crop_level - level))
                slide_dir = os.path.join(out_dir, f"level_{level}_size_{size}", croppable.slide_id)
                slide_dirs_dict[(level,size)] = slide_dir
                patch_dirs_dict[(level,size)] = os.path.join(slide_dir, "patches")
                info_dirs_dict[(level,size)] = (os.path.join(slide_dir, "patch_info.txt"), os.path.join(slide_dir, "general_info.txt"))
                accepted_patch_mask_dict[(level,size)] = np.ones(mask_im.shape)*255
                os.makedirs(patch_dirs_dict[(level,size)], exist_ok = True)
                with open(info_dirs_dict[(level,size)][0], "w") as f:
                    f.write("patch_num,label,tissue_ration,bg_ratio,base_coords,mask_coords,mask_lvl\n")
                    
        # Numbers to keep track
        num_patches = 0
        num_tissue_patches = {key:0 for key in accepted_patch_mask_dict.keys()}
        print("Slide: ", croppable.slide_id)
        print("Mpp: ", croppable.mpp)
        prog_bar = tqdm(total = CL_Hrange * CL_Wrange, desc = "Cropping patches")
        # Crop and save
        for row in range(CL_Hrange): #height
            for col in range(CL_Wrange): #width
                    
                base_coords = (base_x + col*base_stride,base_y + row*base_stride)
                patch = slide.read_region(base_coords,read_level,(base_cropsize,base_cropsize))
                
                patch = np.array(patch)[:,:,:3]
                if row == CL_Hrange - 1:
                    patch = pad_crop_bottom(patch, base_cropsize, remaining_base_height, pad_with = "white")
                if col == CL_Wrange - 1:
                    patch = pad_crop_right(patch, base_cropsize, remaining_base_width, pad_with = "white")
                    
                    
                #do mask calculation and patch division here using the function you define above
                mask_x_start, mask_x_end = mask_x + col*mask_stride, mask_x + col*mask_stride + mask_cropsize
                mask_y_start, mask_y_end = mask_y + row*mask_stride, mask_y + row*mask_stride + mask_cropsize
                patch_mask = mask_im[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                
                for wanted_crop_level in crop_level_list:
                    for wanted_size in crop_size_list:
                        wanted_size = int(wanted_size * 2**(crop_level - wanted_crop_level))
                        patches, patch_infos, accepted_mask, num_tissue = patch_n_mask_division(patch, num_patches, base_coords, (mask_x_start, mask_y_start), patch_mask, level_0_size, wanted_crop_level, wanted_size)
                        for i in range(len(patches)):
                            patch_temp = patches[i]
                            patch_temp_dir = os.path.join(patch_dirs_dict[(wanted_crop_level,wanted_size)], f"{patch_infos[i][0]}_{patch_infos[i][1]}.jpeg")
                            patch_temp.save(patch_temp_dir)
                            with open(info_dirs_dict[(wanted_crop_level,wanted_size)][0], "a") as f: #[0] = patch_info, [1] = general_info
                                        #patch_num, lbl, tissue_ratio, base_level_coords(x,y), mask_coords(x,y), created_mask_level
                                f.write(f"{patch_infos[i][0]},{patch_infos[i][1]},{patch_infos[i][2]},{patch_infos[i][3]},{patch_infos[i][4]},{patch_infos[i][5]},{mask_lvl}\n")
                        
                        accepted_patch_mask_dict[(wanted_crop_level,wanted_size)][mask_y_start:mask_y_end, mask_x_start:mask_x_end] = accepted_mask
                        num_tissue_patches[(wanted_crop_level,wanted_size)] += num_tissue
                
                prog_bar.update(1)
                num_patches += 1
                
        for key in accepted_patch_mask_dict: #one of the dicts is enough because they all have the same keys
            accepted_patch_mask_dict[key] = accepted_patch_mask_dict[key][0:mask_im.shape[0], 0:mask_im.shape[1]]
            accepted_patch_mask_dict[key] = accepted_patch_mask_dict[key].astype(np.uint8)
            imageio.imwrite(os.path.join(slide_dirs_dict[key], "accepted_mask.PNG"), accepted_patch_mask_dict[key])                
            with open(info_dirs_dict[key][1], "w") as f: #[0] = patch_info, [1] = general_info
                f.write(f"slide_path: {croppable.slide_path}\n")
                f.write(f"mask_path: {croppable.mask_path}\n")
                f.write(f"slide_id: {croppable.slide_id}\n")
                f.write(f"source: {croppable.source}\n")
                f.write(f"mpp: {croppable.mpp}\n")
                f.write(f"num_patches: {num_patches}\n")
                f.write(f"num_tissue_patches: {num_tissue_patches[key]}\n")
                f.write(f"tissue_ratio: {num_tissue_patches[key]/num_patches*((level_0_size/((2**key[0]) * key[1]))**2)}\n")
                
    except Exception as e:
        raise RuntimeError("Error occured at: ", croppable.slide_id) from e