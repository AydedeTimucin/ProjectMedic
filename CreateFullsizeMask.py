import os
from lxml import etree
import openslide
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import argparse



def get_slidelist(folder_dir):
    """
    returns a list of:
    [slide_path, xml_file_path, source, slide_id]
    
    """
    
    slidelist = []
    for dir in os.listdir(folder_dir):
        for file in os.listdir(os.path.join(folder_dir, dir, "slides")):
            if file.startswith("._"):
                continue
            if file.endswith(".svs"):
                pass
            elif file.endswith(".tif"):
                pass
            elif file.endswith(".mrxs"):
                pass
            elif file.endswith(".ndpi"):
                pass
            else:
                continue
            
            slide_path = os.path.join(folder_dir, dir, "slides", file)
            slide_path = slide_path.replace("\\", "/")
            xml_file_path = os.path.join(folder_dir, dir, "slides", ".".join(file.split(".")[0:-1]) + ".xml")
            xml_file_path = xml_file_path.replace("\\", "/")
                                #slide_path, xml_file_path, source, patient_id, slide_id
            slidelist.append([slide_path, xml_file_path, dir, ".".join(file.split(".")[0:-1])])
            
    return slidelist
            





def create_mask(args:list, wanted_rlength, create_rgb_image=False, additonal_info=False):
    """
    args: [slide_path, xml_file_path, source, slide_id]
    wanted_rlength: The wanted resolution of the mask in microns per pixel.
    
    
    
    Returns mask_image, mask_info_list.
    mask_info_list contains the following information:
    [slide_id, source, slide_path, wanted_rlength, roi_str]
    
    
    if create_rgb_image == True, it also returns the rgb image of the slide.
    
    if additonal_info == True, mask_info contains additional info.
    [Mask_area_px, im_area_px, Mask_area_ratio]
    
    
    """
    
    slide_path, xml_file_path, source, slide_id = args
    
    
    slide = openslide.OpenSlide(slide_path)
    tree = etree.parse(xml_file_path)
    
    
    # Get the dimensions of the slide at the wanted resolution (width, height)
    mask_width, mask_height = slide.level_dimensions[0][0], slide.level_dimensions[0][1]
    
    # Region of interest information for getting rid of unnecessary parts of the slide
    roi_information = [0, 0, mask_width, mask_height]
    
    # Create the mask
    tissue_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    
    annotations = tree.findall('.//Annotation')
    for ann in annotations:
        part_of_group = ann.attrib.get('PartOfGroup')
        if part_of_group == "tissue":
            coordinates = []
            for co in ann.iter():
                if co.tag == 'Coordinate':
                    X = int(float(co.attrib.get('X').replace(',', '.')))
                    Y = int(float(co.attrib.get('Y').replace(',', '.')))
                    coordinates.append([X, Y])

            vertices = np.array(coordinates, dtype=np.int32)
            cv2.fillPoly(tissue_mask, [vertices], color=255)

        elif part_of_group == "bg":
            parent_annotation = ann.getparent()
            for child_ann in parent_annotation.findall('Annotation'):
                if child_ann.attrib.get('PartOfGroup') != "bg":
                    continue

                child_coordinates = []
                for co in child_ann.iter():
                    if co.tag == 'Coordinate':
                        X = int(float(co.attrib.get('X').replace(',', '.')))
                        Y = int(float(co.attrib.get('Y').replace(',', '.')))
                        child_coordinates.append([X, Y])

                child_vertices = np.array(child_coordinates, dtype=np.int32)
                cv2.fillPoly(tissue_mask, [child_vertices], color=0)
                
        elif part_of_group == "roi":
            roi_coordinates = []
            for co in ann.iter():
                if co.tag == 'Coordinate':
                    X = max(int(float(co.attrib.get('X').replace(',', '.'))), 0)
                    Y = max(int(float(co.attrib.get('Y').replace(',', '.'))), 0)
                    roi_coordinates.append([X, Y])
                    
            top_left = roi_coordinates[0]
            bottom_right = roi_coordinates[2]
            x1, y1 = top_left
            x2, y2 = bottom_right
            width, height = x2 - x1, y2 - y1

            roi_information = [x1, y1, min(mask_width-x1, width), min(mask_height-y1,height)] # [x, y, width, height]
    roi_str = "-".join([str(roi_information[0]), str(roi_information[1]), str(roi_information[2]), str(roi_information[3])])
    
    
    # Create a mask info text
    if additonal_info:
        mask_info = [str(slide_id),str(source),str(slide_path),str(roi_str)]
    else:
        mask_info = [str(slide_id),str(source),str(slide_path),str(roi_str)]
    
    
    # After inverting the mask, the tissue will be white and the background will be black
    inverted_tissue_mask = cv2.bitwise_not(tissue_mask)
    inverted_tissue_mask = Image.fromarray(inverted_tissue_mask)
    
    
    
    if create_rgb_image:
        slide_rgb = slide.read_region((0, 0), 0, (mask_width, mask_height))
        slide_rgb = slide_rgb.resize((mask_width, mask_height), Image.Resampling.LANCZOS)
        slide_rgb = np.array(slide_rgb.convert("RGB"))
        rgb_mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        
        
        rgb_mask[tissue_mask == 255] = slide_rgb[tissue_mask == 255]

        rgb_mask = Image.fromarray(rgb_mask)
        
        slide.close()
        
        return inverted_tissue_mask, rgb_mask, mask_info
    else:
        slide.close()
        return inverted_tissue_mask, mask_info
    
    
    

    
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="D:/AIN3007_project/train_test_splited/test_data", help='The Directory that contains the "slides" folder.(test_data or train_data etc.)')
parser.add_argument('--output_dir', type=str, default='D:/AIN3007_project/outputs_test', help='The directory of the output folder that will contain the masks.')
parser.add_argument('--wanted_rlength', type=float, default=8, help='The wanted resolution of the mask in microns per pixel.')

FLAGS = parser.parse_args()

folder_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
wanted_rlength = FLAGS.wanted_rlength


additional_info = False
create_rgb_image = False




slidelist = get_slidelist(folder_dir)

tissue_mask_dir = os.path.join(output_dir, "tissue_masks")

mask_info_df = pd.DataFrame(columns=["slide_id", "source", "slide_path", "roi_str"])
additional_info_df = pd.DataFrame(columns=["slide_id","source"])

for args in tqdm(slidelist):
    
    slide_id = args[3]
    source = args[2]
    
    source_dir = os.path.join(tissue_mask_dir, source)
    
    # Save the mask images
    if create_rgb_image:
        mask_image, rgb_mask, mask_info = create_mask(args, wanted_rlength, create_rgb_image=True, additonal_info=additional_info)
        
        binary_mask_dir = os.path.join(source_dir, "binary_full_masks")
        rgb_mask_dir = os.path.join(source_dir, "rgb_masks")
        os.makedirs(binary_mask_dir, exist_ok=True)
        os.makedirs(rgb_mask_dir, exist_ok=True)
        
        mask_image.save(os.path.join(binary_mask_dir, slide_id + ".png"))
        rgb_mask.save(os.path.join(rgb_mask_dir, slide_id + ".png"))
    
    
    else:
        mask_image, mask_info = create_mask(args, wanted_rlength, create_rgb_image=False, additonal_info=additional_info)
        
        binary_mask_dir = os.path.join(source_dir, "binary_full_masks")
        os.makedirs(binary_mask_dir, exist_ok=True)

        
        mask_image.save(os.path.join(binary_mask_dir, slide_id + ".png"))



    # Save the mask info
    mask_info_df.loc[len(mask_info_df)] = mask_info[0:4]
    mask_info_df.to_csv(os.path.join(output_dir, "total_mask_info.csv"), index=False)
    
    if additional_info:
        additional_info_df.loc[len(additional_info_df)] = mask_info[0:2] 

    break
    
print("Saving...")
mask_info_df = mask_info_df.groupby("source", as_index=True)
for source, group in mask_info_df:
    group.to_csv(os.path.join(tissue_mask_dir, source, "mask_info.csv"), index=False)
additional_info_df = additional_info_df.groupby("source", as_index=True)
for source, group in additional_info_df:
    group.to_csv(os.path.join(tissue_mask_dir, source, "additional_info.csv"), index=False)