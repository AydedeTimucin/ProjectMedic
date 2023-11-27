import os
os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\MedicalAI\Lib\site-packages\openslide-win64-20231011\bin')

from lxml import etree
import openslide
import PIL
import numpy as np
import cv2
import argparse
import math
import sys

def masking(slide_path, xml_path, output_dir):

    # Load the XML file
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Load the slide image
    slide = openslide.OpenSlide(slide_path)

    mask_level = 5
    downsample_factor = float(2 ** mask_level)
    target_downsample = math.log(int(round(slide.level_downsamples[-1])), 2)

    # Set the resolution of the slide image
    try:
        val_x = float(slide.properties.get('openslide.mpp-x'))
    except:
        try:
            res_type = slide.properties.get("tiff.ResolutionUnit")
            if res_type == "centimeter":
                numerator = 10000
            elif res_type == "inch":
                numerator = 25400
            val_x = numerator / float(slide.properties.get("tiff.XResolution"))
        except:
            print('Unknown Val_x')
            return

    current_res = 1
    if val_x < 0.3:  # resolution:0.25um/pixel
        current_res = 0.25
    elif val_x < 0.6:  # resolution:0.5um/pixel
        current_res = 0.5

    im_read_level = slide.get_best_level_for_downsample(downsample_factor)
    im_read_size = slide.level_dimensions[im_read_level]

    target_res = 2 ** target_downsample
    real_length = target_res * current_res  # real length in micron

    # Get the downsampled image as a NumPy array
    down_sampled_img_array = np.array(slide.read_region((0, 0), im_read_level, im_read_size))[:, :, :3]

    # resize image to level6
    im_size_level0 = slide.level_dimensions[0]
    im_resized_size = (int(im_size_level0[0] / downsample_factor), int(im_size_level0[1] / downsample_factor))
    resized_img_arr = cv2.resize(down_sampled_img_array, im_resized_size)

    # Convert the NumPy array to a PIL Image
    down_sampled_img_pil = PIL.Image.fromarray(resized_img_arr)

    # Initialize the tissue mask
    mask_width, mask_height = slide.level_dimensions[-1]
    tissue_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Initialize the RGB mask
    # rgb_mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)

    roi_information = [0, 0, mask_width, mask_height]  # [x, y, height, width]

    # Loop over all Annotation tags in the XML file
    annotations = tree.findall('.//Annotation')
    for ann in annotations:
        part_of_group = ann.attrib.get('PartOfGroup')
        if part_of_group == "tissue":
            coordinates = []
            for co in ann.iter():
                if co.tag == 'Coordinate':
                    X = int(float(co.attrib.get('X').replace(',', '.')) / target_res)
                    Y = int(float(co.attrib.get('Y').replace(',', '.')) / target_res)
                    coordinates.append([X, Y])

            vertices = np.array(coordinates, dtype=np.int32)
            cv2.fillPoly(tissue_mask, [vertices], color=255)

        elif part_of_group == "bg":
            parent_annotation = ann.getparent()
            parent_coordinates = []
            for child_ann in parent_annotation.findall('Annotation'):
                if child_ann.attrib.get('PartOfGroup') != "bg":
                    continue

                child_coordinates = []
                for co in child_ann.iter():
                    if co.tag == 'Coordinate':
                        X = int(float(co.attrib.get('X').replace(',', '.')) / target_res)
                        Y = int(float(co.attrib.get('Y').replace(',', '.')) / target_res)
                        child_coordinates.append([X, Y])

                child_vertices = np.array(child_coordinates, dtype=np.int32)
                cv2.fillPoly(tissue_mask, [child_vertices], color=0)

        elif part_of_group == "roi":
            roi_coordinates = []
            for co in ann.iter():
                if co.tag == 'Coordinate':
                    X = max(int(float(co.attrib.get('X').replace(',', '.')) / target_res), 0)
                    Y = max(int(float(co.attrib.get('Y').replace(',', '.')) / target_res), 0)
                    roi_coordinates.append([X, Y])

            top_left = roi_coordinates[0]
            bottom_right = roi_coordinates[2]
            x1, y1 = top_left
            x2, y2 = bottom_right
            width, height = x2 - x1, y2 - y1

            roi_information = [x1, y1, min(mask_width, width), min(mask_height, height)]  # [x, y, width, height]
        roi_str = "-".join([str(roi_information[0]), str(roi_information[1]), str(roi_information[2]), str(roi_information[3])])

    # Invert the tissue mask
    inverted_tissue_mask = cv2.bitwise_not(tissue_mask)

    # Calculate the area of the tissue mask
    Mask_area_px = np.count_nonzero(tissue_mask)
    im_area_px = mask_width * mask_height
    Mask_area_ratio = Mask_area_px / im_area_px

    # Convert tissue mask to RGB mask using original RGB colors
    """
    slide_rgb = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1])
    slide_rgb = np.array(slide_rgb.convert("RGB"))
    rgb_mask[tissue_mask == 255] = slide_rgb[tissue_mask == 255]
    """

    mask_output_directory = os.path.join(output_dir, "Masks")
    tissue_output_directory = r"G:\down_scaled_images(level5)\Tissues"
    #crop_info_directory = os.path.join(output_dir, "Crops_info")

    mask_output_directory = os.path.join(mask_output_directory, f'binary-mask')
    #rgb_mask_output_dir = os.path.join(output_directory, f'{data_type}RGB-mask')

    slide_name = str(slide_path.split('\\')[-1])[:-4]

    # Create the output directories if they do not exist
    os.makedirs(mask_output_directory, exist_ok=True)
    os.makedirs(tissue_output_directory, exist_ok=True)
    #os.makedirs(rgb_mask_output_dir, exist_ok=True)

    # Save the down sampled image
    output_file_name = slide_name + '.png'
    down_sampled_image_path = os.path.join(tissue_output_directory, output_file_name)
    down_sampled_img_pil.save(down_sampled_image_path)

    # Save the tissue mask as a PNG file

    output_file_name = slide_name + '(mask).png'
    output_path_mask = os.path.join(mask_output_directory, output_file_name)

    mask_img = PIL.Image.fromarray(inverted_tissue_mask)
    mask_img.save(output_path_mask)

    # Save the RGB mask as a PNG file
    """
    output_path_rgb_mask = os.path.join(rgb_mask_output_dir, output_file_name)
    img_rgb = PIL.Image.fromarray(rgb_mask, 'RGB')
    img_rgb.save(output_path_rgb_mask)
    """

    print(f"Resolution for {slide_name}: {val_x}")
    print(f"Tissue area: {round(Mask_area_px * real_length * real_length / 1000000, 3)} mm^2")
    print(f"Total area: {round(im_area_px * real_length * real_length / 1000000, 3)} mm^2")
    print(f"Mask/Area percentage: %{round(Mask_area_ratio * 100, 3)}")
    print("-----------------------------------")
    print("Mask size: ", mask_img.size, " Memory Usage:", os.stat(output_path_mask).st_size / 1024, " (KB)")
    print("Tissue size: ", down_sampled_img_pil.size, " Memory Usage:", os.stat(down_sampled_image_path).st_size / 1024, " (KB)")

slide_path = r"G:\train_data\Breast1__he\slides\TCGA-A1-A0SM-01Z-00-DX1.svs"
xml_path = r"G:\train_data\Breast1__he\slides\TCGA-A1-A0SM-01Z-00-DX1.xml"
output_dir = r"G:\down_scaled_images(level5)"

masking(slide_path, xml_path, output_dir)
