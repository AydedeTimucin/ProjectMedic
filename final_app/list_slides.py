import os
import math
import time
import utils
import openslide
from lxml import etree
import numpy as np
import cv2

wsi_paths_file = r"D:\AIN3007_project\external_wsi_paths.txt"
output_folder = r"D:/AIN3007_project/external_test_GT_masks"
output_thumb_folder = r"D:/AIN3007_project/external_test_thumbs"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_thumb_folder, exist_ok=True)
with open(wsi_paths_file, "r") as f:
    wsi_paths = f.readlines()
    
for i in range(len(wsi_paths)):
    wsi_paths[i] = wsi_paths[i].strip()
    slide = openslide.OpenSlide(wsi_paths[i])
    xml_path = wsi_paths[i].rsplit(".", 1)[0] + ".xml"

    slide_name = os.path.splitext(wsi_paths[i])[0].split("\\")[-1].split("/")[-1]

    stride = 512

    level = utils.pick_best_level(slide=slide, stride=stride, max_patch_num=400, min_patch_num=20)

    level_dim = level - slide.level_count
    
    downscale = int(slide.level_downsamples[level_dim])
    
    slide_width, slide_height = slide.level_dimensions[level]

    mask_width = math.ceil(slide_width / stride) * stride
    mask_height = math.ceil(slide_height / stride) * stride
    
    
    tree = etree.parse(xml_path)
    
    print("level: ", level)
    print("downscale: ", downscale)
    print("level0: ", slide.level_dimensions[0])
    print("Current level: ", slide.level_dimensions[level_dim])
    
    
    
    tissue_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    
    annotations = tree.findall('.//Annotation')
    for ann in annotations:
        part_of_group = ann.attrib.get('PartOfGroup')
        if part_of_group == "tissue":
            coordinates = []
            for co in ann.iter():
                if co.tag == 'Coordinate':
                    X = int(float(co.attrib.get('X').replace(',', '.')) / downscale)
                    Y = int(float(co.attrib.get('Y').replace(',', '.')) / downscale)
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
                        X = int(float(co.attrib.get('X').replace(',', '.')) / downscale)
                        Y = int(float(co.attrib.get('Y').replace(',', '.')) / downscale)
                        child_coordinates.append([X, Y])

                child_vertices = np.array(child_coordinates, dtype=np.int32)
                cv2.fillPoly(tissue_mask, [child_vertices], color=0)
    tissue_mask = cv2.bitwise_not(tissue_mask)
    cv2.imwrite(f"{output_folder}/{slide_name}_GT.png", tissue_mask)
    
    
    img = slide.read_region((0, 0), level, (slide_width, slide_height))
    img = img.convert("RGB")
    
    padded_img = np.pad(np.array(img), ((0, mask_height - slide_height), (0, mask_width - slide_width), (0, 0)), constant_values=255)
    
    cv2.imwrite(f"{output_thumb_folder}/{slide_name}.png", padded_img)
    
    gray_im = cv2.cvtColor(padded_img, cv2.COLOR_RGB2GRAY)

    # background - foreground seperation using OTSU thresholding
    OTSU_thr, BW = cv2.threshold(gray_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(OTSU_thr)

    #apply_dilation
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.bitwise_not(cv2.dilate(cv2.bitwise_not(BW), kernel, iterations=1))

    # median filtering
    BW_filtered = cv2.medianBlur(img_dilation, 19)

    # invert image for further processing
    des = cv2.bitwise_not(BW_filtered)

    # hole filling
    contour, hier = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(des, [cnt], 0, 255, -1)
    mask_img = cv2.bitwise_not(des)
    cv2.imwrite(f"{output_folder}/{slide_name}_CV2.png", mask_img)
    