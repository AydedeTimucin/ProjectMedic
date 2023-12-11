import os
from PIL import Image
from lxml import etree
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm
import CroppingTools as ct
from shapely.geometry import Point, Polygon

def is_inside_polygon(pixel_coord, outer_polygon_coordinates, inner_polygon_coordinates=[]):
    point = Point(pixel_coord)
    outer_polygon = Polygon(outer_polygon_coordinates)
    
    if inner_polygon_coordinates:
        inner_polygon = Polygon(inner_polygon_coordinates)
        return point.within(outer_polygon) and not point.within(inner_polygon)
    else:
        return point.within(outer_polygon)





slide_path = r"D:\AIN3007_project\train_test_splited\test_data\Breast1__he\slides\TCGA-A2-A3XV-01Z-00-DX1.svs"
xml_path = r"D:\AIN3007_project\train_test_splited\test_data\Breast1__he\slides\TCGA-A2-A3XV-01Z-00-DX1.xml"

out_dir = r"D:\AIN3007_project\patches"

slide_name = os.path.basename(slide_path).split('.')[0]
source = "Breast1__he"


output_dir = os.path.join(out_dir, source, slide_name)
tissue_dir = os.path.join(output_dir, "tissues")
mask_dir = os.path.join(output_dir, "masks")
os.makedirs(tissue_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)


stride = 512
tcga_level = 2


slide = openslide.OpenSlide(slide_path)
print("Slide dimension 2: ", slide.level_dimensions[tcga_level])
downscale = np.round(slide.level_downsamples[tcga_level])

slide_width, slide_height = slide.level_dimensions[tcga_level]

roi_information = [0, 0, slide_width, slide_height] # [x, y, width, height]
child_vertices = None

tree = etree.parse(xml_path)
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
            
    elif part_of_group == "roi":
        roi_coordinates = []
        for co in ann.iter():
            if co.tag == 'Coordinate':
                X = max(int(float(co.attrib.get('X').replace(',', '.')) / downscale), 0)
                Y = max(int(float(co.attrib.get('Y').replace(',', '.')) / downscale), 0)
                roi_coordinates.append([X, Y])
                
        top_left = roi_coordinates[0]
        bottom_right = roi_coordinates[2]
        x1, y1 = top_left
        x2, y2 = bottom_right
        width, height = x2 - x1, y2 - y1

        roi_information = [x1, y1, min(slide_width-x1, width), min(slide_height-y1,height)] # [x, y, width, height]

print("ROI: ", roi_information)
#print("Vertices: ", vertices)
if child_vertices is not None:
    print("Child vertices: ", child_vertices)




print("Slide width: ", slide_width, "Slide height: ", slide_height)

wrange, hrange = ct.calculate_ranges((stride,stride), (slide_width, slide_height))
print("Ranges: ", wrange, hrange)


patch_num = 0
pbar = tqdm(total=wrange*hrange)
for y in range(hrange):
    y_base = y*stride
    y_zero = int(y_base * downscale)
    for x in range(wrange):
        mask_patch = np.zeros((stride, stride), dtype=np.uint8)
        
        x_base = x*stride
        x_zero = int(x_base * downscale)
        
        
        img = slide.read_region((x_zero, y_zero), tcga_level, (stride, stride))
        img.save(os.path.join(tissue_dir, f"{patch_num}.png"))
        
        for yy in range(stride):
            for xx in range(stride):
                xx_for_ann = xx + x_base
                yy_for_ann = yy + y_base
                if is_inside_polygon((xx_for_ann,yy_for_ann), vertices, child_vertices):
                    mask_patch[yy, xx] = 255
        Image.fromarray(mask_patch).save(os.path.join(mask_dir, f"{patch_num}.png"))
        patch_num += 1
        pbar.update(1)
        
    

    
    
    