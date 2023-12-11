import os
from lxml import etree
import openslide
import PIL
import numpy as np
import cv2
import argparse
import math


#Currently works with: TCGA, Camelyon17, HEROHE, HER2
#Data folder's names should be the names above

#to add: recursive slide folder seaching, Support for more file types, Support for more sources
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='Burak_code\Slides\TCGA_nolabel', help='The directory of the source folder that contains the slides.')
parser.add_argument('--output_dir', type=str, default='Burak_code\outputs_test', help='The directory of the output folder that will contain the masks.')
FLAGS = parser.parse_args()

folder_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir

slidelist = []
data_type = ''
print('started')
base_name = os.path.basename(folder_dir)
for root, dirs, files in os.walk(folder_dir):
    f1, f2 = False, False
    data_type = ''
    for file in files:
        if file.startswith('._'):
            continue
        f1, f2 = False, False
        if file.rsplit('.',1)[0] + '.xml' in os.listdir(root):
            f1 = True
        else:
            continue
        root = root.replace('\\','/')
        root_as_list = root.split('/')
        file_index = root_as_list.index(base_name)
        root_as_list = root_as_list[file_index:]
        for i in root_as_list:
            if i.lower() == "train" or i.lower() == "training":
                data_type = "train_"
            if i.lower() == "test" or i.lower() == "testing":
                data_type = "test_"
            if i.lower() == "val" or i.lower() == "validation" or i.lower() == 'validating':
                data_type = "val_"
                
        if file.endswith('.svs') and file.startswith('TCGA'):
            f2 = True
            source = 'TCGA'
            patient_id = root_as_list[-1]
            slide_id = file.split('.')[0]

        if file.endswith('.tif') and file.startswith('patient'):
            f2 = True
            source = 'Camelyon17'
            patient_id = '_'.join(file.split('.')[0].split('_')[:2]) 
            slide_id = '_'.join(file.split('.')[0].split('_')[2:4]) 
        
        if file.endswith('.mrxs'):
            f2 = True
            source = 'HEROHE'
            patient_id = file.split('.')[0]
            slide_id = file.split('.')[0]
            
        if file.endswith('HE.ndpi'):
            f2 = True
            source = 'HER2_HE'
            patient_id = file.split('.')[0].split('_')[0]
            slide_id = file.split('.')[0].split('_')[1]
            
        if file.endswith('Her2.ndpi'):
            f2 = True
            source = 'HER2_IHC'
            patient_id = file.split('.')[0].split('_')[0]
            slide_id = file.split('.')[0].split('_')[1]
        
        if f2 & f1:
            slide_path = os.path.join(root, file)
            xml_path = os.path.join(root, file.rsplit('.',1)[0] + '.xml')
            slidelist.append([slide_path, xml_path, source, patient_id, slide_id, data_type])
            # print(data_type)

data_types_list = ["", "train_", "test_", "val_"]
sources_list = ["TCGA", "Camelyon17", "HEROHE", "HER2_HE", "HER2_IHC"]
Data_stats_path_created_bool_dict = dict()
txt_file_path_created_bool_list = [False, False, False, False] # ["", "train", "test", "val"]
for source in sources_list:
    Data_stats_path_created_bool_dict[source] = [False, False, False, False]
    


for slide_path, xml_file_path, source, patient_id, slide_id, data_type in slidelist:

    # Load the XML file
    tree = etree.parse(xml_file_path)

    # Load the slide image
    slide = openslide.OpenSlide(slide_path)
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
            continue
        
    if val_x < 0.3:  # resolution:0.25um/pixel
        current_res = 0.25
    elif val_x < 0.6:  # resolution:0.5um/pixel
        current_res = 0.5

    im_size = slide.level_dimensions[-1]
    target_res = (2 ** (target_downsample))
    real_length = target_res * current_res  # real length in micron
    
    # Initialize the tissue mask and RGB mask
    mask_width, mask_height = im_size
    tissue_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    
    # Initialize the RGB mask
    rgb_mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)

    roi_information = [0, 0, mask_width, mask_height] # [x, y, height, width]
    
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

            roi_information = [x1, y1, min(mask_width, width), min(mask_height,height)] # [x, y, width, height]
    roi_str = "-".join([str(roi_information[0]), str(roi_information[1]), str(roi_information[2]), str(roi_information[3])])
    
    # Invert the tissue mask
    inverted_tissue_mask = cv2.bitwise_not(tissue_mask)
    
    # Calculate the area of the tissue mask
    Mask_area_px = np.count_nonzero(tissue_mask)
    im_area_px = mask_width * mask_height
    Mask_area_ratio = Mask_area_px / im_area_px
    
    # Convert tissue mask to RGB mask using original RGB colors
    slide_rgb = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1])
    slide_rgb = np.array(slide_rgb.convert("RGB"))
    rgb_mask[tissue_mask == 255] = slide_rgb[tissue_mask == 255]


    output_directory = os.path.join(output_dir, "Tissue_Masks", source + '_masks')
    crop_info_directory = os.path.join(output_dir, "Crops_info")
    
    mask_output_dir = os.path.join(output_directory, f'{data_type}binary-mask')
    rgb_mask_output_dir = os.path.join(output_directory, f'{data_type}RGB-mask')

    # Create the output directories if they do not exist
    os.makedirs(mask_output_dir, exist_ok=True)
    os.makedirs(rgb_mask_output_dir, exist_ok=True)
    os.makedirs(crop_info_directory, exist_ok=True)
    # Save the tissue mask as a PNG file
    if source == "TCGA":
        output_file_name = patient_id + '.png'
    else:
        output_file_name = patient_id + '_' + slide_id + '.png'
        
    output_path_mask = os.path.join(mask_output_dir, output_file_name)
    img = PIL.Image.fromarray(inverted_tissue_mask)
    img.save(output_path_mask)

    # Save the RGB mask as a PNG file
    output_path_rgb_mask = os.path.join(rgb_mask_output_dir, output_file_name)
    img_rgb = PIL.Image.fromarray(rgb_mask, 'RGB')
    img_rgb.save(output_path_rgb_mask)

    print(f"Resolution for {patient_id}: {val_x}")
    print(f"Tissue area: {round(Mask_area_px * real_length * real_length / 1000000,3)} mm^2")
    print(f"Total area: {round(im_area_px * real_length * real_length / 1000000,3)} mm^2")
    print(f"Mask/Area percentage: %{round(Mask_area_ratio * 100,3)}")
    print(f"Data type: {data_type}")
    print("-----------------------------------")

    txt_file_path = os.path.join(output_directory, data_type + 'tissue_mask_info' + '.txt')
    Data_stats_path = os.path.join(crop_info_directory, f"{source}_{data_type}Data_stats.txt")
    
    if txt_file_path_created_bool_list[data_types_list.index(data_type)] == False:
        with open(txt_file_path, 'w') as f:
            f.write("# path of mask, path of slide, Mask Image level(level_x = 2**x), Real Length in microns, Patient ID, Slide ID, Source Name, ROI(X-Y-Width-Height) \n")
        txt_file_path_created_bool_list[data_types_list.index(data_type)] = True
        
    if Data_stats_path_created_bool_dict[source][data_types_list.index(data_type)] == False:
        with open(Data_stats_path, 'w') as f:
            f.write("# Patient ID, Slide ID, Tissue/Total %, Tissue Area(mm), Bg/Total %, Bg Area(mm), Total Area(mm), Tissue-Size(Mb), Bg-Size(Mb), FileSize(Mb) \n")
        Data_stats_path_created_bool_dict[source][data_types_list.index(data_type)] = True
        
    # Write the pathname of the tissue mask image to the text file
    with open(txt_file_path, 'a') as f:
        f.write(output_path_mask + ',' + slide_path + ',' + str(target_downsample) + ',' + str(real_length) + ',' + patient_id + ',' + slide_id + ',' + source + ',' + roi_str + '\n')
    with open(Data_stats_path, 'a') as f:
        f.write(patient_id + ',' + slide_id + ',' + str(round(Mask_area_ratio * 100,2)) + ',' + str(round(Mask_area_px * real_length * real_length / 1000000,2)) + ',' + str(round((1 - Mask_area_ratio) * 100, 2)) + ',' + str(round((1 - Mask_area_ratio) * real_length * real_length / 1000000,2)) + ',' + str(round(im_area_px * real_length * real_length / 1000000,2)) + ',' + str(Mask_area_ratio * os.path.getsize(slide_path) // (2**20)) + ',' + str((1 - Mask_area_ratio) * os.path.getsize(slide_path) // (2**20)) + ',' + str(os.path.getsize(slide_path) // (2**20)) +'\n')
#path of mask, path of slide, Image level(level_x = 2**x), Real Lenght in microns,mask size, Patient ID, Slide ID
