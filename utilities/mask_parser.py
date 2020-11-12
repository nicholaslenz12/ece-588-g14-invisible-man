import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import json

with open('new_mask_labels.json', 'r') as f:
    mask_labels = json.load(f)

mask_labels_pbar = tqdm(mask_labels)
for row in mask_labels_pbar:
    img_name = row['External ID']
    dataset_name = row['Dataset Name']
    if dataset_name == 'Liu_Bolin_Studio':
        mask_labels_pbar.set_description("Processing %s" % img_name)
        if 'objects' in row['Label'].keys():
            mask_coords_dicts = row['Label']['objects'][0]['polygon']
            original_img_color = cv2.imread('Images/Liu_Bolin_Studio/'+img_name)
            original_img = cv2.imread('Images/Liu_Bolin_Studio/'+img_name, 0)
            bg = np.zeros_like(original_img).astype(np.float32)
            mask_coords = []
            for mask_coords_dict in mask_coords_dicts:
                mask_coords.append([mask_coords_dict['x'], mask_coords_dict['y']])
            contour = [np.array(mask_coords, dtype=np.int32)]
            mask = cv2.fillPoly(bg, pts=contour, color=255)
            cv2.imwrite('Images/Studio_Masks/'+img_name, mask)
            cv2.imwrite('Images/Studio_Filtered/'+img_name, original_img_color)



