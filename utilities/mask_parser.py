import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import json

with open('mask_labels.json', 'r') as f:
    mask_labels = json.load(f)

mask_labels_pbar = tqdm(mask_labels)
for row in mask_labels_pbar:
    img_name = row['External ID']
    mask_labels_pbar.set_description("Processing %s" % img_name)
    mask_coords_dicts = row['Label']['objects'][0]['polygon']
    original_img = cv2.imread('Images/Images_from_Liu_Bolin_s_site/'+img_name, 0)
    bg = np.zeros_like(original_img)
    mask_coords = []
    for mask_coords_dict in mask_coords_dicts:
        mask_coords.append([mask_coords_dict['x'], mask_coords_dict['y']])
    contour = [np.array(mask_coords, dtype=np.int32)]
    mask = cv2.fillPoly(bg, pts=contour, color=255)
    cv2.imwrite('masks/'+img_name, mask)



