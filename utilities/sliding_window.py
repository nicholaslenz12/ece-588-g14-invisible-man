import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="")
parser.add_argument('-s', '--size', type=float, default=120)
parser.add_argument('-d', '--directory', type=str, default='Images/Images_from_Liu_Bolin_s_site/')
parser.add_argument('-m', '--margin', type=float, default=20)
args = parser.parse_args()

if not os.path.exists('sliding_windows'):
    os.makedirs('sliding_windows')

filenames = [f for f in os.listdir(args.directory)
                 if os.path.isfile(os.path.join(args.directory, f))]

filename_pbar = tqdm(filenames)
for f in filename_pbar:
    filename_pbar.set_description("Processing %s" % f)
    
    labels = []
    
    img = cv2.imread(args.directory + f)
    mask = cv2.imread('masks/'+f, 0)

    if not os.path.exists('sliding_windows/'+f.split('.')[0]):
        os.makedirs('sliding_windows/'+f.split('.')[0])

    n_blocks_x = img.shape[1] // args.size
    n_blocks_y = img.shape[0] // args.size

    idx = 0

    for y in range(0, img.shape[0], args.margin):
        for x in range(0, img.shape[1], args.margin):
            window = img[y:y + args.size, x:x + args.size]
            mask_window = mask[y:y + args.size, x:x + args.size]

            if window.shape[0] == args.size and window.shape[1] == args.size:
                cv2.imwrite('sliding_windows/' + f.split('.')[0] + '/{}.png'.format(idx), window)
                
                labels.append(int( np.count_nonzero(mask_window) / (mask_window.shape[0] * mask_window.shape[1]) >= 0.5))
                
                idx += 1

    labels = np.asarray(labels)
    np.savetxt('sliding_windows/' + f.split('.')[0] + '/labels.txt', labels)


