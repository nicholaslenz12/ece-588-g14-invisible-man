import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import cv2
import numpy as np
from tqdm import tqdm
import argparse
import random
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern, draw_multiblock_lbp, multiblock_lbp


random.seed(1)

parser = argparse.ArgumentParser(description="")
parser.add_argument('-s', '--size', type=int, default=120)
parser.add_argument('-d', '--directory', type=str, default='Images/Images_from_Liu_Bolin_s_site/')
parser.add_argument('-m', '--margin', type=int, default=20)
args = parser.parse_args()


radius = 2
n_points = 8 * radius
pos = []
neg = []

filenames = [f for f in os.listdir(args.directory)
                 if os.path.isfile(os.path.join(args.directory, f))]

filename_pbar = tqdm(filenames)
throw_away_count = 0

for f in filename_pbar:
    filename_pbar.set_description("Processing %s" % f)

    img = cv2.imread(args.directory + f, 0)
    mask = cv2.imread('masks/'+f, 0)

    n_blocks_x = img.shape[1] // args.size
    n_blocks_y = img.shape[0] // args.size


    for y in range(0, img.shape[0], args.margin):
        for x in range(0, img.shape[1], args.margin):
            window = img[y:y + args.size, x:x + args.size]
            mask_window = mask[y:y + args.size, x:x + args.size]

            lbp = local_binary_pattern(window, n_points, radius, 'uniform')
            lbp_counts, _ = np.histogram(lbp, bins=np.arange(radius ** 8 + 1), density=True)

            if window.shape[0] == args.size and window.shape[1] == args.size:
                if np.count_nonzero(mask_window) / (mask_window.shape[0] * mask_window.shape[1]) >= 0.8:
                    pos.append(lbp_counts)
                if np.count_nonzero(mask_window) / (mask_window.shape[0] * mask_window.shape[1]) <= 0.15:
                    neg.append(lbp_counts)
                else:
                    throw_away_count += 1

neg = random.sample(neg, min(len(pos), len(neg)))
pos = random.sample(pos, min(len(pos), len(neg)))
pos = np.asarray(pos)
neg = np.asarray(neg)

y = np.concatenate((np.ones(pos.shape[0]), np.zeros(neg.shape[0])))
X = np.concatenate((pos, neg), axis=0)
print(X.shape, y.shape, throw_away_count)
np.savetxt('X.txt', X)
np.savetxt('y.txt', y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=50)
print(X_train.shape, X_test.shape)
np.savetxt('X_train.txt', X_train)
np.savetxt('X_test.txt', X_test)
np.savetxt('y_train.txt', y_train)
np.savetxt('y_test.txt', y_test)
