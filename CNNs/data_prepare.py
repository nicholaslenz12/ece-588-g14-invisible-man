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

random.seed(1)

shutil.rmtree('keras_test/dataset/train/0')
shutil.rmtree('keras_test/dataset/train/1')
shutil.rmtree('keras_test/dataset/test/0')
shutil.rmtree('keras_test/dataset/test/1')
shutil.rmtree('keras_test/dataset/valid/0')
shutil.rmtree('keras_test/dataset/valid/1')
os.mkdir('keras_test/dataset/train/0/')
os.mkdir('keras_test/dataset/train/1/')
os.mkdir('keras_test/dataset/test/0/')
os.mkdir('keras_test/dataset/test/1/')
os.mkdir('keras_test/dataset/valid/0/')
os.mkdir('keras_test/dataset/valid/1/')

parser = argparse.ArgumentParser(description="")
parser.add_argument('-s', '--size', type=int, default=120)
parser.add_argument('-d', '--directory', type=str, default='Images/Images_from_Liu_Bolin_s_site/')
parser.add_argument('-m', '--margin', type=int, default=20)
args = parser.parse_args()

pos = []
neg = []

filenames = [f for f in os.listdir(args.directory)
                 if os.path.isfile(os.path.join(args.directory, f))]

filename_pbar = tqdm(filenames)
for f in filename_pbar:
    filename_pbar.set_description("Processing %s" % f)

    img = cv2.imread(args.directory + f)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('masks/'+f, 0)

    n_blocks_x = img.shape[1] // args.size
    n_blocks_y = img.shape[0] // args.size

    idx = 0

    for y in range(0, img.shape[0], args.margin):
        for x in range(0, img.shape[1], args.margin):
            window = img[y:y + args.size, x:x + args.size]
            mask_window = mask[y:y + args.size, x:x + args.size]

            if window.shape[0] == args.size and window.shape[1] == args.size:
                if np.count_nonzero(mask_window) / (mask_window.shape[0] * mask_window.shape[1]) >= 0.8:
                    pos.append(window)
                else:
                    neg.append(window)
                idx += 1

neg = random.sample(neg, len(pos))

pos = np.asarray(pos)
neg = np.asarray(neg)

'''pos - 0, neg - 1'''

y = np.concatenate((np.zeros(pos.shape[0]), np.ones(neg.shape[0])))
X = np.concatenate((pos, neg), axis=0)
print(X.shape, y.shape)

# for i in tqdm(range(X.shape[0])):
#     label = y[i]
#     cv2.imwrite('keras_test/dataset/{}/{}.jpg'.format(str(int(label)), i), X[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=50)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=50)
print(X_train.shape, X_test.shape)


print('Saving Train')
for i in tqdm(range(X_train.shape[0])):
    label = y_train[i]
    cv2.imwrite('keras_test/dataset/train/{}/{}.jpg'.format(str(int(label)), i), X_train[i])
print('Saving Valid')
for i in tqdm(range(X_valid.shape[0])):
    label = y_valid[i]
    cv2.imwrite('keras_test/dataset/valid/{}/{}.jpg'.format(str(int(label)), i), X_train[i])
print('Saving Test')
for i in tqdm(range(X_test.shape[0])):
    label = y_test[i]
    cv2.imwrite('keras_test/dataset/test/{}/{}.jpg'.format(str(int(label)), i), X_test[i])

