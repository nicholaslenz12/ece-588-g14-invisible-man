import cv2
import skimage
import numpy as np
import os
from tqdm import tqdm
from AmbrosioTortorelliMinimizer import *
from mumford_sha import apply_mumford_sha
import matplotlib.pyplot as plt


block_size = 120


def calc_entropy(img):
    margin = np.histogramdd(np.ravel(img), bins=256)[0] / img.size
    margin = list(filter(lambda p: p > 0, np.ravel(margin)))
    entropy = -np.sum(np.multiply(margin, np.log2(margin)))
    return entropy


filenames = [f for f in os.listdir('Images/Images_from_Liu_Bolin_s_site/')
                 if f.endswith('PNG')]

filename_pbar = tqdm(filenames)
for fn in filename_pbar:
    filename_pbar.set_description("Processing %s" % fn)
    img = cv2.imread('Images/Images_from_Liu_Bolin_s_site/' + fn)
    first_iter = apply_mumford_sha(img)
    second_iter = apply_mumford_sha(first_iter, alpha=100000)

    n_blocks_x = img.shape[1] // block_size
    n_blocks_y = img.shape[0] // block_size

    dpi_ = 200
    fig = plt.figure(figsize=(float(img.shape[1]) / dpi_, float(img.shape[0]) / dpi_), dpi=dpi_)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    grid_x_ticks = np.arange(0, img.shape[1], block_size)
    grid_y_ticks = np.arange(0, img.shape[0], block_size)
    ax.set_xticks(grid_x_ticks, minor=False)
    ax.set_yticks(grid_y_ticks, minor=False)
    ax.grid(axis='both', linestyle='-')
    ax.imshow(cv2.cvtColor(second_iter, cv2.COLOR_BGR2RGB))
    nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(block_size)))
    ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(block_size)))
    for b_y in range(ny):
        y = block_size / 2 + b_y * block_size
        for b_x in range(nx):
            block = second_iter[b_y * block_size:(b_y + 1) * block_size, b_x * block_size:(b_x + 1) * block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                entropy = calc_entropy(gray_block)

                x = block_size / 2. + float(b_x) * block_size

                ax.text(x, y, '{}'.format(round(entropy, 2)), color='w', ha='center', va='center', fontsize=6)

    fig.savefig('entropy_grid/' + fn)
    plt.clf()
    plt.close()
    # break






