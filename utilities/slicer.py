import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="")
parser.add_argument('-s', '--size', type=float, default=60)
parser.add_argument('-d', '--directory', type=str, default='Images/Images_from_Liu_Bolin_s_site/')
args = parser.parse_args()

if not os.path.exists('blocks'):
    os.makedirs('blocks')

filenames = [f for f in os.listdir(args.directory)
                 if os.path.isfile(os.path.join(args.directory, f))]

filename_pbar = tqdm(filenames)
for f in filename_pbar:
    filename_pbar.set_description("Processing %s" % f)
    img = cv2.imread(args.directory + f)

    if not os.path.exists('blocks/'+f.split('.')[0]):
        os.makedirs('blocks/'+f.split('.')[0])

    dpi_ = 200
    fig = plt.figure(figsize=(float(img.shape[1]) / dpi_, float(img.shape[0]) / dpi_), dpi=dpi_)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    grid_x_ticks = np.arange(0, img.shape[1], args.size)
    grid_y_ticks = np.arange(0, img.shape[0], args.size)
    ax.set_xticks(grid_x_ticks, minor=False)
    ax.set_yticks(grid_y_ticks, minor=False)
    ax.grid(axis='both', linestyle='-')
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(args.size)))
    ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(args.size)))
    for j in range(ny):
        y = args.size / 2 + j * args.size
        for i in range(nx):
            x = args.size / 2. + float(i) * args.size
            ax.text(x, y, '{:d}'.format(i + j * nx), color='w', ha='center', va='center', fontsize=6)

    fig.savefig('blocks/'+f.split('.')[0]+'/grid_index.png')
    plt.clf()
    plt.close()

    n_blocks_x = img.shape[1] // args.size
    n_blocks_y = img.shape[0] // args.size

    idx = 0
    for b_x in range(n_blocks_x):
        for b_y in range(n_blocks_y):
            block = img[b_y * args.size:(b_y + 1) * args.size, b_x * args.size:(b_x + 1) * args.size]

            if block.shape[0] == args.size and block.shape[1] == args.size:
                cv2.imwrite('blocks/'+f.split('.')[0]+'/{}.png'.format(idx), block)

            idx += 1



