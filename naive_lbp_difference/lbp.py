# LBP https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html

import cv2
import numpy as np
import random
import os
import io
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, draw_multiblock_lbp, multiblock_lbp
import json


def valley_filter(img, x, y, sample_height, sample_width, valley_map):
    if 2*sample_width <= x <= img.shape[1] - sample_width*2 and 2*sample_height <= y <= img.shape[0] - sample_height*2:
        left = img[y - 2*sample_height: y + 2*sample_height+1, x - 2*sample_width: x - sample_width+1]
        right = img[y - 2*sample_height: y + 2*sample_height+1, x + sample_width: x + 2*sample_width+1]
        center = img[y - 2*sample_height: y + 2*sample_height+1, x - sample_width: x + sample_width+1]
        valley_map[y, x] = int(np.average(left) > np.average(center) and np.average(right) > np.average(center))
    return valley_map


def valley_filter_2(img):
    pass


if __name__ == '__main__':
    filenames = [f for f in os.listdir('Images/Images_from_Liu_Bolin_s_site')
                 if os.path.isfile(os.path.join('Images/Images_from_Liu_Bolin_s_site', f))]

    for f in filenames:
        print(f)
        plt.clf()
        img_ = cv2.imread('Images/Images_from_Liu_Bolin_s_site/' + f)
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
        sharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0, gaussian)

        # sobel_x = cv2.Sobel(sharp, cv2.CV_16S, 1, 0)
        # magx = np.linalg.norm(sobel_x)
        # magx = magx * magx / (sobel_x.shape[0] * sobel_x.shape[1])
        #
        # sobel_y = cv2.Sobel(sharp, cv2.CV_16S, 0, 1)
        # magy = np.linalg.norm(sobel_y)
        # magy = magy * magy / (sobel_y.shape[0] * sobel_y.shape[1])

        # valley_map = np.zeros_like(sharp)
        # for y in range(sharp.shape[0]):
        #     for x in range(sharp.shape[1]):
        #         valley_map = valley_filter(sobel_x, x, y, 50, 1, valley_map)

        lbp_dict = dict()

        radius = 2
        n_points = 8 * radius
        n_blocks_x = 8
        n_blocks_y = 5
        step_x = img.shape[1] // n_blocks_x
        step_y = img.shape[0] // n_blocks_y

        fig, axarr = plt.subplots(n_blocks_y, n_blocks_x*2, figsize=(n_blocks_x*6, 18))

        w = radius - 1
        edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
        flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
        i_14 = n_points // 4  # 1/4th of the histogram
        i_34 = 3 * (n_points // 4)  # 3/4th of the histogram
        corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                         list(range(i_34 - w, i_34 + w + 1)))

        selected_features = list(edge_labels) + corner_labels

        for b_x in range(n_blocks_x):
            for b_y in range(n_blocks_y):

                block = sharp[b_y*step_y:(b_y+1)*step_y+1, b_x*step_x:(b_x+1)*step_x+1].copy()
                lbp = local_binary_pattern(block, n_points, radius, 'uniform')
                lbp = lbp.astype(np.int)
                bins = range(np.max(np.unique(lbp))+1)
                # n_bins = int(lbp.max() + 1)
                # hist, bin_edges = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
                axarr[b_y, b_x*2].axis('off')
                axarr[b_y, b_x*2].imshow(block, cmap='Greys_r')
                axarr[b_y, b_x*2+1].hist(x=lbp, bins=bins)
                lbp_counts = np.unique(lbp, return_counts=True)[1]
                # lbp_dict[str(b_x)+'_'+str(b_y)] = lbp_counts[selected_features].tolist()
                lbp_dict[str(b_x)+'_'+str(b_y)] = lbp_counts.tolist()

        with io.BytesIO() as stream:
            plt.savefig(stream, pad_inches=0, bbox_inches='tight')
            stream.seek(0)
            byte_stream = np.fromstring(stream.getvalue(), dtype=np.uint8)

        lbp_analysis = cv2.imdecode(byte_stream, cv2.IMREAD_UNCHANGED)
        lbp_analysis = cv2.cvtColor(lbp_analysis, cv2.COLOR_BGR2RGB)
        new_shape = (int(sharp.shape[1] / sharp.shape[0] * lbp_analysis.shape[0]), lbp_analysis.shape[0])
        new_sharp = cv2.resize(sharp, new_shape)
        cv2.imwrite('lbp/'+f, np.hstack((np.stack((new_sharp, ) * 3, axis=-1), lbp_analysis)))

        # hstacked1 = np.hstack((sharp, sobel_y+sobel_x))
        # hstacked2 = np.hstack((sobel_y, sobel_x))
        # cv2.imwrite('valley_filter/'+f, np.vstack((hstacked2, hstacked1)))

        with open('lbp/{}_lbp_hist.json'.format(f.split('.')[0]), 'w') as fp:
            json.dump(lbp_dict, fp)


