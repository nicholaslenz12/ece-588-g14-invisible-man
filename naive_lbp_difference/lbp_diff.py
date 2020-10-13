import cv2
import numpy as np
import random
import os
import io
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, draw_multiblock_lbp, multiblock_lbp
import json

if __name__ == '__main__':
    filenames = [f for f in os.listdir('lbp')
                 if f.endswith('json')]

    for f in filenames:
        img_name = f.split('_')[0]+'.PNG'
        print(img_name)
        img_ = cv2.imread('Images/Images_from_Liu_Bolin_s_site/' + img_name)
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
        sharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0, gaussian)

        with open('lbp/'+f) as fp:
            lbp_dict = json.load(fp)

        n_blocks_x = 0
        n_blocks_y = 0
        for keys in lbp_dict.keys():
            x_, y_ = keys.split('_')
            x_, y_ = int(x_), int(y_)
            n_blocks_x = max(x_, n_blocks_x)
            n_blocks_y = max(y_, n_blocks_y)

        n_blocks_x += 1
        n_blocks_y += 1

        step_x = img.shape[1] // n_blocks_x
        step_y = img.shape[0] // n_blocks_y

        fig, axarr = plt.subplots(n_blocks_y, n_blocks_x, figsize=(n_blocks_x * 2 * (step_x/100), 15 * (step_x/100)))

        for b_x in range(n_blocks_x):
            for b_y in range(n_blocks_y):

                block = sharp[b_y * step_y:(b_y + 1) * step_y + 1, b_x * step_x:(b_x + 1) * step_x + 1].copy()
                lbp_hist_bins = lbp_dict[str(b_x)+'_'+str(b_y)]

                if 1 <= b_x <= n_blocks_x-2 and 1 <= b_y <= n_blocks_y-2:
                    ul_hist = np.asarray(lbp_dict[str(b_x-1)+'_'+str(b_y-1)])
                    l_hist = np.asarray(lbp_dict[str(b_x-1)+'_'+str(b_y)])
                    ll_hist = np.asarray(lbp_dict[str(b_x-1)+'_'+str(b_y+1)])
                    ur_hist = np.asarray(lbp_dict[str(b_x + 1) + '_' + str(b_y - 1)])
                    r_hist = np.asarray(lbp_dict[str(b_x + 1) + '_' + str(b_y)])
                    lr_hist = np.asarray(lbp_dict[str(b_x + 1) + '_' + str(b_y + 1)])
                    c_hist = lbp_hist_bins

                    neighbours = [ul_hist, l_hist, ll_hist, ur_hist, r_hist, lr_hist]
                    mses = []
                    for neighbour in neighbours:
                        mses.append(np.sqrt(np.mean(np.square(c_hist - neighbour))) / np.sum(neighbour))
                    mse = np.round(np.mean(np.asarray(mses)), 3)

                    block = cv2.putText(block, str(mse), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5*(step_x/100), 0, 2, cv2.LINE_AA)
                    block = cv2.putText(block, str(mse), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5*(step_x/100), 255, 1, cv2.LINE_AA)

                # if 1 <= b_x <= n_blocks_x-2:
                #     l_hist = np.asarray(lbp_dict[str(b_x-1)+'_'+str(b_y)][0])
                #     r_hist = np.asarray(lbp_dict[str(b_x + 1) + '_' + str(b_y)][0])
                #     c_hist = lbp_hist_bins[0]
                #
                #     neighbours = [l_hist, r_hist]
                #     mses = []
                #     for neighbour in neighbours:
                #         mses.append(np.sqrt(np.mean(np.square(c_hist - neighbour))) / np.sum(neighbour))
                #     mse = np.round(np.mean(np.asarray(mses)), 3)
                #
                #     block = cv2.putText(block, str(mse), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5*(step_x/100), 0, 2, cv2.LINE_AA)
                #     block = cv2.putText(block, str(mse), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5*(step_x/100), 255, 1, cv2.LINE_AA)

                axarr[b_y, b_x].axis('off')
                axarr[b_y, b_x].imshow(block, cmap='Greys_r')

        with io.BytesIO() as stream:
            plt.savefig(stream, pad_inches=0, bbox_inches='tight')
            stream.seek(0)
            byte_stream = np.fromstring(stream.getvalue(), dtype=np.uint8)

        lbp_analysis = cv2.imdecode(byte_stream, cv2.IMREAD_UNCHANGED)
        lbp_analysis = cv2.cvtColor(lbp_analysis, cv2.COLOR_BGR2RGB)
        new_shape = (int(sharp.shape[1] / sharp.shape[0] * lbp_analysis.shape[0]), lbp_analysis.shape[0])
        new_sharp = cv2.resize(sharp, new_shape)
        cv2.imwrite('lbp/'+f.split('_')[0]+'_mse.png', np.hstack((np.stack((new_sharp, ) * 3, axis=-1), lbp_analysis)))





