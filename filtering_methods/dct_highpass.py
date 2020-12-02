# -*- coding: utf-8 -*-
#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from scipy import fftpack

# Code from discussion section,
# https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
# at Cal. Added low-pass filtering in blocks

def dct2(M):
    return fftpack.dct(fftpack.dct(M, axis=0, norm='ortho'),
                       axis=1,
                       norm='ortho')

def idct2(M):
    return fftpack.idct(fftpack.idct(M, axis=0, norm='ortho'),
                        axis=1,
                        norm='ortho')

def highpass_image(im, cutoff_x, cutoff_y, patchSize=8, level=0):
    imsize = im.shape
    dct = np.zeros(imsize)
    
    for i in np.r_[:imsize[0]:patchSize]:
        for j in np.r_[:imsize[1]:patchSize]:
            dct[i:(i+patchSize),j:(j+patchSize)] = dct2(im[i:(i+patchSize),
                                                           j:(j+patchSize)])
            
            # High pass filter in the frequency domain
            # dct[i:(i+patchSize), j:(j+cutoff_y)] *= level
            # dct[i:(i+cutoff_x), j:(j+patchSize)] *= level
            dct[i, j] *= level
            dct[i, j] *= level
            
    out = np.zeros(imsize)
    for i in np.r_[:imsize[0]:patchSize]:
        for j in np.r_[:imsize[1]:patchSize]:
            out[i:(i+patchSize),j:(j+patchSize)] = idct2(dct[i:(i+patchSize),
                                                             j:(j+patchSize)])
    return out

def apply_all(direc, method, *args):
    processed_direc = direc + 'Processed/'
    os.mkdir(processed_direc)
    files = os.listdir(direc)
    files_filt = [file for file in files if file.endswith("jpg")]
    for file in files_filt:
        print(file)
        im = cv2.imread(direc + file)
        proc = method(im, *args)
        cv2.imwrite(processed_direc + file, proc)
        

#%%
im = cv2.imread("/Users/nicholaslenz/Desktop/ece-588-g14-invisible-man/Images/" +
                "Images_from_Liu_Bolin_s_site/Liu9.jpg")
out = highpass_image(im, 1, 1, 8, 0)
out_uint8 = out.astype('uint8')
plt.imshow(cv2.cvtColor(out_uint8, cv2.COLOR_BGR2RGB))
plt.show()

# print(os.getcwd())
# cv2.imwrite("/Users/nicholaslenz/Desktop/out.jpg", out)

# apply_all("/Users/nicholaslenz/Desktop/ece-588-g14-invisible-man/Images/" +
#           "Images_from_Liu_Bolin_s_site/",
#           highpass_image,
#           0,
#           1)
