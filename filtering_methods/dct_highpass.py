# -*- coding: utf-8 -*-
#%%
from numpy import zeros, r_
import matplotlib.pyplot as plt
import cv2

from scipy import fftpack

# Code from discussion section,
# https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
# at Cal. Added low-pass filtering in blocks

def dct2(M):
    return fftpack.dct(fftpack.dct(M, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(M):
    return fftpack.idct(fftpack.idct(M, axis=0, norm='ortho'), axis=1, norm='ortho')

#%%
im = cv2.imread("/Users/nicholaslenz/Desktop/ece-588-g14-invisible-man/Images/" +
                "Images_from_Liu_Bolin_s_site/Liu1.jpg")

imsize = im.shape
dct = zeros(imsize)

cutoff_x = 1
cutoff_y = 1
L = 8

for i in r_[:imsize[0]:L]:
    for j in r_[:imsize[1]:L]:
        dct[i:(i+L),j:(j+L)] = dct2( im[i:(i+L),j:(j+L)])
        
        # High pass filter in the frequency domain
        dct[i:(i+L), j:(j+cutoff_x)] = 0
        dct[i:(i + cutoff_y), j:(j+L)] = 0
        
out = zeros(imsize)
for i in r_[:imsize[0]:L]:
    for j in r_[:imsize[1]:L]:
        out[i:(i+L),j:(j+L)] = idct2( dct[i:(i+L),j:(j+L)])

plt.imshow(cv2.cvtColor(out.astype('uint8'), cv2.COLOR_BGR2RGB))