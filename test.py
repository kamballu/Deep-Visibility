# -*- coding: utf-8 -*-

from model import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
# Basic working example shown below. Sampling here is by non overlapping patches.

fname = "D:\Documents\PhD\LDR datasets\CSIQ\\1600.png"
qmodel  = model_deep_vis(load_weights=1)
img = get_luminance(cv2.imread(fname))
cv2.imwrite("clean.bmp", np.uint8( img) )
plt.imshow( img,cmap='gray' )
plt.title("Luminance")
plt.colorbar()
plt.show()


fmap = qmodel.get_thresh(img)
fmap = cv2.resize(fmap, img.shape)
plt.imshow(fmap,cmap='jet')
plt.title('Thresholds')
plt.colorbar()
plt.show()

info_mask = np.zeros(fmap.shape)
info_mask[fmap>4] = 1
plt.imshow(info_mask)
plt.show()

# insert information into image
# Scale the information so that the intensity of is less than visibility 
# threshold of the grass, but more than sky. 
noise = np.mean( cv2.imread("noiseadd.bmp"), axis = 2 )/255.0*4
plt.imshow(noise)
plt.title("Noise added")
plt.colorbar()
plt.show()
plt.imshow(img+noise)
cv2.imwrite("noisy1.bmp", np.uint8( np.clip(img+noise, 0, 255) ) )


noise3  = noise/4.0*1.7
plt.imshow(noise3)
plt.title("Noise added")
plt.colorbar()
plt.show()

plt.imshow(img+noise)
cv2.imwrite("noisy3.bmp", np.uint8( np.clip(img+noise3, 0, 255) ) )
