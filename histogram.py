import cv2
import numpy as np
import imageio as imo
from matplotlib import pyplot as plt

#GIF
gif  = imo.mimread('imGIF.gif')
im_gif = cv2.cvtColor(gif[0],cv2.COLOR_BGR2GRAY)
plt.subplot(10,2,1), plt.imshow(im_gif,'gray')
plt.subplot(10,2,2), plt.hist(im_gif.ravel(),256,[0,256]),plt.title("GIF")

#JPG
jpg = cv2.imread('imJPG.jpg', cv2.IMREAD_GRAYSCALE)
im_jpg = cv2.imread('imJPG.jpg',0)
plt.subplot(5,2,3), plt.imshow(im_jpg, 'gray'), 
plt.subplot(5,2,4), plt.hist(jpg.ravel(),256,[0,256]),plt.title("JPG")

#BMP
bmp = cv2.imread('imBMP.bmp', cv2.IMREAD_GRAYSCALE)
im_bmp = cv2.imread('imBMP.bmp',0)
plt.subplot(5,2,5), plt.imshow(im_bmp, 'gray')
plt.subplot(5,2,6), plt.hist(bmp.ravel(),256,[0,256]),plt.title("BMP")

#PNG
png = cv2.imread('imPNG.png', cv2.IMREAD_GRAYSCALE)
im_png = cv2.imread('imPNG.png',0)
plt.subplot(5,2,7), plt.imshow(im_png, 'gray')
plt.subplot(5,2,8), plt.hist(png.ravel(),256,[0,256]),plt.title("PNG")

#TIFF
tiff = cv2.imread('imTIFF.tiff', cv2.IMREAD_GRAYSCALE)
im_tiff = cv2.imread('imTIFF.tiff',0)
plt.subplot(5,2,9), plt.imshow(im_tiff, 'gray')
plt.subplot(5,2,10), plt.hist(tiff.ravel(),256,[0,256]),plt.title("TIFF")

plt.show()

