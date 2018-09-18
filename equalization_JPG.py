import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as ss

img = cv2.imread('imJPG.jpg',0)

#BEFORE
his1 = cv2.calcHist([img],[0],None,[256],[0,256])
cummu1 = np.cumsum(his1)
plt.subplot(231), plt.imshow(img,'gray'),plt.title("BEFORE")
plt.subplot(232), plt.plot(his1)
plt.subplot(233), plt.plot(cummu1)

#AFTER
equ = cv2.equalizeHist(img)
his2 = cv2.calcHist([equ],[0],None,[256],[0,256])
cummu2 = np.cumsum(his2)
plt.subplot(234), plt.imshow(equ,'gray'),plt.title("AFTER")
plt.subplot(235), plt.plot(his2)
plt.subplot(236), plt.plot(cummu2)

plt.show()
