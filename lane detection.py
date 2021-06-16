# https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132
# https://www.geeksforgeeks.org/opencv-real-time-road-lane-detection/
# https://www.youtube.com/watch?v=eLTLtUVuuy4

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from library import *

# imagetest = test_image()
# plt.imshow(imagetest, cmap='gray')
# plt.show()
# hough(imagetest)

image = cv2.imread('pics/test_image.jpg')
# image = plt.imread('pics/test_image.jpg')
print(np.shape(image))
# myimage = (image[:,:,0]+image[:,:,1]+image[:,:,2])/3

myimage = blackwhite(image)
whos(myimage)
# myimage = sobel(myimage)
# whos(myimage)
# myimage = region_of_interest(myimage)
# whos(myimage)
# myimage = hough(myimage)
# whos(myimage)

# cv2.imwrite('pics/mine.jpg', myimage)
# cv2.imshow('result', myimage)
# cv2.waitKey(0)

plt.imshow(myimage, cmap='gray')
plt.show()