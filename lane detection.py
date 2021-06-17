# https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132
# https://www.geeksforgeeks.org/opencv-real-time-road-lane-detection/
# https://www.youtube.com/watch?v=eLTLtUVuuy4

import cv2
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from library import *

# imagetest = test_image()
# plt.imshow(imagetest, cmap='gray')
# plt.show()
# myimage = hough(imagetest)
# plt.imshow(myimage, cmap='gray')
# plt.show()

image = plt.imread('pics/test_image.jpg')
whos(image)

myimage = blackwhite(image)
# print(myimage)
whos(myimage)
myimage = sobel(myimage)
whos(myimage)
myimage = region_of_interest(myimage)
whos(myimage)

# # plt.imsave('pics/mine.jpg', myimage, cmap='gray')
# # myimage = plt.imread('pics/mine.jpg')

myimage = hough(myimage)
whos(myimage)

plt.imshow(myimage, cmap='gray')
plt.show()
