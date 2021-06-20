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
# plt.imsave('pics/blackwhite.jpg', myimage, cmap='gray')
whos(myimage)
myimage = sobel(myimage)  # slow
# plt.imsave('pics/sobel.jpg', myimage, cmap='gray')
whos(myimage)
myimage = region_of_interest(myimage)
# plt.imsave('pics/mask.jpg', myimage, cmap='gray')
whos(myimage)

# # plt.imsave('pics/mine.jpg', myimage, cmap='gray')
# # myimage = plt.imread('pics/mine.jpg')

accumulator, thetas, rhos = hough(myimage)  # slow
myimage = inverse_hough(myimage, accumulator, thetas, rhos)
# plt.imsave('pics/inverse.jpg', myimage, cmap='gray')
whos(myimage)
lines = find_left_right_line(myimage, accumulator, thetas, rhos)
myimage = plot_lines(myimage, lines)

# p1 = [1309, 704]
# p2 = [1176, 422]
# pp1 = [817, 704]
# pp2 = [929, 422]
# plt.plot([1309, 1176], [704, 422], 'ro-')
# plt.plot([817, 929], [704, 422], 'go-')
# plt.imshow(myimage, cmap='gray')
plt.show()

# m1, b1 = [-2.11442821e+00, -2.06536814e+03]  # 14
# m2, b2 = [2.51428030e+00, 2.76010084e+03]
# # x1 = 1000
# # x2 = 1000
# # y1 = m1*x1 + b1
# # y2 = m2*x2 + b2
# y1 = 704  # 704
# y2 = int(y1*(3/5))  # 424
# x1 = int(-(y1 - b1)/m1)
# x2 = int(-(y2 - b1)/m1)
# print(x1, y1)
# print(x2, y2)
# xx1 = int(-(y1 - b2)/m2)
# xx2 = int(-(y2 - b2)/m2)
# print(xx1, y1)
# print(xx2, y2)
# plt.plot([x1, x2], [y1, y2], 'ro-')
# plt.plot([xx1, xx2], [y1, y2], 'go-')
# plt.show()
