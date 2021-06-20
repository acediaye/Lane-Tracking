import matplotlib.pyplot as plt
from library import *

# imagetest = test_image()
# plt.imshow(imagetest, cmap='gray')
# plt.show()
# myimage = hough(imagetest)
# plt.imshow(myimage, cmap='gray')
# plt.show()

rawimage = plt.imread('pics/test_image.jpg')
whos(rawimage)

myimage = blackwhite(rawimage)
whos(myimage)
# plt.imsave('pics/blackwhite.jpg', myimage, cmap='gray')
myimage = sobel(myimage)  # slow
whos(myimage)
# plt.imsave('pics/sobel.jpg', myimage, cmap='gray')
myimage = region_of_interest(myimage)
whos(myimage)
# plt.imsave('pics/mask.jpg', myimage, cmap='gray')

accumulator, thetas, rhos = hough(myimage)  # slow
myimage = inverse_hough(myimage, accumulator, thetas, rhos)
whos(myimage)
# plt.imsave('pics/inverse.jpg', myimage, cmap='gray')
lines = find_left_right_line(accumulator, thetas, rhos)
plot_lines(rawimage, lines)

# plt.imshow(myimage, cmap='gray')
# plt.show()
