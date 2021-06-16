import numpy as np
import matplotlib.pyplot as plt

def whos(x: np.ndarray):
    if isinstance(x, np.ndarray):
        print(type(x), np.shape(x), type(x[0, 0]))
    else:
        print(type(x))
    
def blackwhite(image: np.ndarray):
    image = image.astype(np.int)
    height, width, channels = np.shape(image)
    output = np.zeros(shape=(height, width))
    # for r in range(height):
    #     for c in range(width):
    #         # print(image[0, 0, 0], type(image[0, 0, 0]))
    #         # print((image[r, c, 0] + image[r, c, 1] + image[r, c, 2]) / 3)
    #         output[r, c] = round((image[r, c, 0] + image[r, c, 1] + image[r, c, 2]) / 3)
    output = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
    output = output.astype(np.uint8)
    return output

def sobel(image: np.ndarray):
    image = image.astype(np.int)
    threshold = 255 * 0.5
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    height, width = np.shape(image)
    output = np.zeros(shape=(height, width))
    
    for r in range(1, height-1):
        for c in range(1, width-1):
            Sx = np.sum(Gx*image[r-1:r+2, c-1:c+2])
            Sy = np.sum(Gy*image[r-1:r+2, c-1:c+2])
            # print(type(S1), S1)
            mag = round(np.sqrt(Sx**2 + Sy**2))
            theta = np.arctan2(Sy, Sx)
            if mag >= threshold:
                output[r, c] = mag
            else:
                output[r, c] = 0
    output = output.astype(np.uint8)
    # print(np.amax(output), np.amin(output))
    return output

def region_of_interest(image: np.ndarray):
    height, width = np.shape(image)
    # p1 = 0, height
    # p2 = width, 0
    left_slope = (0 - height) / (width - 0)  # neg, + height
    # p1 = 0, 0
    # p2 = width, height
    right_slope = (height - 0) / (width - 0)  # pos
    # x = range(width)
    # y1 = left_slope*x + height
    # y2 = right_slope*x
    output = np.zeros(shape=(height, width))
    for r in range(height):
        for c in range(width):
            y1 = left_slope*c + height
            y2 = right_slope*c
            if r > y1 and r > y2:
                output[r, c] = 255 and image[r, c]  # white
            else:
                output[r, c] = 0 and image[r, c]  # black
    output = output.astype(np.uint8)
    return output
    
def test_image():
    image = np.zeros(shape=(1000, 1000))
    for i in range(1000):
        image[i, i] = 255
        image[500, i] = 255
        image[i, 500] = 255
        image[i, 1000-1-i] = 255
    return image 

def hough(image: np.ndarray):
    image = image.astype(np.int)
    height, width = np.shape(image)
    maxdist = round(np.sqrt(height**2 + width**2))  # max distance
    thetas = range(-90+1, 90+1, 1)  # range of thetas
    rhos = range(-maxdist+1, maxdist+1, 1)  # range of rhos
    # print(maxdist)
    accumulator = np.zeros(shape=(len(rhos), len(thetas)))
    count = 0
    for r in range(height):
        for c in range(width):
            if image[r, c] > 0:
                # count += 1
                for i in range(len(thetas)):  # for each theta
                    rho = c*np.cos(np.deg2rad(i)) + r*np.sin(np.deg2rad(i))  # compute rho
                    accumulator[round(rho)+maxdist, i] += 1  # vote rho x theta
    accumulator = accumulator.astype(np.uint8)
    plt.imshow(255*accumulator, cmap='gray', aspect='auto')    
    plt.show()

    threshold = 255 * 0.9
    a_height, a_width = np.shape(accumulator)
    output = np.zeros(shape=(height, width))
    for r in range(a_height):
        for c in range(a_width):
            value = accumulator[r, c]  # at each point
            if value > threshold:  # if > than threshold
                rho = rhos[r]  # fetch rho
                theta = thetas[c]  # fetch theta
                print(f'->rho: {rhos[r]}, theta: {thetas[c]}')

                if theta == 0:  # for vertical line
                    x = (width)*[rho]  # list of x
                    y = range(0, height)  # list of y
                    for i in range(width):
                        output[y[i], x[i]] = 255
                else:  # any other line
                    x = range(0, width)  # list of x
                    for i in range(width):
                        y = (rho-x[i]*np.cos(np.deg2rad(theta))) / np.sin(np.deg2rad(theta))  # calc y
                        # print(x[i], y)  +1278, -1535
                        if y>=0 and y<=height-1:
                            count += 1
                            output[round(y), x[i]] = 255
    output = output.astype(np.uint8)
    print(count)
    plt.imshow(output, cmap='gray')    
    plt.show()
    return output
