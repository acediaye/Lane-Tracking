import numpy as np
import matplotlib.pyplot as plt


def whos(x: np.ndarray):
    if len(np.shape(x)) == 3:
        print(type(x), np.shape(x), type(x[0, 0, 0]))
    else:
        print(type(x), np.shape(x), type(x[0, 0]))


def blackwhite(image: np.ndarray):
    output = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
    # output = output.astype(np.uint8)
    return output


def sobel(image: np.ndarray):
    threshold = 1 * 0.5
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    height, width = np.shape(image)
    output = np.zeros(shape=(height, width), dtype=np.float32)

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
    output = np.zeros(shape=(height, width), dtype=np.float32)
    for r in range(height):
        for c in range(width):
            y1 = left_slope*c + height
            y2 = right_slope*c
            if r > y1 and r > y2:
                output[r, c] = 1.0 and image[r, c]  # white
            else:
                output[r, c] = 0 and image[r, c]  # black
    return output


def test_image():
    image = np.zeros(shape=(1000, 1000), dtype=np.float32)
    for i in range(1000):
        image[i, i] = 1.0
        image[500, i] = 1.0
        image[i, 500] = 1.0
        image[i, 1000-1-i] = 1.0
    return image


def normalize(image: np.ndarray):
    mymin = np.min(image)
    mymax = np.max(image)
    image = ((image - mymin) / (mymax - mymin))
    return image


def ceiling(image: np.ndarray):
    image = np.where(image > 0, 1.0, 0.0)
    return image


def hough(image: np.ndarray):
    height, width = np.shape(image)
    maxdist = round(np.sqrt(height**2 + width**2))  # max distance
    thetas = range(-90, 90, 1)  # range of thetas
    rhos = range(-maxdist, maxdist, 1)  # range of rhos
    # print(maxdist)
    accumulator = np.zeros(shape=(len(rhos), len(thetas)), dtype=np.float32)
    count = 0
    for r in range(height):
        for c in range(width):
            if image[r, c] > 0:
                # count += 1
                for i in range(len(thetas)):  # for each theta
                    rho = (c*np.cos(np.deg2rad(i))
                           + r*np.sin(np.deg2rad(i)))  # compute rho
                    accumulator[round(rho)+maxdist, i] += 1  # vote rho x theta

    accumulator = normalize(accumulator)
    # temp = ceiling(accumulator)
    # plt.imshow(temp, cmap='gray', aspect='auto')
    # plt.show()

    threshold = 1 * 0.8
    a_height, a_width = np.shape(accumulator)
    output = np.zeros(shape=(height, width), dtype=np.float32)
    for r in range(a_height):
        for c in range(a_width):
            value = accumulator[r, c]  # at each point
            if value > threshold:  # if > than threshold
                rho = rhos[r]  # fetch rho
                theta = thetas[c]  # fetch theta
                print(f'->rho: {rhos[r]}, theta: {thetas[c]},'
                      '\trow: {r}, col: {c}')

                if theta == 0:  # for vertical line
                    # x = (width)*[rho]  # list of x
                    y = range(0, height)  # list of y
                    for i in range(height):
                        output[y[i], rho] = 1.0
                else:  # any other line
                    x = range(0, width)  # list of x
                    for i in range(width):
                        y = ((rho-x[i]*np.cos(np.deg2rad(theta)))
                             / np.sin(np.deg2rad(theta)))  # calc y
                        y = np.abs(round(y))
                        print(x[i], y)  # +1278, -1535
                        if y >= 0 and y <= height-1:
                            count += 1
                            output[y, x[i]] = 1.0

                # x = rho*np.cos(np.deg2rad(theta))
                # y = rho*np.sin(np.deg2rad(theta))
                # print(x, y)
                # count += 1
                # output[round(y), round(x)] = 1.0
    print(count)
    # plt.imshow(output, cmap='gray')
    # plt.show()
    return output
