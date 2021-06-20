import numpy as np
import matplotlib.pyplot as plt


def whos(x: np.ndarray):
    if len(np.shape(x)) == 3:
        print(type(x), np.shape(x), type(x[0, 0, 0]))
    else:
        print(type(x), np.shape(x), type(x[0, 0]))


def blackwhite(image: np.ndarray):
    """
    convert 3 channel image to 1 channel image
    expect image array 0-1
    """
    output = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
    return output


def sobel(image: np.ndarray):
    """
    sobel edge detector
    image array 0-1
    """
    threshold = 1 * 0.5
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    height, width = np.shape(image)
    output = np.zeros(shape=(height, width), dtype=np.float32)

    for y in range(1, height-1):
        for x in range(1, width-1):
            sum_x = np.sum(Gx*image[y-1:y+2, x-1:x+2])
            sum_y = np.sum(Gy*image[y-1:y+2, x-1:x+2])
            mag = round(np.sqrt(sum_x**2 + sum_y**2))
            theta = np.arctan2(sum_y, sum_x)
            if mag >= threshold:
                output[y, x] = mag
            else:
                output[y, x] = 0
    return output


def region_of_interest(image: np.ndarray):
    """
    masking image
    image array 0-1
    """
    height, width = np.shape(image)
    # p1 = 0, height
    # p2 = width, 0
    left_slope = (0 - height) / (width - 0)  # neg
    # p1 = 0, 0
    # p2 = width, height
    right_slope = (height - 0) / (width - 0)  # pos
    # y1 = left_slope*x + height
    # y2 = right_slope*x
    output = np.zeros(shape=(height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            y_left = left_slope*x + height
            y_right = right_slope*x
            if y > y_left and y > y_right:
                output[y, x] = 1.0 and image[y, x]  # white
            else:
                output[y, x] = 0 and image[y, x]  # black
    return output


def test_image():
    """
    manual test image, scale 0-1
    """
    image = np.zeros(shape=(1000, 1000), dtype=np.float32)
    for i in range(1000):
        # image[i, i] = 1.0  # theta -45, rho 0
        # image[500, i] = 1.0  # theta -90, rho -500
        # image[i, 500] = 1.0  # theta 0, rho 500
        image[i, 1000-1-i] = 1.0  # theta 45, rho 706
    return image


def normalize(image: np.ndarray):
    """
    rescale image array into 0-1
    """
    mymin = np.min(image)
    mymax = np.max(image)
    image = ((image - mymin) / (mymax - mymin))
    return image


def ceiling(image: np.ndarray):
    """
    rescale image array into 0 or 1
    """
    image = np.where(image > 0, 1.0, 0.0)
    return image


def hough(image: np.ndarray):
    """
    hough transform
    """
    height, width = np.shape(image)
    maxdist = round(np.sqrt(height**2 + width**2))  # max distance
    thetas = np.deg2rad(range(-90, 90, 1))  # range of thetas
    rhos = range(-maxdist, maxdist, 1)  # range of rhos
    accumulator = np.zeros(shape=(len(rhos), len(thetas)), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            if image[y, x] > 0:
                for i in range(len(thetas)):  # for each theta
                    rho = (x*np.cos(thetas[i])
                           + y*np.sin(thetas[i]))  # compute rho
                    accumulator[round(rho)+maxdist, i] += 1  # vote rho x theta

    accumulator = normalize(accumulator)
    # temp = ceiling(accumulator)
    # plt.imshow(temp, cmap='gray', aspect='auto')
    # plt.show()
    return accumulator, thetas, rhos


def inverse_hough(image: np.ndarray, accumulator: np.ndarray,
                  thetas: np.ndarray, rhos: np.ndarray):
    """
    expect image array 0-1
    uses accumulator to take out rhos and thetas
    draw lines onto image
    """
    height, width = np.shape(image)
    threshold = 1 * 0.8
    a_height, a_width = np.shape(accumulator)
    output = np.zeros(shape=(height, width), dtype=np.float32)
    for r in range(a_height):
        for c in range(a_width):
            value = accumulator[r, c]  # at each point
            if value > threshold:  # if > than threshold
                rho = rhos[r]  # fetch rho
                theta = thetas[c]  # fetch theta
                print(f'->rho: {rhos[r]}, theta: {np.rad2deg(thetas[c])},'
                      f'\trow: {r}, col: {c}')
                if rho*np.cos(theta) != 0 and np.sin(theta) != 0:  # divide 0
                    m = round(-np.cos(theta) / np.sin(theta), 10)
                    y_intercept = round(rho / np.sin(theta), 10)
                    print(f'-->{m}, {y_intercept}')
                if theta == 0:  # for vertical line
                    y = range(0, height)  # list of y
                    for i in range(height):
                        # output[y[i], rho] = 1.0
                        image[y[i], rho] = 1.0
                else:  # any other line
                    x = range(0, width)  # list of x
                    for i in range(width):
                        y = round((rho-x[i]*np.cos(theta))
                                  / np.sin(theta))  # calc y
                        if y >= 0 and y <= height-1:
                            # output[y, x[i]] = 1.0
                            image[y, x[i]] = 1.0
    # plt.imshow(output, cmap='gray')
    # plt.show()
    # return output
    return image


def find_left_right_line(accumulator: np.ndarray, thetas: np.ndarray,
                         rhos: np.ndarray):
    """
    expect image array 0-1
    uses accumulator to take out rhos and thetas
    finds negative slope lines and positive slope lines
    returns left/right line (m, b) tuples
    """
    a_height, a_width = np.shape(accumulator)
    threshold = 1 * 0.1
    left_fit = []
    right_fit = []
    for r in range(a_height):
        for c in range(a_width):
            value = accumulator[r, c]  # at each point
            if value > threshold:
                rho = rhos[r]  # fetch rho
                theta = thetas[c]  # fetch theta
                if rho*np.cos(theta) != 0 and np.sin(theta) != 0:  # divide 0
                    m = round(-np.cos(theta) / np.sin(theta), 10)
                    y_intercept = round(rho / np.sin(theta), 10)
                    if m < 0:  # left line has neg slope
                        left_fit.append((m, y_intercept))
                    elif m > 0:  # right line has pos slope
                        right_fit.append((m, y_intercept))
                    else:
                        pass
    print(np.shape(left_fit))  # TODO may be empty
    print(np.shape(right_fit))  # TODO may be empty
    lines = []  # save (m, b) tuples
    if len(left_fit) != 0:
        average_left_fit = np.average(left_fit, axis=0)
        print(average_left_fit)
        lines.append(average_left_fit)
    if len(right_fit) != 0:
        average_right_fit = np.average(right_fit, axis=0)
        print(average_right_fit)
        lines.append(average_right_fit)
    return lines


def plot_lines(image: np.ndarray, lines: np.ndarray):
    """
    expect image array 0-1
    plot lines with (m, b) tuples
    """
    height, width, channels = np.shape(image)
    for line in lines:
        m = line[0]
        b = line[1]
        y1 = height  # 704
        y2 = int(y1*(1/2))
        x1 = int((y1 - b)/m)
        x2 = int((y2 - b)/m)
        plt.plot([x1, x2], [y1, y2], 'ro-')
        plt.imshow(image, cmap='gray')
    plt.show()
    # return image
