import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def draw_binary_matrix(mat):
    """
    draws the given matrix with matplotlib
    :param mat: the binary matrix to draw
    :return:
    """
    cmap = ListedColormap(["b", "g"])
    matrix = np.array(mat, dtype=np.uint8)
    plt.imshow(matrix, cmap=cmap)


def add_lists(lists):
    """
    adds lists of length at most three by index
    :param lists: list of lists to add
    :return: the sum of the lists
    """
    ret = [0, 0, 0]
    for lst in lists:
        for i in range(len(lst)):
            ret[i] += lst[i]
    return ret


def plot_line_low(x0, y0, x1, y1, matrix):
    """
    helper function to plot a binary line in a binary matrix
    """
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    d = (2 * dy) - dx
    y = y0

    for x in range(x0, x1 + 1):
        if x < len(matrix) and y < len(matrix):
            matrix[x][y] = 1
        else:
            pass
            # print("illegal", x, y)
        if d > 0:
            y = y + yi
            d = d + (2 * (dy - dx))
        else:
            d = d + 2 * dy


def plot_line_high(x0, y0, x1, y1, matrix):
    """
    helper function to plot a binary line in a binary matrix
    """
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx

    d = (2 * dx) - dy
    x = x0

    for y in range(y0, y1 + 1):
        if x < len(matrix) and y < len(matrix):
            matrix[x][y] = 1
        else:
            # print("illegal", x, y)
            pass
        if d > 0:
            x = x + xi
            d = d + (2 * (dx - dy))
        else:
            d = d + 2 * dx


def plot_line(x0, y0, x1, y1, matrix):
    """
    plots a binary line in a binary matrix
    :param x0: starting x coordinate of line
    :param y0: starting y coordinate of line
    :param x1: ending x coordinate of line
    :param y1: ending y coordinate of line
    :param matrix: matrix to draw the line in
    :return:
    """
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            plot_line_low(x1, y1, x0, y0, matrix)
        else:
            plot_line_low(x0, y0, x1, y1, matrix)
    else:
        if y0 > y1:
            plot_line_high(x1, y1, x0, y0, matrix)
        else:
            plot_line_high(x0, y0, x1, y1, matrix)


def norm(a):
    """
    calculates norm of a vector
    :param a: the vector
    :return:
    """
    return math.sqrt(sum((x ** 2 for x in a)))
