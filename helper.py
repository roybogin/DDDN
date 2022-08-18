import itertools
import math
from typing import Sequence, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import consts


def dist(point1: consts.vector, point2: consts.vector) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def draw_binary_matrix(mat: consts.binary_matrix) -> None:
    """
    draws the given matrix with matplotlib
    :param mat: the binary matrix to draw
    :return:
    """
    cmap = ListedColormap(["b", "g"])
    matrix = np.array(mat, dtype=np.uint8)
    plt.imshow(matrix, cmap=cmap)


def add_lists(lists: List[List[float]]) -> List[float]:
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


def plot_line_low(x0: int, y0: int, x1: int, y1: int, matrix: consts.binary_matrix) -> None:
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


def plot_line_high(x0: int, y0: int, x1: int, y1: int, matrix: consts.binary_matrix) -> None:
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


def plot_line(x0: int, y0: int, x1: int, y1: int, matrix: consts.binary_matrix) -> None:
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


def norm(a: consts.vector) -> float:
    """
    calculates norm of a vector
    :param a: the vector
    :return:
    """
    return math.sqrt(sum((x ** 2 for x in a)))


def map_index_from_pos(pos: consts.vector) -> Sequence[int]:
    """
    transforms a position on the map to indices in the binary matrices
    :param pos: (x,y) pair on the map
    :return: (x, y) indices that the point is contained in
    """
    indices = [int((value + consts.size_map_quarter) / consts.block_size) for value in pos[:2]]
    # keep the return value within the wanted limits for edge cases
    return [max(0, min(idx, int((2 * consts.size_map_quarter) // consts.block_size) - 1)) for idx in indices]


def pos_from_map_index(block_index: Sequence[int]) -> consts.vector:
    """
    transforms indices in the binary matrices to a position on the map (the position is the middle of the block)
    :param block_index: index of block in the matrix (2 dimensional)
    :return: (x, y) pair to mark the position in the map
    """
    return [consts.block_size * (value + 0.5) - consts.size_map_quarter for value in block_index]


def get_neighbors(index: Sequence[int], map_shape: Tuple[int]) -> List[Sequence[int]]:
    """
    returns the neighbors of a block in the map (including diagonal)
    :param index: the index of the block in the map (list with length 2)
    :param map_shape: the shape of the map
    :return: list of the neighbors in the map
    """
    r, c = index
    rows, cols = [r], [c]  # possible rows and columns for neighbors
    if r != 0:
        rows.append(r - 1)
    if r != map_shape[0] - 1:
        rows.append(r + 1)
    if c != 0:
        cols.append(c - 1)
    if c != map_shape[1] - 1:
        cols.append(c + 1)
    neighbors = list(itertools.product(rows, cols))
    return neighbors[1:]  # the cell itself is not a neighbor


def calculate_distances(partial_map: consts.binary_matrix, index: Sequence[int]) -> consts.binary_matrix:
    """
    calculates the distance in steps to all blocks from the given location using BFS
    :param partial_map: the partial obstacle map - 0 if unexplored or empty and 1 if the block has an obstacle
    :param index: index from which we calculate the distances
    :return: a new map if the distances
    """
    index = tuple(index)
    distances = np.full(np.shape(partial_map), np.inf)
    distances[index] = 0
    blocks_to_check = []  # BFS queue
    neighbors = get_neighbors(index, distances.shape)
    for n in neighbors:
        blocks_to_check.append((n, 1))  # the neighbor and its distance
    while len(blocks_to_check) != 0:
        curr_idx, distance = blocks_to_check.pop(0)
        if partial_map[curr_idx[0]][curr_idx[1]] == 1 or distances[curr_idx] != np.inf:
            continue    # don't process walls or visited cells
        distances[curr_idx] = distance
        neighbors = get_neighbors(curr_idx, distances.shape)
        for n in neighbors:
            blocks_to_check.append((n, distance + 1))  # the neighbor and its distance
    return distances
