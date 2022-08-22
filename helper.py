import heapq
import itertools
import math
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import consts
import scan_to_map

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


def plot_line_low(x0, y0, x1, y1, length):
    """
    helper function to plot a binary line in a binary matrix
    """
    returned = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    d = (2 * dy) - dx
    y = y0

    for x in range(x0, x1 + 1):
        if x < length and y < length:
            returned.append((x, y))
        else:
            pass
            # print("illegal", x, y)
        if d > 0:
            y = y + yi
            d = d + (2 * (dy - dx))
        else:
            d = d + 2 * dy
    return returned


def plot_line_high(x0, y0, x1, y1, length):
    """
    helper function to plot a binary line in a binary matrix
    """
    returned = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx

    d = (2 * dx) - dy
    x = x0

    for y in range(y0, y1 + 1):
        if x < length and y < length:
            returned.append((x, y))
        else:
            # print("illegal", x, y)
            pass
        if d > 0:
            x = x + xi
            d = d + (2 * (dx - dy))
        else:
            d = d + 2 * dx
    return returned


def get_line(x0, y0, x1, y1, length):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plot_line_low(x1, y1, x0, y0, length)
        else:
            return plot_line_low(x0, y0, x1, y1, length)
    else:
        if y0 > y1:
            return plot_line_high(x1, y1, x0, y0, length)
        else:
            return plot_line_high(x0, y0, x1, y1, length)


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
    for x, y in get_line(x0, y0, x1, y1, len(matrix)):
        matrix[x][y] = 1


def norm(a):
    """
    calculates norm of a vector
    :param a: the vector
    :return:
    """
    return math.sqrt(sum((x ** 2 for x in a)))


def map_index_from_pos(pos):
    """
    transforms a position on the map to indices in the binary matrices
    :param pos: (x,y) pair on the map
    :return: (x, y) indices that the point is contained in
    """
    indices = [int((value + consts.size_map_quarter) / consts.block_size) for value in pos[:2]]
    # keep the return value within the wanted limits for edge cases
    return [max(0, min(idx, int((2 * consts.size_map_quarter) // consts.block_size) - 1)) for idx in indices]


def pos_from_map_index(block_index):
    """
    transforms indices in the binary matrices to a position on the map (the position is the middle of the block)
    :param block_index: index of block in the matrix (2 dimensional)
    :return: (x, y) pair to mark the position in the map
    """
    return [consts.block_size * (value + 0.5) - consts.size_map_quarter for value in block_index]


def get_neighbors(index, map_shape):
    """
    returns the neighbors of a block in the map (including diagonal)
    :param index: the index of the block in the map (list with length 2)
    :param map_shape: the shape of the map
    :return: list of the neighbors in the map
    """

    r, c = index
    if r < 0 or r >= map_shape[0] or c < 0 or c >= map_shape[1]:
        return []
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


def get_by_direction(index, map_shape, direction, distance):
    ray_end = [int(index[0] + distance * np.sin(direction)), int(index[1] + distance * np.cos(direction))]
    line = get_line(index[0], index[1], ray_end[0], ray_end[1], map_shape[0])
    line.sort(key=lambda a: (a[0] - index[0]) ** 2 + (a[1] - index[1]) ** 2)
    return line[1:]


def calculate_distances(partial_map, index):
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
            continue  # don't process walls or visited cells
        distances[curr_idx] = distance
        neighbors = get_neighbors(curr_idx, distances.shape)
        for n in neighbors:
            blocks_to_check.append((n, distance + 1))  # the neighbor and its distance
    return distances


def radius_delta(delta: float, length: float, length, precentage_of_length: float = 1.0):
    return np.sqrt((length*precentage_of_length/2) ** 2 + (l/np.tan(delta)) ** 2)


def radius_x_y_squared(x, y):
    t = (x ** 2 + 2 * consts.a_2 * x + y ** 2) / (2 * y ** 2)
    return t**2 + consts.a_2 ** 2


def theta_curve(x, y):
    if y == 0:
        return 0
    val = (x + consts.a_2) / np.sqrt(radius_x_y_squared(x, y) - (x+consts.a_2)**2)
    return np.sign(y) * np.arctan(val)

def sample_points(N: int, width: float, length: float, num_sample_car: int = 10)
    samples = []
    while len(samples) < N:
        x, y = np.random.rand(2) * 2 * consts.size_map_quarter - consts.size_map_quarter
        theta = np.random.rand() * 2 * np.pi
        to_check = []
        for i in range(num_sample_car):
            for j in range(num_sample_car):
                x_temp = width *(-1/2 + i/(num_sample_car-1))
                y_temp = length*(-1/2 + j/(num_sample_car-1))
                to_check.append((x_temp * np.cos(theta) - y_temp * np.sin(theta), x_temp * np.sin(theta) + y_temp * np.cos(theta)))
        if scan_to_map.check_batch(to_check)
            samples.append((np.array([x, y]), theta))
    return samples


def edge_generation(vertices: List, res: float, tol: float, max_radius: float, length: float, precentage_of_length: float: 1.0) -> List[List[int]]:
    """
    edge generation for non holonomic prm grapg
    :param vertices: Vertices of the graph ([x,y],theta) - [x,y] is numpy array
    :param res: resolution of the path planner
    :param tol: tolerance of the path planner
    :return: connection graph for the path
    """
    graph = [[] for _ in range(len(vertices))]  # initialize an empty graph
    for index, vertex in enumerate(vertices):
        options = []
        for index_2, vertex_2 in enumerate(vertices):
            if index <= index_2:
                continue
            pos, theta = vertex_2
            weight = dist(vertex[0], pos)
            if weight <= res:
                transformed = (radius_delta(-theta) * (pos - vertex[0]), theta - vertex[1])
                x, y = transformed[0], transformed[1]
                needed_theta = theta_curve(x, y)
                if abs(needed_theta - theta) < tol and np.sqrt(radius_x_y_squared(x, y)) < max_radius:
                    graph[index].append((index_2, weight))
                    graph[index_2].append((index, weight))
    return graph


def dijkstra(graph, root):
    n = len(graph)
    distances = [np.inf for _ in range(n)]
    distances[root] = 0
    visited = [False for _ in range(n)]
    pq = [(0, root)]
    while len(pq) > 0:
        _, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        for v, l in graph[u]:
            if distances[u] + l < distances[v]:
                distances[v] = distances[u] + l
                heapq.heappush(pq, (distances[v], v))
    return distances
