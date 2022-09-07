import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import consts


def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_wall(point1, point2, width):
    """
    function return a wall with width around
    :param point1:
    :param point2:
    :param width:
    :return:
    """
    theta = math.atan2(point1[1] - point2[1], point1[0] - point2[0])
    ret = [(point1[0] + math.sqrt(2) * width * math.cos(theta - math.pi / 2),
            point1[1] + math.sqrt(2) * width * math.sin(theta - math.pi / 2)),
           (point1[0] + math.sqrt(2) * width * math.cos(theta + math.pi / 2),
            point1[1] + math.sqrt(2) * width * math.sin(theta + math.pi / 2)),
           (point2[0] + math.sqrt(2) * width * math.cos(-theta - math.pi / 2),
            point2[1] + math.sqrt(2) * width * math.sin(-theta - math.pi / 2)),
           (point2[0] + math.sqrt(2) * width * math.cos(-theta + math.pi / 2),
            point2[1] + math.sqrt(2) * width * math.sin(-theta + math.pi / 2)),
           (point1[0] + math.sqrt(2) * width * math.cos(theta - math.pi / 2),
            point1[1] + math.sqrt(2) * width * math.sin(theta - math.pi / 2))]
    return ret


def is_in_rect(rect, point):
    """
    :param rect: a 5 point list of the rectanle, where the last point is the first point.
    :param point:  the point we want to check
    :returns True if the point is inide of the rectangle, false o.w.
    """
    for i in range(4):
        if orientation(rect[i], rect[i + 1], point) < 0:
            return False
    return True


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
    :return: list of the new indices that were turned to 1
    """
    new_ones = []
    for x, y in get_line(x0, y0, x1, y1, len(matrix)):
        if matrix[x][y] == 0:
            matrix[x][y] = 1
            new_ones.append((x, y))
    return new_ones


def norm(a):
    """
    calculates norm of a vector
    :param a: the vector
    :return:
    """
    return math.sqrt(sum((x ** 2 for x in a)))


def map_index_from_pos(pos, size_map_quarter):
    """
    transforms a position on the map to indices in the vertex list
    :param pos: (x,y) pair on the map
    :param size_map_quarter: length of half of the map
    :return: (x, y) indices that the point is contained in
    """
    indices = [int((value + size_map_quarter) / consts.vertex_offset) for value in pos[:2]]
    # keep the return value within the wanted limits for edge cases
    return tuple([max(0, min(idx, int((2 * size_map_quarter) // consts.vertex_offset) - 1)) for idx in indices])

def block_options(index, radius, map_shape, only_positives=False):
    """
    returns the neighbors of a block in the map (including diagonal)
    :param index: the index of the block in the map (list with length 2)
    :param radius: the radius around the block we will return
    :param map_shape: the shape of the map
    :param only_positives: indicates if we only want values lexicographically larger
    :return: list of the neighbors in the map
    """
    radius = int(radius)
    c, r = index
    padding = consts.amount_vertices_from_edge
    if c < padding or c >= map_shape[1] - padding or r < padding or r >= map_shape[0] - padding:
        return []
    neighbors = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if padding <= r + y < map_shape[0] - padding and padding <= c + x < map_shape[1] - padding:
                if x * x + y * y <= radius * radius + radius:
                    neighbors.append((c + x, r + y))
    if only_positives:
        neighbors = [n for n in neighbors if n >= (c, r)]
    return neighbors


def onSegment(p, q, r):
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    return val


# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = np.sign(orientation(p1, q1, p2))
    o2 = np.sign(orientation(p1, q1, q2))
    o3 = np.sign(orientation(p2, q2, p1))
    o4 = np.sign(orientation(p2, q2, q1))

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    # If none of the cases
    return False


def distance_between_lines(start_point1, end_point1, start_point2, end_point2):
    if doIntersect(start_point1, end_point1, start_point2, end_point2):
        return 0
    return min([perpendicularDistance(start_point1, start_point2, end_point2),
                perpendicularDistance(end_point1, start_point2, end_point2),
                perpendicularDistance(start_point2, start_point1, end_point1),
                perpendicularDistance(end_point2, start_point1, end_point1)])


def perpendicularDistance(point, start_point, end_point):
    x0, y0 = start_point
    x1, y1 = end_point
    x, y = point
    def_val = min(dist(point, start_point), dist(point, end_point))
    if x0 == x1:
        if min(y0, y1) <= y <= max(y0, y1):
            return abs(x - x0)
        return def_val
    if y0 == y1:
        if min(x0, x1) <= x <= max(x0, x1):
            return abs(y - y0)
        return def_val
    m = (y0 - y1) / (x0 - x1)
    inter_x = (y - y0 + m * x0 + x / m) / (m + 1 / m)
    inter_y = y0 + m * (inter_x - x0)
    if min(x0, x1) <= inter_x <= max(x0, x1):
        return math.sqrt((x - inter_x) ** 2 + (y - inter_y) ** 2)
    return def_val