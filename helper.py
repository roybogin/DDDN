import math
from typing import Sequence, Iterable, Tuple, List

import numpy as np
import consts


def pos_to_car_center(pos: np.ndarray, theta) -> np.ndarray:
    """
    get approximate position of the center of mass of the car from the position of the car according to pybullet (
    middle of
    back wheels)
    :param pos: position of back wheels middle
    :param theta: rotation of car
    :return: numpy array that corresponds to the center's position
    """
    return pos[:2] + consts.a_2 * np.array([np.cos(theta), np.sin(theta)])


def car_center_to_pos(pos: np.ndarray, theta) -> np.ndarray:
    """
    get the position of the of car according to pybullet (middle of back wheels) from the approximate position of the center of mass of the car
    :param pos: position of back wheels middle
    :param theta: rotation of car
    :return: numpy array that corresponds to the center's position
    """
    return pos[:2] - consts.a_2 * np.array([np.cos(theta), np.sin(theta)])


def dist(point1: Sequence[float], point2: Sequence[float]) -> float:
    """
    L2 distance between the two points
    :param point1: first point
    :param point2: second point
    :return: the distance between them
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_wall(point1, point2):
    # TODO: docstring and typing
    theta = math.atan2(point1[1] - point2[1], point1[0] - point2[0])
    ret = [(point1[0] + math.sqrt(2) * consts.epsilon * math.cos(theta - math.pi / 2),
            point1[1] + math.sqrt(2) * consts.epsilon * math.sin(theta - math.pi / 2)),
           (point1[0] + math.sqrt(2) * consts.epsilon * math.cos(theta + math.pi / 2),
            point1[1] + math.sqrt(2) * consts.epsilon * math.sin(theta + math.pi / 2)),
           (point2[0] + math.sqrt(2) * consts.epsilon * math.cos(-theta - math.pi / 2),
            point2[1] + math.sqrt(2) * consts.epsilon * math.sin(-theta - math.pi / 2)),
           (point2[0] + math.sqrt(2) * consts.epsilon * math.cos(-theta + math.pi / 2),
            point2[1] + math.sqrt(2) * consts.epsilon * math.sin(-theta + math.pi / 2)),
           (point1[0] + math.sqrt(2) * consts.epsilon * math.cos(theta - math.pi / 2),
            point1[1] + math.sqrt(2) * consts.epsilon * math.sin(theta - math.pi / 2))]
    return ret


def is_in_rect(rect: list, point: Sequence[float]) -> bool:
    """
    :param rect: a 5 point list of the rectangle, where the last point is the first point.
    :param point: the point we want to check
    :returns True if the point is inside the rectangle, false o.w.
    """
    for i in range(4):
        if orientation(rect[i], rect[i + 1], point) < 0:
            return False
    return True


def add_lists(lists: Iterable[list]) -> list:
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


def norm(vec: Sequence[float]):
    """
    calculates norm of a vector
    :param vec: the vector
    :return: the norm of the vector
    """
    return dist(vec, [0]*len(vec))


def map_index_from_pos(pos: Sequence[float]) -> Sequence[int]:
    """
    transforms a position on the map to indices in the binary matrices
    :param pos: (x,y) pair on the map
    :return: (x, y) indices that the point is contained in
    """
    indices = [int((value + consts.size_map_quarter) / consts.vertex_offset) for value in pos[:2]]
    # keep the return value within the wanted limits for edge cases
    return tuple([max(0, min(idx, int((2 * consts.size_map_quarter) // consts.vertex_offset) - 1)) for idx in indices])


def block_options(index: Sequence[int], radius: int, map_shape: Tuple[int, int], only_positives: bool = False) -> \
        List[Tuple[int, int]]:
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


def on_segment(p, q, r):
    # TODO: dicstring and typing
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False


def orientation(p, q, r):
    # TODO: dicstring and typing
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    return val


# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def do_intersect(p1, q1, p2, q2):
    # TODO: dicstring and typing
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
    if (o1 == 0) and on_segment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and on_segment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and on_segment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and on_segment(p2, q1, q2):
        return True

    # If none of the cases
    return False


def distance_between_lines(start_point1, end_point1, start_point2, end_point2):
    # TODO: dicstring and typing
    if do_intersect(start_point1, end_point1, start_point2, end_point2):
        return 0
    return min([perpendicular_distance(start_point1, start_point2, end_point2),
                perpendicular_distance(end_point1, start_point2, end_point2),
                perpendicular_distance(start_point2, start_point1, end_point1),
                perpendicular_distance(end_point2, start_point1, end_point1)])


def perpendicular_distance(point, start_point, end_point):
    # TODO: dicstring and typing
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
