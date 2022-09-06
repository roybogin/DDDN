import math
from typing import Sequence

import pybullet as p

from helper import dist


def create_wall(pos: Sequence[float], orientation: Sequence[float], length: float, width: float, client) -> int:
    """
    creating a wall on the pybullet map
    :param pos: middle point of the wall
    :param orientation: wanted orientation of the wall (quaternions)
    :param length:  length of the wall
    :param width:   width of the wall
    :param client:  pybullet client to create the wall
    :return: id of the created wall
    """
    box_half_length = length / 2
    box_half_width = width / 2
    box_half_height = 0.5
    body = client.createCollisionShape(
        p.GEOM_BOX, halfExtents=[box_half_length, box_half_width, box_half_height]
    )
    # block creation
    block = client.createMultiBody(
        10000,
        body,
        -1,
        pos,
        orientation,
    )
    return block


def create_poly_wall(poly, epsilon, client):
    # TODO: finish docstring and typing - create_poly_wall
    """
    create a polygonal chain of walls
    :param poly:
    :param epsilon:
    :param client: pybullet client to construct the walls
    :return:
    """
    walls = []
    start = 1
    if len(poly) < 2 or (len(poly) == 2 and poly[0] == poly[1]):
        print("Illegal polygonal wall")
        return
    length = dist(poly[0], poly[1]) + 2 * epsilon
    width = 2 * epsilon
    if poly[0] != poly[-1]:
        prev_angle = math.atan2(poly[1][1] - poly[0][1], poly[1][0] - poly[0][0])
        euler = [0, 0, prev_angle]
        orientation = p.getQuaternionFromEuler(euler)
        pos = [(poly[0][0] + poly[1][0]) / 2, (poly[0][1] + poly[1][1]) / 2, 0.5]
        walls.append(create_wall(pos, orientation, length, width, client))
    else:
        start = 0
        prev_angle = math.atan2(poly[-1][1] - poly[-2][1], poly[-1][0] - poly[-2][0])

    for i in range(start, len(poly) - 1):
        length = dist(poly[i], poly[i + 1]) + epsilon
        angle = math.atan2(poly[i + 1][1] - poly[i][1], poly[i + 1][0] - poly[i][0])
        diff = angle - prev_angle
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff <= -math.pi:
            diff += 2 * math.pi
        euler = [0, 0, angle]
        orientation = p.getQuaternionFromEuler(euler)

        length_from = 0
        if diff == math.pi:
            continue
        if diff == 0 or diff == math.pi / 2 or diff == -math.pi / 2:
            length_from = epsilon

        elif -math.pi / 2 < diff < math.pi / 2:
            if diff > 0:
                length_from = math.sqrt(2) * epsilon * math.cos(math.pi / 4 - diff)
            else:
                length_from = math.sqrt(2) * epsilon * math.cos(math.pi / 4 + diff)
        else:
            if diff > 0:
                length_from += epsilon / math.tan(math.pi - diff)
                length_from += epsilon / math.sin(math.pi - diff)
            else:
                length_from += epsilon / math.tan(math.pi + diff)
                length_from += epsilon / math.sin(math.pi + diff)

        length -= length_from
        pos = [
            poly[i][0] + math.cos(angle) * (length / 2 + length_from),
            poly[i][1] + math.sin(angle) * (length / 2 + length_from),
            0.5,
        ]
        walls.append(create_wall(pos, orientation, length, width, client))
        prev_angle = angle
    return walls


def create_map(in_map, epsilon, client):
    # TODO: finish docstring and typing - create_map
    """
    creates the map in pybullet
    :param in_map: the walls inside the map
    :param epsilon:
    :param client:  pybullet client to use
    :return: a list of IDs of the walls in pybullet
    """
    walls = []
    for poly in in_map:
        walls += create_poly_wall(poly, epsilon, client)
    # end = [end_point[0], end_point[1], 0.5]
    # create_wall(end, p.getQuaternionFromEuler([0, 0, 0]), epsilon / 2, epsilon / 2, client)
    return walls

