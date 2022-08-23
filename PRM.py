import heapq
from typing import Sequence

import numpy as np

import consts
import scan_to_map
from helper import pos_from_map_index, dist


class Vertex:
    def __init__(self, pos, theta, index):
        self.pos = pos
        self.index = index
        self.theta = theta


class PRM:
    def __init__(self):
        self.length = 0.325
        self.width = 0.2
        self.a_2 = 0.1477  # a_2 of the car

        self.sample_amount = 10000
        self.graph = []
        self.vertices = []

        # TODO: fill values
        self.res = 0  # resolution of the path planner
        self.tol = 0  # tolerance of the path planner
        self.max_radius = 0  # maximum radius of arc possible for the car

    def radius_delta(self, delta: float):
        return np.sqrt(self.a_2 ** 2 + (self.length / (np.tan(delta) ** 2)))

    def radius_x_y_squared(self, x, y):
        t = (x ** 2 + 2 * self.a_2 * x + y ** 2) / (2 * y ** 2)
        return t ** 2 + self.a_2 ** 2

    def theta_curve(self, x, y):
        if y == 0:
            return 0
        val = (x + self.a_2) / np.sqrt(self.radius_x_y_squared(x, y) - (x + self.a_2) ** 2)
        return np.sign(y) * np.arctan(val)

    def sample_points(self, segment_map: scan_to_map.Map, np_random, indices: Sequence[Sequence[int]],
                      num_sample_car: int = 10):
        new_index = len(self.vertices)
        samples_block_count = self.sample_amount * int((2 * consts.size_map_quarter) // consts.block_size)
        for index in indices:
            count = 0
            block_x, block_y = pos_from_map_index(index)
            block_x -= consts.block_size / 2
            block_y -= consts.block_size / 2
            while count < samples_block_count:
                x, y = np_random.rand(2) * consts.block_size
                x += block_x
                y += block_y
                theta = np_random.rand() * 2 * np.pi
                to_check = []
                for i in range(num_sample_car):
                    for j in range(num_sample_car):
                        x_temp = self.width * (1 / 2 + i / (num_sample_car - 1))
                        y_temp = self.length * (1 / 2 + j / (num_sample_car - 1))
                        to_check.append(
                            (x + x_temp * np.cos(theta) - y_temp * np.sin(theta),
                             y + x_temp * np.sin(theta) + y_temp * np.cos(theta)))
                        # TODO: check confuse between x and y with angle
                if segment_map.check_batch(to_check):
                    new_vertex = Vertex(np.array([x, y]), theta, len(self.vertices))
                    self.vertices.append(new_vertex)
                    count += 1
        return new_index

    def edge_generation(self) -> None:
        """
        edge generation for non holonomic prm graph
        """
        for index in range(len(self.vertices)):
            vertex = self.vertices[index]
            for index_2 in range(index+1, len(self.vertices)):
                vertex_2 = self.vertices[index_2]
                pos, theta = vertex_2.pos, vertex_2.theta
                weight = dist(vertex[0], pos)
                if weight <= self.res:
                    transformed = (self.radius_delta(-theta) * (pos - vertex.pos), theta - vertex.theta)
                    x, y = transformed[0], transformed[1]
                    needed_theta = self.theta_curve(x, y)
                    if abs(needed_theta - theta) < self.tol and \
                            np.sqrt(self.radius_x_y_squared(x, y)) < self.max_radius:
                        self.graph[index].append((index_2, weight))
                        self.graph[index_2].append((index, weight))

    def dijkstra(self, root):
        n = len(self.vertices)
        distances = [np.inf for _ in range(n)]
        distances[root] = 0
        visited = [False for _ in range(n)]
        pq = [(0, root)]
        while len(pq) > 0:
            _, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            for v, l in self.graph[u]:
                if distances[u] + l < distances[v]:
                    distances[v] = distances[u] + l
                    heapq.heappush(pq, (distances[v], v))
        return distances
