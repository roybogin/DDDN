import heapq
from itertools import combinations
from typing import Sequence, Set

import numpy as np

import consts
import scan_to_map
from helper import pos_from_map_index, dist


class Vertex:
    def __init__(self, pos: np.ndarray, theta: float):
        self.pos: np.ndarray = pos
        self.theta: float = theta
        self.edges: Set[Edge] = set()  # the corresponding edges in the graph


class Edge:
    def __init__(self, vertex_1: Vertex, vertex_2: Vertex, weight: float):
        self.v1: Vertex = vertex_1
        self.v2: Vertex = vertex_2
        self.weight: float = weight


class WeightedGraph:
    def __init__(self):
        self.vertices: Set[Vertex] = set()
        self.n: int = 0

    def add_vertex(self, pos: np.ndarray, theta: float):
        new_vertex = Vertex(pos, theta)
        self.vertices.add(new_vertex)
        self.n += 1

    def add_edge(self, vertex_1: Vertex, vertex_2: Vertex, weight: float):
        edge = Edge(vertex_1, vertex_2, weight)
        vertex_1.edges.add(edge)
        vertex_2.edges.add(edge)

    def remove_vertex(self, v: Vertex):
        for edge in v.edges:
            other_vertex = edge.v1
            if v == other_vertex:
                other_vertex = edge.v2
            other_vertex.edges.remove(edge)
        self.vertices.remove(v)
        self.n -= 1


class PRM:
    def __init__(self):
        self.length = 0.325
        self.width = 0.2
        self.a_2 = 0.1477  # a_2 of the car

        self.sample_amount = 10000
        self.graph = WeightedGraph()

        self.max_radius = self.radius_delta(consts.max_steer)  # radius of arc for maximum steering
        self.res = 0.7 * np.sqrt(self.max_radius ** 2 + (self.max_radius - self.a_2) ** 2)  # resolution of the path
        # planner
        # TODO: fill values
        self.tol = 0  # tolerance of the path planner

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
        new_index = self.graph.n
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
                    self.graph.add_vertex(np.array([x, y]), theta)
                    count += 1
        return new_index

    def edge_generation(self) -> None:
        """
        edge generation for non holonomic prm graph
        """
        for v_1, v_2 in combinations(self.graph.vertices, 2):
            weight = dist(v_1.pos, v_2.pos)
            if weight <= self.res:
                transformed = (self.radius_delta(-v_2.theta) * (v_2.pos - v_1.pos), v_2.theta - v_1.theta)
                x, y = transformed[0], transformed[1]
                needed_theta = self.theta_curve(x, y)
                if abs(needed_theta - v_2.theta) < self.tol and \
                        np.sqrt(self.radius_x_y_squared(x, y)) < self.max_radius:
                    self.graph.add_edge(v_1, v_2, weight)

    def dijkstra(self, root: Vertex):
        distances = {v: np.inf for v in self.graph.vertices}
        distances[root] = 0
        visited = {v: False for v in self.graph.vertices}
        pq = [(0, root)]
        while len(pq) > 0:
            _, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            for edge in u.edges:
                weight = edge.weight
                v = edge.v1
                if v == u:
                    v = edge.v2
                dist_u_weight = distances[u] + weight
                dist_v = distances[v]
                if dist_u_weight < dist_v:
                    distances[v] = dist_u_weight
                    dist_v = dist_u_weight
                    heapq.heappush(pq, (dist_v, v))
        return distances
