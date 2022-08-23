import heapq
from itertools import combinations
from typing import Set

import numpy as np

import consts
import scan_to_map
from helper import dist, map_index_from_pos


class Vertex:
    """
    Class to represent a graph vertex for the PRM
    """
    def __init__(self, pos: np.ndarray, theta: float):
        self.pos: np.ndarray = pos  # position of the car
        self.theta: float = theta   # angle if the car
        self.edges: Set[Edge] = set()  # the corresponding edges in the graph


class Edge:
    """
    Class to represent a graph edge for the PRM
    """
    def __init__(self, vertex_1: Vertex, vertex_2: Vertex, weight: float):
        self.v1: Vertex = vertex_1      # first vertex in the edge
        self.v2: Vertex = vertex_2      # second vertex in the edge
        self.weight: float = weight     # weight of the edge


class WeightedGraph:
    """
    Class to represent a weighted graph for the PRM
    """
    def __init__(self):
        self.vertices: Set[Vertex] = set()  # A set of the graph vertices
        self.n: int = 0                     # size of the graph

    def add_vertex(self, pos: np.ndarray, theta: float) -> Vertex:
        """
        add a vertex to the graph
        :param pos: the corresponding  position of the car
        :param theta: the corresponding ange of the car
        :return:
        """
        new_vertex = Vertex(pos, theta)
        self.vertices.add(new_vertex)
        self.n += 1
        return new_vertex

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

        self.vertices_by_blocks = {}

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

    def sample_points(self, segment_map: scan_to_map.Map, np_random, num_sample_car: int = 10):
        while self.graph.n < self.sample_amount:
            x, y = np_random.rand(2) * 2 * consts.size_map_quarter - consts.size_map_quarter
            theta = np_random.rand() * 2 * np.pi
            if segment_map.check_state(x, y, theta, self.length, self.width, num_sample_car):
                new_vertex = self.graph.add_vertex(np.array([x, y]), theta)
                block = map_index_from_pos(new_vertex.pos)
                self.vertices_by_blocks.setdefault(block, [])
                self.vertices_by_blocks[block].append(new_vertex)

    def try_add_edge(self, v_1: Vertex, v_2: Vertex, angle_matters: bool = True):
        weight = dist(v_1.pos, v_2.pos)
        if weight <= self.res:
            transformed = (self.radius_delta(-v_2.theta) * (v_2.pos - v_1.pos), v_2.theta - v_1.theta)
            x_tag, y_tag = transformed[0][0], transformed[0][1]
            differential_theta = self.theta_curve(x_tag, y_tag)
            if (not angle_matters) or abs(differential_theta - transformed[1]) < self.tol:
                if np.sqrt(self.radius_x_y_squared(x_tag, y_tag)) < self.max_radius:
                    self.graph.add_edge(v_1, v_2, weight)

    def edge_generation(self) -> None:
        """
        edge generation for non holonomic prm graph
        """
        for v_1, v_2 in combinations(self.graph.vertices, 2):
            self.try_add_edge(v_1, v_2)

    def add_vertex(self, pos: np.ndarray, theta: float) -> Vertex:
        """
        add vertex for the prm
        :param pos: corresponding position for the car
        :param theta: angle of the car
        :return:
        """
        new_vertex = self.graph.add_vertex(pos, theta)
        for vertex in self.graph.vertices:
            if vertex == new_vertex:
                continue
            self.try_add_edge(vertex, new_vertex)
        return new_vertex

    def add_end_vertex(self, pos: np.ndarray) -> Vertex:
        """
        adding the end verex to the graph, we don't care about the angle
        :param pos: position of the end
        :return: the end vertex
        """
        end_vertex = self.graph.add_vertex(pos, 0)
        for vertex in self.graph.vertices:
            if vertex == end_vertex:
                continue
            self.try_add_edge(end_vertex, vertex, False)
        return end_vertex

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
