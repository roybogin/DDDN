import heapq
from collections import defaultdict
from typing import Set

import numpy as np
from tqdm import tqdm

import consts
import scan_to_map
from helper import dist, map_index_from_pos, get_neighbors


def pos_to_car_center(pos: np.ndarray, theta) -> np.ndarray:
    return pos[:2] + consts.a_2 * np.array([np.cos(theta), np.sin(theta)])


def car_center_to_pos(pos: np.ndarray, theta) -> np.ndarray:
    return pos[:2] - consts.a_2 * np.array([np.cos(theta), np.sin(theta)])


class Vertex:
    """
    Class to represent a graph vertex for the PRM
    """

    def __init__(self, pos: np.ndarray, theta: float, index: int):
        self.pos: np.ndarray = pos  # position of the car
        self.theta: float = theta  # angle if the car
        self.edges: Set[Edge] = set()  # the corresponding edges in the graph
        self.index = index

    def __getstate__(self):
        neighbors = []
        for edge in self.edges:
            neighbor = edge.v1
            if neighbor == self:
                neighbor = edge.v2
            neighbors.append((neighbor.index, edge.weight))
        pos_in_normal = car_center_to_pos(self.pos, self.theta)
        return {
            'pos': list(pos_in_normal),
            'theta': self.theta,
            'index': self.index,
            'edges': neighbors
        }


class Edge:
    """
    Class to represent a graph edge for the PRM
    """

    def __init__(self, vertex_1: Vertex, vertex_2: Vertex, weight: float):
        self.v1: Vertex = vertex_1  # first vertex in the edge
        self.v2: Vertex = vertex_2  # second vertex in the edge
        self.weight: float = weight  # weight of the edge


class WeightedGraph:
    """
    Class to represent a weighted graph for the PRM
    """

    def __init__(self):
        self.vertices: Set[Vertex] = set()  # A set of the graph vertices
        self.n: int = 0  # size of the graph
        self.e: int = 0  # amount of edges in the graph

    def add_vertex(self, pos: np.ndarray, theta: float, index: int = None) -> Vertex:
        """
        add a vertex to the graph
        :param pos: the corresponding position of the car - middle of rear wheels
        :param theta: the corresponding ange of the car
        :return:
        """
        if index is None:
            index = self.n
        new_vertex = Vertex(pos_to_car_center(pos, theta), theta, index)
        self.vertices.add(new_vertex)
        self.n += 1
        return new_vertex

    def add_edge(self, vertex_1: Vertex, vertex_2: Vertex, weight: float):
        edge = Edge(vertex_1, vertex_2, weight)
        vertex_1.edges.add(edge)
        vertex_2.edges.add(edge)
        self.e += 1

    def remove_vertex(self, v: Vertex):
        for edge in v.edges:
            other_vertex = edge.v1
            if v == other_vertex:
                other_vertex = edge.v2
            other_vertex.edges.remove(edge)
        self.vertices.remove(v)
        self.n -= 1

    def __getstate__(self):
        return [v.__getstate__() for v in self.vertices]

    def __setstate__(self, state):
        vertex_list = [None for _ in state]
        for s in tqdm(state):
            v = self.add_vertex(np.array(s['pos']), s['theta'], s['index'])
            vertex_list[s['index']] = v
            for index, weight in s['edges']:
                if vertex_list[index] is not None:
                    self.add_edge(v, vertex_list[index], weight)
        self.vertices = set(vertex_list)


class PRM:
    def __init__(self, shape):

        self.sample_amount = 1e6
        self.graph = WeightedGraph()
        self.shape = shape
        self.vertices_by_blocks = defaultdict(lambda: [])

        self.max_angle_radius = self.radius_delta(consts.max_steer)  # radius of arc for maximum steering
        self.res = 0.9 * np.sqrt(self.max_angle_radius ** 2 + (self.max_angle_radius - consts.a_2) ** 2)  #
        # resolution of the path planner
        self.tol = 0.02  # tolerance of the path planner
        self.distances = defaultdict(lambda: np.inf)

    def radius_delta(self, delta: float):
        if np.tan(delta) == 0:
            return np.inf
        return np.sqrt(consts.a_2 ** 2 + (consts.length / (np.tan(delta) ** 2)))

    def rotate_angle(self, vec: np.ndarray, alpha: float):
        cos, sin = np.cos(alpha), np.sin(alpha)
        return np.array([[cos, -sin], [sin, cos]]) @ vec

    def radius_x_y_squared(self, x, y):
        if y == 0:
            return np.inf
        t = (x ** 2 + 2 * consts.a_2 * x + y ** 2) / (2 * y)
        return t ** 2 + consts.a_2 ** 2

    def theta_curve(self, x_tag, y_tag):
        if y_tag == 0:
            return 0
        to_root = self.radius_x_y_squared(x_tag, y_tag) - (consts.a_2 ** 2)
        val = consts.a_2 / np.sqrt(to_root)
        return np.sign(y_tag) * np.arctan(val)

    def sample_points(self, segment_map: scan_to_map.Map, np_random, num_sample_car: int = 3):
        while self.graph.n < self.sample_amount:
            x, y = np_random.rand(2) * 2 * consts.size_map_quarter - consts.size_map_quarter
            theta = np_random.rand() * 2 * np.pi
            if segment_map.check_state(x, y, theta, num_sample_car):
                new_vertex = self.graph.add_vertex(np.array([x, y]), theta)
                block = map_index_from_pos(new_vertex.pos)
                self.vertices_by_blocks[block].append(new_vertex)

    def try_add_edge(self, v_1: Vertex, v_2: Vertex, angle_matters: bool = True):
        weight = dist(v_1.pos, v_2.pos)
        if weight <= self.res:
            transformed = self.transform_pov(v_1, v_2)  # show v_2 from POV of v_1
            x_tag, y_tag = transformed[0][0], transformed[0][1]
            differential_theta = self.theta_curve(x_tag, y_tag)
            if (not angle_matters) or abs(differential_theta - transformed[1]) < self.tol:
                if self.radius_x_y_squared(x_tag, y_tag) >= self.max_angle_radius ** 2:
                    self.graph.add_edge(v_1, v_2, weight)

    def edge_generation(self) -> None:
        """
        edge generation for non holonomic prm graph
        """
        for v_1 in tqdm(self.graph.vertices):
            for block in get_neighbors(map_index_from_pos(v_1.pos), self.shape, True):
                for v_2 in self.vertices_by_blocks[block]:
                    if v_1 == v_2:
                        continue
                    self.try_add_edge(v_1, v_2)

    def add_vertex(self, pos: np.ndarray, theta: float, angle_matters: bool = True) -> Vertex:
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
            self.try_add_edge(new_vertex, vertex, angle_matters)
        return new_vertex

    def dijkstra(self, root: Vertex):
        self.distances[root] = 0
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
                dist_u_weight = self.distances[u] + weight
                dist_v = self.distances[v]
                if dist_u_weight < dist_v:
                    self.distances[v] = dist_u_weight
                    dist_v = dist_u_weight
                    heapq.heappush(pq, (dist_v, v))

    def next_in_path(self, root: Vertex):
        min_dist = self.distances[root]
        best_neighbor = None
        for edge in root.edges:
            weight = edge.weight
            v = edge.v1
            if v == root:
                v = edge.v2

            dist_v = self.distances[v] + weight
            if dist_v <= min_dist:
                min_dist = dist_v
                best_neighbor = v
        self.distances[root] = min_dist
        return best_neighbor

    def __getstate__(self):
        return self.shape, self.graph.__getstate__()

    def __setstate__(self, state):
        self.__init__(state[0])
        self.graph.__setstate__(state[1])

    def transform_pov(self, vertex_1: Vertex, vertex_2: Vertex):
        """
        show vertex_2 from the POV of vertex_1
        :param vertex_1:
        :param vertex_2:
        :return:
        """
        return self.rotate_angle(vertex_2.pos - vertex_1.pos, -vertex_1.theta), vertex_2.theta - vertex_1.theta

    def transform_by_values(self, pos: np.ndarray, theta: float, vertex_2: Vertex):
        return self.rotate_angle(vertex_2.pos - pos, -theta), vertex_2.theta - theta
