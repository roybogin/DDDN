import heapq
from collections import defaultdict
from typing import Set, Tuple

import numpy as np
from tqdm import tqdm

import pickle
import consts
import scan_to_map
from helper import dist, map_index_from_pos, block_options


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
        self.e: int = 0  # amount of edges in the grapp
        self.e_counter = 0


    def DFSUtil(self, temp, v, visited):
 
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for e in v.edges:
            u = e.v1
            if u == v:
                u = e.v2
            if not visited[u]:
                temp = self.DFSUtil(temp, u, visited)
        return temp
 
    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = {}
        cc = []
        for v in self.vertices:
            visited[v]=False
        for v in self.vertices:
            if not visited[v]:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc

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

        self.graph = WeightedGraph()
        self.shape = shape
        self.vertices = []
        offset = consts.block_size
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        for _ in range(shape[0]):
            self.vertices.append([])
            for _ in range(shape[1]):
                self.vertices[-1].append([])
        x_temp = offset / 2 + consts.amount_blocks_from_edge * offset
        for row_idx in tqdm(range(consts.amount_blocks_from_edge, self.shape[0] - consts.amount_blocks_from_edge)):
            y_temp = offset / 2 + consts.amount_blocks_from_edge * offset
            for col_idx in range(consts.amount_blocks_from_edge, self.shape[1] - consts.amount_blocks_from_edge):
                theta_temp = 0
                for angle_idx in range(consts.directions_per_vertex):
                    new_vertex = self.graph.add_vertex(np.array([x_temp, y_temp]), theta_temp)
                    self.vertices[row_idx][col_idx].append(new_vertex)
                    theta_temp += angle_offset
                y_temp += offset
            x_temp += offset
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

    def possible_edges(self):
        ret = []
        x = consts.size_map_quarter / consts.block_size
        y = x
        block = np.array([x, y])
        for theta in range(consts.directions_per_vertex):
            ret.append([])
            v = self.vertices[x][y][theta]
            for neighbor_block in block_options(block, np.ceil(self.res / consts.block_size), self.shape):
                for u in self.vertices[neighbor_block[0]][neighbor_block[1]]:
                    if all(v.pos == u.pos):
                        continue
                weight = dist(v.pos, u.pos)
                if weight <= self.res:
                    transformed = self.transform_pov(v, u)  # show v_2 from POV of v_1
                    x_tag, y_tag = transformed[0][0], transformed[0][1]
                    differential_theta = self.theta_curve(x_tag, y_tag)
                    if abs(differential_theta - transformed[1]) < self.tol:
                        if self.radius_x_y_squared(x_tag, y_tag) >= self.max_angle_radius ** 2:
                            ret[-1].append((u.pos[0] - v.pos[0], u.pos[1] - v.pos[1], u.theta - v.theta))
        return ret

    def generate_graph(self):
        offset = consts.block_size / consts.vertices_per_block_horizontal
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        for row_idx in tqdm(range(3, self.shape[0] - 3)):
            for col_idx in range(3, self.shape[1] - 3):
                x = offset / 2
                for i in range(consts.vertices_per_block_horizontal):
                    y = offset / 2
                    for j in range(consts.vertices_per_block_horizontal):
                        theta = 0
                        for k in range(consts.directions_per_vertex):
                            x_temp = x + col_idx * consts.block_size - consts.size_map_quarter + (offset/10) * np.cos(theta)
                            y_temp = y + row_idx * consts.block_size - consts.size_map_quarter + (offset/10) * np.sin(theta)
                            new_vertex = self.add_vertex(np.array([x_temp, y_temp]), theta, block=(row_idx, col_idx))
                            self.vertices_by_blocks[(row_idx, col_idx)].append(new_vertex)
                            theta += angle_offset
                        y += offset
                    x += offset
            if row_idx % 5 == 0:
                with open(consts.graph_file, 'wb') as f:
                    pickle.dump(self, f)

    def try_add_edge(self, v_1: Vertex, v_2: Vertex, angle_matters: bool = True):
        weight = dist(v_1.pos, v_2.pos)
        if weight <= self.res:
            transformed = self.transform_pov(v_1, v_2)  # show v_2 from POV of v_1
            x_tag, y_tag = transformed[0][0], transformed[0][1]
            differential_theta = self.theta_curve(x_tag, y_tag)
            if (not angle_matters) or abs(differential_theta - transformed[1]) < self.tol:
                if self.radius_x_y_squared(x_tag, y_tag) >= self.max_angle_radius ** 2:
                    self.graph.add_edge(v_1, v_2, weight)

    def add_vertex(self, pos: np.ndarray, theta: float, angle_matters: bool = True, block: Tuple[int, int] = None) -> \
            Vertex:
        """
        add vertex for the prm
        """
        new_vertex = self.graph.add_vertex(pos, theta)
        if block is None:
            block = map_index_from_pos(pos)
        for neighbor_block in block_options(block, np.ceil(self.res / consts.block_size), self.shape):
            for vertex in self.vertices_by_blocks[neighbor_block]:
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
        for vertex in self.graph.vertices:
            self.vertices_by_blocks[map_index_from_pos(vertex.pos)].append(vertex)

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
