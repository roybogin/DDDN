import heapq
from collections import defaultdict
from typing import Set, Tuple, List, Dict

import numpy as np
from tqdm import tqdm

import consts
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

    def __lt__(self, other):
        return self.index < other.index

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
            try:
                other_vertex.edges.remove(edge)
            except:
                pass
        try:
            self.vertices.remove(v)
        except:
                pass
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
        self.vertices: List[List[List[Vertex]]] = []
        self.end = None
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        for _ in range(shape[0]):
            self.vertices.append([])
            for _ in range(shape[1]):
                self.vertices[-1].append([])
        x_temp = consts.vertex_offset / 2 + consts.amount_vertices_from_edge * consts.vertex_offset - consts.size_map_quarter
        for row_idx in tqdm(range(consts.amount_vertices_from_edge, self.shape[0] - consts.amount_vertices_from_edge)):
            y_temp = consts.vertex_offset / 2 + consts.amount_vertices_from_edge * consts.vertex_offset - consts.size_map_quarter
            for col_idx in range(consts.amount_vertices_from_edge, self.shape[1] - consts.amount_vertices_from_edge):
                theta_temp = 0
                for angle_idx in range(consts.directions_per_vertex):
                    new_vertex = self.graph.add_vertex(np.array([x_temp, y_temp]), theta_temp)
                    self.vertices[row_idx][col_idx].append(new_vertex)
                    theta_temp += angle_offset
                y_temp += consts.vertex_offset
            x_temp += consts.vertex_offset
        self.max_angle_radius = self.radius_delta(consts.max_steer)  # radius of arc for maximum steering
        self.res = np.sqrt(self.max_angle_radius ** 2 + (self.max_angle_radius - consts.a_2) ** 2)  #
        # resolution of the path planner
        self.tol = 0.02  # tolerance of the path planner
        self.distances: Dict[Vertex, Tuple[float, Vertex]] = defaultdict(lambda: (np.inf, None))

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
        if to_root == 0:
            return np.sign(y_tag) * np.pi / 2
        val = consts.a_2 / np.sqrt(to_root)
        return np.sign(y_tag) * np.arctan(val)

    def possible_offsets_angle(self, pos: np.ndarray, angle: int):
        ret = []
        block = map_index_from_pos(pos)
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        v = self.vertices[block[0]][block[1]][angle]
        for neighbor_block in block_options(block, np.ceil(self.res / consts.vertex_offset), self.shape):
            for u in self.vertices[neighbor_block[0]][neighbor_block[1]]:
                weight = dist(v.pos, u.pos)
                if weight == 0:
                    continue
                if weight <= self.res:
                    transformed = self.transform_pov(v, u)
                    x_tag, y_tag = transformed[0][0], transformed[0][1]
                    differential_theta = self.theta_curve(x_tag, y_tag)
                    if abs(differential_theta - transformed[1]) < self.tol:
                        if self.radius_x_y_squared(x_tag, y_tag) >= self.max_angle_radius ** 2:
                            ret.append((int((u.pos[0] - v.pos[0])/consts.vertex_offset), int((u.pos[1] - v.pos[1])/consts.vertex_offset), int((u.theta - v.theta)/angle_offset)))
        return ret

    def possible_offsets(self, pos: np.ndarray):
        ret = []
        for theta in range(consts.directions_per_vertex):
            ret.append(self.possible_offsets_angle(pos, theta))
        return ret

    def generate_graph(self):
        to_add = self.possible_offsets(np.array([0, 0]))
        for theta, angle in tqdm(enumerate(to_add), total=consts.directions_per_vertex):
            for diff in angle:
                for x in range(consts.amount_vertices_from_edge, self.shape[0] - consts.amount_vertices_from_edge):
                    for y in range(consts.amount_vertices_from_edge, self.shape[0] - consts.amount_vertices_from_edge):
                        if consts.amount_vertices_from_edge <= x+diff[0] < self.shape[0] - consts.amount_vertices_from_edge and consts.amount_vertices_from_edge <= y + diff[1] < self.shape[1] - consts.amount_vertices_from_edge:
                            v1 = self.vertices[x][y][theta]
                            v2 = self.vertices[x+diff[0]][y+diff[1]][(theta+diff[2]) % consts.directions_per_vertex]
                            weight = dist(v1.pos, v2.pos)
                            self.graph.add_edge(v1, v2, weight)

    # TODO: possibly remove, not needed
    def try_add_edge(self, v_1: Vertex, v_2: Vertex, angle_matters: bool = True):
        weight = dist(v_1.pos, v_2.pos)
        if weight == 0 and angle_matters:
            return
        if weight <= self.res:
            transformed = self.transform_pov(v_1, v_2)  # show v_2 from POV of v_1
            x_tag, y_tag = transformed[0][0], transformed[0][1]
            differential_theta = self.theta_curve(x_tag, y_tag)
            if (not angle_matters) or abs(differential_theta - transformed[1]) < self.tol:
                if self.radius_x_y_squared(x_tag, y_tag) >= self.max_angle_radius ** 2:
                    self.graph.add_edge(v_1, v_2, weight)

    # TODO: possibly remove, not needed
    def add_vertex(self, pos: np.ndarray, theta: float, angle_matters: bool = True, block: Tuple[int, int] = None) -> \
            Vertex:
        """
        add vertex for the prm
        """
        new_vertex = self.graph.add_vertex(pos, theta)
        if block is None:
            block = map_index_from_pos(pos)
        for neighbor_block in block_options(block, np.ceil(self.res / consts.vertex_offset), self.shape):
            for vertex in self.vertices[neighbor_block[0]][neighbor_block[1]]:
                self.try_add_edge(new_vertex, vertex, angle_matters)
        return new_vertex

    def set_end(self, pos):
        index = map_index_from_pos(pos)
        self.end = self.graph.add_vertex(self.vertices[index[0]][index[1]][0].pos, 0)
        for v in self.vertices[index[0]][index[1]]:
            self.graph.add_edge(self.end, v, 0)

    def dijkstra(self, root: Vertex):
        self.distances[root] = (0, root)
        visited = {v: False for v in self.graph.vertices}
        pq = [(0.0, root)]
        while len(pq) > 0:
            _, u = heapq.heappop(pq)
            dist_u = self.distances[u][0]
            if visited[u]:
                continue
            visited[u] = True
            for edge in u.edges:
                weight = edge.weight
                v = edge.v1
                if v == u:
                    v = edge.v2
                dist_v = self.distances[v][0]
                if dist_u + weight < dist_v:
                    self.distances[v] = (dist_u + weight, u)
                    dist_v = dist_u + weight
                    heapq.heappush(pq, (dist_v, v))

    def get_closest_vertex(self, pos: np.ndarray, theta: float):
        block = map_index_from_pos(pos)
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        angle = round(theta / angle_offset)
        return self.vertices[block[0]][block[1]][angle]

    def next_in_path(self, pos: np.ndarray, theta: float):
        block = map_index_from_pos(pos)
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        angle = round(theta / angle_offset)
        vertex = self.vertices[block[0]][block[1]][angle]
        return self.distances[vertex][1]

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
