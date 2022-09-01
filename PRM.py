import heapq
import time
from typing import Set, List

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import consts
from consts import Direction
from WeightedGraph import Edge, WeightedGraph, Vertex
from d_star import DStar
from helper import dist, map_index_from_pos, block_options


def pos_to_car_center(pos: np.ndarray, theta) -> np.ndarray:
    return pos[:2] + consts.a_2 * np.array([np.cos(theta), np.sin(theta)])


def car_center_to_pos(pos: np.ndarray, theta) -> np.ndarray:
    return pos[:2] - consts.a_2 * np.array([np.cos(theta), np.sin(theta)])


class PRM:
    def __init__(self, shape):
        # TODO: maybe save space by combining vertices and edges in opposite directions
        self.deleted_edges: Set[Edge] = set()
        self.graph = WeightedGraph()
        self.shape = shape
        self.vertices: List[List[List[List[Vertex]]]] = []
        self.end = None
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        for _ in range(shape[0]):
            self.vertices.append([])
            for _ in range(shape[1]):
                self.vertices[-1].append([])
        x_temp = consts.vertex_offset / 2 + consts.amount_vertices_from_edge * consts.vertex_offset - consts.size_map_quarter
        for col_idx in tqdm(range(consts.amount_vertices_from_edge, self.shape[1] - consts.amount_vertices_from_edge)):
            y_temp = consts.vertex_offset / 2 + consts.amount_vertices_from_edge * consts.vertex_offset - consts.size_map_quarter
            for row_idx in range(consts.amount_vertices_from_edge, self.shape[0] - consts.amount_vertices_from_edge):
                theta_temp = 0
                for _ in range(consts.directions_per_vertex):
                    new_vertex_forward = self.graph.add_vertex(np.array([x_temp, y_temp]), theta_temp, Direction.FORWARD)
                    new_vertex_backward = self.graph.add_vertex(np.array([x_temp, y_temp]), theta_temp, Direction.BACKWARD)
                    self.vertices[col_idx][row_idx].append([])
                    self.vertices[col_idx][row_idx][-1].append(new_vertex_forward)
                    self.vertices[col_idx][row_idx][-1].append(new_vertex_backward)
                    self.graph.add_edge(new_vertex_forward, new_vertex_backward, consts.direction_change_weight)
                    self.graph.add_edge(new_vertex_backward, new_vertex_forward, consts.direction_change_weight)
                    theta_temp += angle_offset
                y_temp += consts.vertex_offset
            x_temp += consts.vertex_offset
        self.max_angle_radius = self.radius_delta(consts.max_steer)  # radius of arc for maximum steering
        self.res = np.sqrt(self.max_angle_radius ** 2 + (self.max_angle_radius - consts.a_2) ** 2)  #
        # resolution of the path planner
        self.tol = 0.02  # tolerance of the path planner
        self.d_star: DStar | None = None
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

    def possible_offsets_angle(self, pos: np.ndarray, angle: int, only_forward=False):
        ret = []
        block = map_index_from_pos(pos)
        v = self.vertices[block[0]][block[1]][angle][Direction.FORWARD]
        for neighbor_block in block_options(block, np.ceil(self.res / consts.vertex_offset), self.shape):
            for theta, u_nodir in enumerate(self.vertices[neighbor_block[0]][neighbor_block[1]]):
                u = u_nodir[Direction.FORWARD]
                weight = dist(v.pos, u.pos)
                if weight == 0:
                    continue
                # TODO: move code to new function - edges to add
                if weight <= self.res:
                    transformed = self.transform_pov(v, u)
                    x_tag, y_tag = transformed[0][0], transformed[0][1]
                    differential_theta = self.theta_curve(x_tag, y_tag)
                    if not only_forward or x_tag >= 0:
                        if abs(differential_theta - transformed[1]) < self.tol or abs(
                                2 * np.pi + differential_theta - transformed[1]) < self.tol or abs(
                                -2 * np.pi + differential_theta - transformed[1]) < self.tol:
                            if self.radius_x_y_squared(x_tag, y_tag) >= self.max_angle_radius ** 2:
                                ret.append((neighbor_block[0] - block[0], neighbor_block[1] - block[1], theta - angle))
        return ret

    def possible_offsets(self, pos: np.ndarray, only_forward=False):
        ret = []
        for theta in range(consts.directions_per_vertex):
            ret.append(self.possible_offsets_angle(pos, theta, only_forward))
        return ret

    def generate_graph(self):
        to_add = self.possible_offsets(np.array([0, 0]), True)
        for theta, angle in tqdm(enumerate(to_add), total=consts.directions_per_vertex):
            for diff in angle:
                for x in range(consts.amount_vertices_from_edge, self.shape[0] - consts.amount_vertices_from_edge):
                    for y in range(consts.amount_vertices_from_edge, self.shape[0] - consts.amount_vertices_from_edge):
                        if consts.amount_vertices_from_edge <= x + diff[0] < self.shape[
                            0] - consts.amount_vertices_from_edge and consts.amount_vertices_from_edge <= y + diff[1] < \
                                self.shape[1] - consts.amount_vertices_from_edge:
                            v1 = self.vertices[x][y][theta]
                            v2 = self.vertices[x + diff[0]][y + diff[1]][
                                (theta + diff[2]) % consts.directions_per_vertex]
                            weight = dist(v1[0].pos, v2[0].pos)
                            self.graph.add_edge(v1[Direction.FORWARD], v2[Direction.FORWARD], weight)
                            self.graph.add_edge(v2[Direction.BACKWARD], v1[Direction.BACKWARD], weight)

    def set_end(self, pos):
        index = map_index_from_pos(pos)
        self.end = self.vertices[index[0]][index[1]][0][Direction.FORWARD]
        for v_list in self.vertices[index[0]][index[1]]:
            for v in v_list:
                self.graph.add_edge(v, self.end, 0)
        return self.end.pos

    '''def dijkstra(self, root: Vertex):
        print("dijkstra")
        t1 = time.time()
        self.distances.clear()
        self.distances[root] = (0, root)
        visited = {v: False for v in self.graph.vertices}
        pq = [(0.0, root)]
        while len(pq) > 0:
            _, u = heapq.heappop(pq)
            dist_u = self.distances[u][0]
            if visited[u]:
                continue
            visited[u] = True
            for edge in u.in_edges:
                v = edge.dst
                weight = edge.weight
                dist_v = self.distances[v][0]
                if dist_u + weight < dist_v:
                    self.distances[v] = (dist_u + weight, u)
                    dist_v = dist_u + weight
                    heapq.heappush(pq, (dist_v, v))
        print(f"finished dijkstra in {time.time() - t1}")'''

    def get_closest_vertex(self, pos: np.ndarray, theta: float, direction: consts.Direction):
        block = map_index_from_pos(pos)
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        angle = round(theta / angle_offset)
        return self.vertices[block[0]][block[1]][angle][direction]

    def next_in_path(self, vertex: Vertex):
        successors = ((e.dst, e.weight) for e in vertex.out_edges)
        next_vertex_key = lambda dest, weight: self.d_star.g[dest] + weight
        try:
            next_vertex = min(successors, key=lambda tup: next_vertex_key(*tup))[0]
        except ValueError:
            print('a')
            next_vertex = vertex
        return next_vertex

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

    def draw_path(self, current_vertex: Vertex, idx=''):
        x_list = [current_vertex.pos[0]]
        y_list = [current_vertex.pos[1]]
        plt.scatter(x_list, y_list, c='black')
        vertex = current_vertex
        parent = self.next_in_path(vertex)
        while (parent != vertex) and (parent is not None):
            vertex = parent
            parent = self.next_in_path(vertex)
            x_list.append(vertex.pos[0])
            y_list.append(vertex.pos[1])
        plt.plot(x_list, y_list, label=f'projected path {idx}')
        plt.scatter(x_list[-1], y_list[-1], c='green')

    def remove_vertex(self, v: Vertex):
        index = map_index_from_pos(v.pos)
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        angle = round(v.theta / angle_offset)
        self.vertices[index[0]][index[1]][angle][v.direction] = None
        return self.graph.remove_vertex(v, self.deleted_edges)

    def remove_edge(self, e: Edge):
        return self.graph.remove_edge(e, self.deleted_edges)

    def init_d_star(self, start_vertex: Vertex):
        self.d_star = DStar(self.graph, start_vertex, self.end)




