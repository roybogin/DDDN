from typing import Set, List, Optional, Sequence, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt

import consts
import d_star
from WeightedGraph import Edge, WeightedGraph, Vertex
from d_star import DStar
from helper import dist, map_index_from_pos, block_options
from tqdm import tqdm


def radius_delta(delta: float) -> float:
    """
    radius of the path given wheel angle
    :param delta: angle of wheels relative to the car body
    :return: radius of the path
    """
    if np.tan(delta) == 0:
        return np.inf
    return np.sqrt(consts.a_2 ** 2 + (consts.length / (np.tan(delta) ** 2)))


def rotate_angle(vec: np.ndarray, alpha: float) -> np.ndarray:
    """
    rotate a vector by an angle
    :param vec: vector to rotate
    :param alpha: angle to rotate the vector by
    :return: the rotated vector
    """
    cos, sin = np.cos(alpha), np.sin(alpha)
    return np.array([[cos, -sin], [sin, cos]]) @ vec


def radius_x_y_squared(x: float, y: float) -> float:
    """
    radius of the path given another position (x, y) assuming the current location is 0, 0 and rotation 0
    :param x: x of the other position
    :param y: y of the other position
    :return: radius of the path
    """
    if y == 0:
        return np.inf
    t = (x ** 2 + 2 * consts.a_2 * x + y ** 2) / (2 * y)
    return t ** 2 + consts.a_2 ** 2


def theta_curve(x_tag: float, y_tag: float) -> float:
    """
    calculated derivative of y wrt. x on the calculated path at (0, 0) when another known point on the path is given
    :param x_tag: x of the other position
    :param y_tag: y of the other position
    :return: derivative of y wrt. x on the calculated path
    """
    if y_tag == 0:
        return 0
    to_root = radius_x_y_squared(x_tag, y_tag) - (consts.a_2 ** 2)
    if to_root == 0:
        return np.sign(y_tag) * np.pi / 2
    val = consts.a_2 / np.sqrt(to_root)
    return np.sign(y_tag) * np.arctan(val)


def transform_pov(vertex_1: Vertex, vertex_2: Vertex) -> Tuple[np.ndarray, float]:
    """
    show vertex_2 from the POV of vertex_1
    :param vertex_1: a vertex
    :param vertex_2: a vertex
    :return: the position and angle of vertex_2 from the perspective of vertex_1
    """
    return rotate_angle(vertex_2.pos - vertex_1.pos, -vertex_1.theta), vertex_2.theta - vertex_1.theta


def transform_by_values(pos: np.ndarray, theta: float, vertex_2: Vertex) -> Tuple[np.ndarray, float]:
    """
    show vertex_2 from the POV of (pos, angle)
    :param pos: position for POV
    :param theta: angle for POV
    :param vertex_2: vertex to transform
    :return: the position and angle of vertex_2 from the perspective of (pos, theta)
    """
    return rotate_angle(vertex_2.pos - pos, -theta), vertex_2.theta - theta


class PRM:
    def __init__(self, size_map_quarter, shape, prm=None):
        self.end: Optional[Vertex] = None   # The end vertex for the PRM
        self.shape: Tuple[int, int] = shape  # shape of the vertex grid (in rows and columns)
        self.size_map_quarter: float = size_map_quarter # length of half of the grid (0 to end)
        self.max_angle_radius: float = radius_delta(consts.max_steer)  # radius of arc for maximum steering
        self.res: float = np.sqrt(self.max_angle_radius ** 2 + (self.max_angle_radius - consts.a_2) ** 2)
        # resolution of the path planner
        self.tol: float = 0.02  # tolerance of the path planner
        self.d_star: Optional[DStar] = None  # D* object for path planning
        self.s_last: Optional[Vertex] = None    # last vertex visited
        if prm is not None:  # copy graph and vertex list
            self.graph: WeightedGraph = prm.graph
            self.vertices: List[List[List[Optional[Vertex]]]] = prm.vertices
        else:  # generate graph and vertex list
            self.graph = WeightedGraph()
            self.vertices: List[List[List[Optional[Vertex]]]] = []
            angle_offset = 2 * np.pi / consts.directions_per_vertex
            for _ in range(shape[0]):
                self.vertices.append([])
                for _ in range(shape[1]):
                    self.vertices[-1].append([])
            x_temp = (
                consts.vertex_offset / 2
                + consts.amount_vertices_from_edge * consts.vertex_offset
                - self.size_map_quarter
            )
            for col_idx in tqdm(
                range(
                    consts.amount_vertices_from_edge,
                    self.shape[1] - consts.amount_vertices_from_edge,
                )
            ):
                y_temp = (
                    consts.vertex_offset / 2
                    + consts.amount_vertices_from_edge * consts.vertex_offset
                    - self.size_map_quarter
                )
                for row_idx in range(
                    consts.amount_vertices_from_edge,
                    self.shape[0] - consts.amount_vertices_from_edge,
                ):
                    theta_temp = 0
                    for _ in range(consts.directions_per_vertex):
                        new_vertex = self.graph.add_vertex(
                            np.array([x_temp, y_temp]), theta_temp
                        )
                        self.vertices[col_idx][row_idx].append(new_vertex)
                        theta_temp += angle_offset
                    y_temp += consts.vertex_offset
                x_temp += consts.vertex_offset

    def possible_offsets_angle(self, pos: np.ndarray, angle: int, only_forward: bool = False) -> List[Tuple]:
        #TODO: docstring
        ret = []
        block = map_index_from_pos(pos, self.size_map_quarter)
        v = self.vertices[block[0]][block[1]][angle]
        for neighbor_block in block_options(block, np.ceil(self.res / consts.vertex_offset), self.shape):
            for theta, u in enumerate(self.vertices[neighbor_block[0]][neighbor_block[1]]):
                weight = dist(v.pos, u.pos)
                if weight == 0:
                    continue
                # TODO: move code to new function - edges to add
                if weight <= self.res:
                    transformed = transform_pov(v, u)
                    x_tag, y_tag = transformed[0][0], transformed[0][1]
                    differential_theta = theta_curve(x_tag, y_tag)
                    if not only_forward or x_tag >= 0:
                        if (
                            abs(differential_theta - transformed[1]) < self.tol
                            or abs(2 * np.pi + differential_theta - transformed[1])
                            < self.tol
                            or abs(-2 * np.pi + differential_theta - transformed[1])
                            < self.tol
                        ):
                            if (
                                radius_x_y_squared(x_tag, y_tag)
                                >= self.max_angle_radius ** 2
                            ):
                                ret.append(
                                    (
                                        neighbor_block[0] - block[0],
                                        neighbor_block[1] - block[1],
                                        theta - angle,
                                    )
                                )
        return ret

    def possible_offsets(self, pos: np.ndarray, only_forward=False):
        # TODO: docstring and typing
        ret = []
        for theta in range(consts.directions_per_vertex):
            ret.append(self.possible_offsets_angle(pos, theta, only_forward))
        return ret

    def generate_graph(self):
        """
        genere the graph by computing edges for one vertex and copying it for other vertices
        """
        to_add = self.possible_offsets(np.array([0, 0]), True)
        for theta, angle in tqdm(enumerate(to_add), total=consts.directions_per_vertex):
            for diff in angle:
                for x in range(consts.amount_vertices_from_edge, self.shape[0] - consts.amount_vertices_from_edge):
                    for y in range(consts.amount_vertices_from_edge, self.shape[0] - consts.amount_vertices_from_edge):
                        if (
                            consts.amount_vertices_from_edge
                                <= x + diff[0]
                                < self.shape[0] - consts.amount_vertices_from_edge
                                and consts.amount_vertices_from_edge
                                <= y + diff[1]
                                < self.shape[1] - consts.amount_vertices_from_edge
                        ):
                            v1 = self.vertices[x][y][theta]
                            v2 = self.vertices[x + diff[0]][y + diff[1]][(theta + diff[2]) % consts.directions_per_vertex]
                            weight = dist(v1.pos, v2.pos)
                            self.graph.add_edge(v1, v2, weight)

    def set_end(self, pos: Sequence[int]):
        """
        sets the end position of the car (rounded to nearest vertex) and connect edges to ignore rotation at end
        :param pos: end position of the car
        :return: the rounded end position
        """
        index = map_index_from_pos(pos, self.size_map_quarter)
        self.end = self.get_closest_vertex(pos, 0)
        for v in self.vertices[index[0]][index[1]]:
            if v != self.end:
                self.graph.add_edge(v, self.end, 0)
        return self.end.pos

    def get_closest_vertex(self, pos: np.ndarray | Sequence[float], theta: float) -> Vertex:
        """
        gets the closest vertex to a given position and rotation
        :param pos: given position
        :param theta: given rotation
        :return: the closest vertex to a given position and rotation
        """
        block = map_index_from_pos(pos, self.size_map_quarter)
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        angle = round(theta / angle_offset)
        return self.vertices[block[0]][block[1]][angle]

    def next_in_path(self, vertex: Vertex):
        successors = ((e.dst, e.weight) for e in vertex.out_edges if e.weight != np.inf)
        next_vertex_key = lambda dest, weight: self.d_star.g[dest] + weight
        next_vertex = min(
            successors, key=lambda tup: next_vertex_key(*tup), default=(None,)
        )[0]
        if next_vertex is None:
            if consts.debugging:
                print("no successors")
        return next_vertex

    def draw_path(self, current_vertex: Vertex, idx: Any = ""):
        """
        draw the planned path of the car from an initial vertex
        :param current_vertex: vertex to start the planning from
        :param idx: index of the car to add to legend
        """
        x_list = [current_vertex.pos[0]]
        y_list = [current_vertex.pos[1]]

        plt.scatter(x_list, y_list, label=f"start {idx}")
        vertex = current_vertex
        parent = self.next_in_path(vertex)
        while (parent != self.end) and (parent is not None):
            vertex = parent
            if parent != self.end:
                parent = self.next_in_path(vertex)
            x_list.append(vertex.pos[0])
            y_list.append(vertex.pos[1])
        plt.scatter(x_list[-1], y_list[-1], label=f"end goal {idx}")
        plt.plot(x_list, y_list, label=f"projected {idx}")

    def remove_vertex(self, v: Vertex):
        """
        remove vertex from the graph
        :param v: vertex to be removed
        """
        index = map_index_from_pos(v.pos, self.size_map_quarter)
        angle_offset = 2 * np.pi / consts.directions_per_vertex
        angle = round(v.theta / angle_offset)
        self.vertices[index[0]][index[1]][angle] = None
        self.graph.remove_vertex(v)

    def remove_edge(self, e: Edge):
        """
        remove edge from the graph
        :param e: edge to be removed
        """
        self.graph.remove_edge(e)

    def init_d_star(self, start_vertex: Vertex):
        """
        initialize D* algorithm
        :param start_vertex: starting vertex of he algorithm
        """
        self.d_star = DStar(self.graph, start_vertex, self.end)
        self.s_last = start_vertex

    def update_d_star(self, edge_set: Set[Edge], current_vertex: Vertex):
        """
        code to run when updated edges need to be accounted by the D* (assuming weight changed fr original to
        infinity or the opposite direction)
        :param edge_set: edges that wre changed
        :param current_vertex: current vertex of the car
        """
        # assumes edges weights have already changed
        self.d_star.k_m += d_star.h(self.s_last, self.end)
        self.s_last = current_vertex
        for edge in edge_set:
            u, v, curr_weight = edge.src, edge.dst, edge.weight
            old_weight = np.inf
            if not edge.active:
                curr_weight = np.inf
            if curr_weight == np.inf:
                old_weight = edge.original_weight
            rhs = self.d_star.rhs
            g = self.d_star.g
            # TODO: maybe add stuff to edge and support an edge returning to a state before updating and we skip the
            #  calculation
            if old_weight > curr_weight:
                rhs[u] = min(rhs[u], curr_weight + g[v])
            elif rhs[u] == old_weight + g[v]:
                if u != self.end:
                    possible_rhs = (e.weight + g[e.dst] for e in u.out_edges)
                    rhs[u] = min(possible_rhs, default=np.inf)
            self.d_star.update_vertex(u)
