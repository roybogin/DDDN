from typing import Set

import numpy as np

import consts


class Vertex:
    """
    Class to represent a graph vertex for the PRM
    """

    def __init__(self, pos: np.ndarray, theta: float, index: int):
        self.pos: np.ndarray = pos  # position of the car
        self.theta: float = theta  # angle if the car
        self.in_edges: Set[Edge] = set()  # the corresponding edges in the graph
        self.out_edges: Set[Edge] = set()  # the corresponding edges in the graph
        self.index = index

    def __lt__(self, other):
        return self.index < other.index


class Edge:
    """
    Class to represent a graph edge for the PRM
    """

    def __init__(self, vertex_1: Vertex, vertex_2: Vertex, weight: float):
        self.src: Vertex = vertex_1  # vertex that edge exits from
        self.dst: Vertex = vertex_2  # vertex that edge enters
        self.weight: float = weight  # weight of the edge
        self.original_weight: float = weight  # original edge weight for temporary changes
        self.parked_cars: int = 0  # amount of cars parked near this edge
        self.active = True


class WeightedGraph:
    """
    Class to represent a weighted graph for the PRM
    """

    def __init__(self):
        self.vertices: Set[Vertex] = set()  # A set of the graph vertices
        self.n: int = 0  # size of the graph
        self.e: int = 0  # amount of edges in the graph
        self.deleted_edges: Set[Edge] = set()

    def add_vertex(self, pos: np.ndarray, theta: float, index: int = None) -> Vertex:
        """
        add a vertex to the graph
        :param pos: the corresponding position of the car - middle of rear wheels
        :param theta: the corresponding ange of the car
        :return:
        """
        if index is None:
            index = self.n
        # new_vertex = Vertex(pos_to_car_center(pos, theta), theta, index)    # TODO: need to fix
        new_vertex = Vertex(pos, theta, index)
        self.vertices.add(new_vertex)
        self.n += 1
        return new_vertex

    def add_edge(self, vertex_1: Vertex, vertex_2: Vertex, weight: float):
        edge = Edge(vertex_1, vertex_2, weight)
        vertex_1.out_edges.add(edge)
        vertex_2.in_edges.add(edge)
        self.e += 1

    def remove_edge(self, edge: Edge):
        removed_edge = True
        edge.active = False
        v1, v2 = edge.src, edge.dst
        if edge in v1.out_edges:
            v1.out_edges.remove(edge)
        else:
            removed_edge = False
        if edge in v2.in_edges:
            v2.in_edges.remove(edge)
        else:
            removed_edge = False
        if removed_edge:
            self.e -= 1
            self.deleted_edges.add(edge)
            edge.weight = np.inf
        else:
            if consts.debugging:
                print('false edge')  # just to be sure that there is no error

    def remove_vertex(self, v: Vertex):
        if v not in self.vertices:
            if consts.debugging:
                print('false vertex')  # just to be sure that there is no error
            return
        for edge in v.in_edges:
            other_vertex = edge.src
            if edge in other_vertex.out_edges:
                other_vertex.out_edges.remove(edge)
                self.e -= 1
                self.deleted_edges.add(edge)
                edge.weight = np.inf
        v.in_edges.clear()  # not needed but just in case
        for edge in v.out_edges:
            other_vertex = edge.dst
            if edge in other_vertex.in_edges:
                other_vertex.in_edges.remove(edge)
                self.e -= 1
                self.deleted_edges.add(edge)
                edge.weight = np.inf
        v.out_edges.clear()  # not needed but just in case

        self.vertices.remove(v)
        self.n -= 1
