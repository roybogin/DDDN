import heapq
import itertools
from collections import defaultdict
from typing import Tuple, DefaultDict, Any

import numpy as np

import consts
from WeightedGraph import Vertex, WeightedGraph
from helper import dist


class PriorityQueue:
    """
    a class for a priority queue
    """
    def __init__(self):
        self.queue = []  # the actual queue
        self.entry_dict = {}  # dictionary to map object to its heap instance
        self.count = itertools.count()  # counting value to sort equalities

    def insert(self, obj: Any, priority: Any):
        """
        insert the object to the queue
        :param obj: the object to insert
        :param priority:
        :return:
        """
        entry = [priority, next(self.count), obj]
        self.entry_dict[obj] = entry
        heapq.heappush(self.queue, entry)

    def remove(self, obj: Any):
        """
        remove object from the queue (nullifies it and removes later)
        :param obj: object to remove
        """
        self.entry_dict.pop(obj)[-1] = None

    def update(self, obj, new_priority):
        """
        update the priority of an object
        :param obj: object to update
        :param new_priority: new priority for the object
        """
        self.remove(obj)
        self.insert(obj, new_priority)

    def pop(self) -> Any:
        """
        remove object with the smallest priority (min of the heap)
        :return: the matching object
        """
        while len(self.queue) != 0:
            priority, _, obj = heapq.heappop(self.queue)
            if obj is not None:
                del self.entry_dict[obj]
                return obj
        if consts.debugging:
            print('empty queue')
        return None

    def top(self) -> Any:
        """
        return top of heap
        :return: top of heap
        """
        while len(self.queue) != 0:
            priority, _, obj = self.queue[0]
            if obj is not None:
                return obj
            else:
                heapq.heappop(self.queue)
        if consts.debugging:
            print('empty queue')
        return None

    def top_key(self) -> Any:
        """
        return top of heap
        :return: key for top of heap
        """
        while len(self.queue) != 0:
            priority, _,  obj = self.queue[0]
            if obj is not None:
                return priority
            else:
                heapq.heappop(self.queue)
        if consts.debugging:
            print('empty queue')
        return np.inf, np.inf

    def __contains__(self, item):
        return item in self.entry_dict


def h(v_1: Vertex, v_2: Vertex) -> float:
    """
    heuristic function that must obey triangle inequality with the edge between the vertices
    we chose distance
    :param v_1: first vertex
    :param v_2: second vertex
    :return: distance between position
    """
    return dist(v_1.pos, v_2.pos)


class DStar:
    # D* lite algorithm that we implement - code is inferred from the psuedo-code in the article
    # https://www.cs.cmu.edu/~maxim/files/dlite_tro05.pdf
    def __init__(self, graph: WeightedGraph, start_vertex: Vertex, goal_vertex: Vertex):
        self.graph = graph
        self.start_vertex = start_vertex
        self.goal_vertex = goal_vertex

        self.q: PriorityQueue = PriorityQueue()  # priority queue for vertices - contains believed
        # distance from start
        self.k_m = 0  # key modifier
        self.rhs: DefaultDict[Vertex, float] = defaultdict(lambda: np.inf)  # right hand side function (minimum
        # distance be predecessors)
        self.g: DefaultDict[Vertex, float] = defaultdict(lambda: np.inf)

        self.rhs[self.goal_vertex] = 0
        self.q.insert(self.goal_vertex, (h(self.start_vertex, self.goal_vertex), 0))

    def calc_key(self, v: Vertex) -> Tuple[float, float]:
        """
        calculate key for the heap
        :param v: vertex to calculate the key for
        :return: tuple that corresponds to the heap key
        """
        min_val = min(self.g[v], self.rhs[v])
        return min_val + h(self.start_vertex, v) + self.k_m, min_val

    def update_vertex(self, v):
        """
        update key in priority queue
        :param v: vertex to update
        :return:
        """
        g_val = self.g[v]
        rhs_val = self.rhs[v]
        in_queue = v in self.q
        if g_val != rhs_val:
            if in_queue:
                self.q.update(v, self.calc_key(v))
            else:
                self.q.insert(v, self.calc_key(v))
        else:
            if in_queue:
                self.q.remove(v)

    def compute_shortest_path(self, vertex: Vertex):
        """
        computes the shortest path
        :param: vertex -> the current vertex that we want to compute from
        """
        self.start_vertex = vertex
        while self.q.top_key() < self.calc_key(self.start_vertex) or \
                self.rhs[self.start_vertex] > self.g[self.start_vertex]:
            u: Vertex = self.q.top()
            k_old = self.q.top_key()
            k_new = self.calc_key(u)

            if k_old < k_new:
                self.q.update(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.q.remove(u)
                for edge in u.in_edges:
                    s = edge.src
                    self.rhs[s] = min(self.rhs[s], edge.weight + self.g[u])
                    self.update_vertex(s)
            else:
                g_old = self.g[u]
                self.g[u] = np.inf
                for edge in u.in_edges:
                    s = edge.src
                    if self.rhs[s] == edge.weight + g_old:
                        if s != self.goal_vertex:
                            self.rhs[s] = min((e.weight + self.g[e.dst] for e in s.out_edges), default=np.inf)
                    self.update_vertex(s)
                if self.rhs[u] == g_old:
                    if u != self.goal_vertex:
                        self.rhs[u] = min((e.weight + self.g[e.dst] for e in u.out_edges), default=np.inf)
                self.update_vertex(u)
