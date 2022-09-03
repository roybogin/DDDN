import heapq
import itertools
import time
from collections import defaultdict
from typing import Tuple, DefaultDict

import numpy as np

from WeightedGraph import Vertex
from helper import dist
from PRM import PRM
from matplotlib import pyplot as plt


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.entry_dict = {}
        self.count = itertools.count()

    def insert(self, obj, priority):
        entry = [priority, next(self.count), obj]
        self.entry_dict[obj] = entry
        heapq.heappush(self.queue, entry)

    def remove(self, obj):
        self.entry_dict.pop(obj)[-1] = None

    def update(self, obj, new_priority):
        self.remove(obj)
        self.insert(obj, new_priority)

    def pop(self):
        while len(self.queue) != 0:
            priority, _, obj = heapq.heappop(self.queue)
            if obj is not None:
                del self.entry_dict[obj]
                return obj
        print('empty queue')
        return None

    def top(self):
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
        print('empty queue')
        return None

    def top_key(self):
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
    # D* lite algorithm that we implement
    def __init__(self, start_vertex: Vertex, goal_vertex: Vertex, prm):
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

        self.prm = prm

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

    def next_in_path(self, vertex: Vertex):

        successors = (e for e in self.prm.outgoing_edges(vertex))
        next_vertex_key = lambda dest, weight: self.g[dest] + weight
        try:
            next_vertex = min(successors, key=lambda tup: next_vertex_key(*tup))[0]
        except ValueError:
            print('a')
            next_vertex = None
        return next_vertex

    def draw_path(self, current_vertex: Vertex, idx=''):
        x_list = [current_vertex.pos[0]]
        y_list = [current_vertex.pos[1]]
        plt.scatter(x_list, y_list, c='black', label='start')
        vertex = current_vertex
        parent = self.next_in_path(vertex)
        while (parent != vertex) and (parent is not None):
            vertex = parent
            parent = self.next_in_path(vertex)
            x_list.append(vertex.pos[0])
            y_list.append(vertex.pos[1])
        plt.scatter(x_list[-1], y_list[-1], c='green', label='end goal')
        plt.plot(x_list, y_list, label=f'projected path {idx}')

    def compute_shortest_path(self, vertex: Vertex):
        """
        computes the shortest path
        :return:
        """
        print('computing path')
        t = time.time()
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
                for s, weight in self.prm.incoming_edges(u):
                    self.rhs[s] = min(self.rhs[s], weight + self.g[u])
                    self.update_vertex(s)
            else:
                g_old = self.g[u]
                self.g[u] = np.inf
                for s, weight in self.prm.incoming_edges(u):
                    if self.rhs[s] == weight + g_old:
                        if s != self.goal_vertex:
                            self.rhs[s] = min((weight + self.g[dst] for dst, weight in self.prm.outgoing_edges(s)))
                    self.update_vertex(s)
                if self.rhs[u] == g_old:
                    if u != self.goal_vertex:
                        self.rhs[u] = min((weight + self.g[dst] for dst, weight in self.prm.outgoing_edges(s)))
                self.update_vertex(u)
        print('path computed in ', time.time() - t)
