import os
import time
from typing import Set, List, Dict, Optional

import pybullet as p
import pybullet_data as pd
from matplotlib import pyplot as plt

import PRM
import consts
import maze_create
from WeightedGraph import Edge, WeightedGraph
from car import Car
from scan_to_map import Map


class Env:
    def __init__(self, maze: Dict):

        super(Env, self).__init__()

        # define Matplotlib figure and axis

        self.size_map_quarter = maze["size"] / 2

        self.map_borders = [
            (self.size_map_quarter, self.size_map_quarter),
            (self.size_map_quarter, -self.size_map_quarter),
            (-self.size_map_quarter, -self.size_map_quarter),
            (-self.size_map_quarter, self.size_map_quarter),
            (self.size_map_quarter, self.size_map_quarter),
        ]

        if consts.drawing:
            self.ax = plt.gca()  # pyplot to draw and debug
            plt_size = self.size_map_quarter + 1  # pyplot size
            plt.axis([-plt_size, plt_size, -plt_size, plt_size])

        self.maze_title = maze["title"]

        self.segments_partial_map: Map = Map([self.map_borders.copy()], maze['size'])

        self.run_time: int = 0  # time of the run

        self.maze: List = maze["walls"]  # walls of the maze - list of points
        self.borders: Optional[List[int]] = None  # the maze borders - object IDs

        map_length = int((2 * self.size_map_quarter) // consts.vertex_offset)

        # Initialize structures
        self.prm: PRM.PRM = PRM.PRM(self.size_map_quarter, (map_length, map_length))  # A PRM object to generate
        # structures
        self.generate_graph()
        self.graph: WeightedGraph = self.prm.graph  # the graph used by all the cars
        self.obstacles: List[int] = []  # list of obstacle IDs in pybullet
        self.bodies: List[int] = []  # list of all collision body IDs in pybullet

        # Generate cars
        positions = maze["positions"]
        self.number_of_cars = len(positions)
        self.cars: List[Optional[Car]] = [
            Car(
                i,
                positions[i],
                self.prm,
                self.segments_partial_map,
                self.size_map_quarter,
            )
            for i in range(self.number_of_cars)
        ]
        if consts.drawing:
            self.traces: List[List] = [car.trace for car in self.cars]

        # initialize pybullet
        self.start_env()
        self.add_borders()
        self.obstacles = maze_create.create_map(
            self.maze, epsilon=consts.epsilon, client=p
        )
        self.bodies = self.borders + self.obstacles

        for car in self.cars:
            car.bodies = self.bodies
            car.borders = self.borders
            car.after_pybullet_init()
            car.set_cars(self.cars)

    def generate_graph(self):
        """
        calls the prm generate graph function
        """
        self.prm.generate_graph()

        if consts.debugging:
            print("generated graph")
            print(self.prm.graph.n, self.prm.graph.e)

    def start_env(self):
        """
        start the pybullet environment and create the car
        """
        if consts.is_visual:
            a = p.connect(p.GUI)
            p.setAdditionalSearchPath(pd.getDataPath(), a)

        b = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath(), b)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        p.setGravity(0, 0, -10)

        p.setRealTimeSimulation(consts.use_real_time)
        # p.setTimeStep(consts.time_step)
        p.loadSDF(os.path.join(pd.getDataPath(), "stadium.sdf"))
        p.resetDebugVisualizerCamera(
            cameraDistance=consts.cameraDistance,
            cameraYaw=consts.cameraYaw,
            cameraPitch=consts.cameraPitch,
            cameraTargetPosition=consts.cameraTargetPosition,
        )

    def add_borders(self):
        """
        adds the boarder walls to the maze
        """
        self.borders = maze_create.create_poly_wall(self.map_borders, epsilon=consts.epsilon, client=p)

    def step(self):
        """
        runs the simulation one step
        calls the step and scan on all cars
        """
        if consts.print_runtime and self.run_time % 400 == 0:
            print("time:", self.run_time)

        # updating target velocity and steering angle
        changed_edges: Set[Edge] = set()
        for car in self.cars:
            if car and car.step():
                changed_edges.update(car.changed_edges)
                if not car.is_parked:  # car stopped parking - doesn't need its changed edges
                    car.changed_edges.clear()
                    
        if len(changed_edges) != 0:
            # update graph so cars won't collide
            print('computing paths - parking')
            t = time.time()
            for car in self.cars:
                if car:
                    if car.is_parked or car.finished:
                        continue
                    car.prm.update_d_star(changed_edges, car.current_vertex)
                    car.prm.d_star.compute_shortest_path(car.current_vertex)
                    car.calculations_clock = 0
            print("all paths computed in ", time.time() - t)

        p.stepSimulation()  # make a step for all cars
        self.run_time += 1

        should_scan = self.run_time % consts.scan_time == 0

        for car in self.cars:
            if car:
                car.update_state(should_scan)  # all the cars scan their environment
                if car.finished:
                    p.removeBody(car.car_model)
                    idx = car.car_number
                    del car
                    self.cars[idx] = None

        if len(self.graph.deleted_edges) != 0 and self.run_time % consts.calculate_d_star_time == 0:
            # cars saw a wall - compute paths
            print("computing paths - wall")
            t = time.time()
            for car in self.cars:
                if car:
                    if car.is_parked or car.finished:
                        continue
                    car.prm.update_d_star(self.graph.deleted_edges, car.current_vertex)
                    car.prm.d_star.compute_shortest_path(car.current_vertex)
                    car.calculations_clock = 0
            print("all paths computed in ", time.time() - t)
            self.graph.deleted_edges.clear()

        if self.run_time >= consts.max_time:    # the car ran out of time
            print(f"out of time in {self.maze_title}")

            return True

        # did any car crash or all finished
        crashed = any(car.crashed for car in self.cars if car)
        finished = all(car.finished for car in self.cars if car)        
        if not (crashed or finished):
            return False

        if crashed:
            print(f"crashed {self.maze_title} - time {self.run_time}")
            return True
        if finished:
            print(f"finished {self.maze_title}" f" - time {self.run_time}")
            return True
        return False
