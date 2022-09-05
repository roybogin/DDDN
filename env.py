import os
import time
from typing import Set, List, Dict

import pybullet as p
import pybullet_data as pd

import PRM
import consts
import d_star
import map_create
import mazes
from WeightedGraph import Edge
from helper import *
from scan_to_map import Map
from car import Car


def add_discovered_matrix(discovered_matrix, start, end):
    """
    updates the discovered matrix by drawing a line on it matching the endpoints of the raycast
    :param discovered_matrix: matrix that represents the discovered areas by the car
    :param start: start of the raycast
    :param end: end of the raycast
    :return: the indices of discovered area
    """
    x0, y0 = map_index_from_pos(start)
    x1, y1 = map_index_from_pos(end)
    return plot_line(x0, y0, x1, y1, discovered_matrix)


class Env:
    def __init__(self, maze: Dict):

        super(Env, self).__init__()

        # define Matplotlib figure and axis
        if consts.drawing:
            self.ax = plt.gca()  # pyplot to draw and debug
            plt_size = consts.size_map_quarter + 1  # pyplot size
            plt.axis([-plt_size, plt_size, -plt_size, plt_size])
            self.maze_title = maze["title"]

        self.segments_partial_map: Map = Map([consts.map_borders.copy()])

        self.run_time = None  # time of the run

        self.maze = None  # walls of the maze - list of points
        self.borders = None  # the maze borders - object IDs

        map_length = int((2 * consts.size_map_quarter) // consts.vertex_offset)

        self.prm = PRM.PRM((map_length, map_length))

        self.generate_graph()

        self.graph = self.prm.graph

        self.obstacles = []  # list of obstacle IDs in pybullet
        self.bodies = []  # list of all collision body IDs in pybullet

        self.maze = maze["walls"]

        self.start_env()
        positions = maze["positions"]
        self.number_of_cars = len(positions)
        self.cars: List[Car] = [
            Car(i, positions[i], self.prm, self.segments_partial_map)
            for i in range(self.number_of_cars)
        ]

        self.add_borders()

        self.run_time = 0

        self.obstacles = map_create.create_map(
            self.maze, epsilon=consts.epsilon, client=p
        )
        self.bodies = self.borders + self.obstacles

        for car in self.cars:
            car.bodies = self.bodies
            car.borders = self.borders

        for car in self.cars:
            car.pybullet_init()

        for car in self.cars:
            car.set_cars([other for other in self.cars if other != car])

    def generate_graph(self):
        """
        calls the prm genearte_graph function
        """
        self.prm.generate_graph()

        if consts.debug:
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
        p.setTimeStep(consts.time_step)
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
        self.borders = map_create.create_poly_wall(
            consts.map_borders, epsilon=consts.epsilon, client=p
        )

    # TODO: call check collision on each car
    def check_collision(self, car_model, obstacles, margin=0, max_distance=1.0):
        """
        did the car collide with an obstacle
        :param car_model: car ID
        :param obstacles: list of body IDs to check collision of the car with
        :param margin: margin of error for collision - if the distance is smaller than the margin - the car collided
        :param max_distance: distance from the car to search for collisions in
        :return: did the car collide with an obstacle
        """
        for ob in obstacles:
            closest_points = p.getClosestPoints(car_model, ob, distance=max_distance)
            closest_points = [
                a for a in closest_points if not (a[1] == a[2] == car_model)
            ]
            if len(closest_points) != 0:
                distance = np.min([pt[8] for pt in closest_points])
                if distance < margin:
                    return True
        return False

    # TODO: handle finishing the maze in all various ways, the change should go up to car level.

    def step(self):
        """
        runs the simulation one step
        calls the step and scan on all cars
        """

        if consts.print_runtime and self.run_time % 1 == 0:
            print("time:", self.run_time)

        # updating target velocity and steering angle
        print('a')
        changed_edges: Set[Edge] = set()
        for car in self.cars:
            if car.step():
                print('time now is ', self.run_time)
                changed_edges.update(car.changed_edges)
                if not car.parked:
                    car.changed_edges.clear()


                # changed_edges.add(0)
                #
                # map = []
                # vert = self.prm.vertices
                # for row in range(0, len(vert)):
                #     map.append([])
                #     for col in range(0, len(vert[row])):
                #         if len(vert[row][col]) == 0:
                #             map[row].append(0)
                #             continue
                #         v = vert[row][col][0]
                #
                #         o = sum((1 for e in v.out_edges if e.weight != np.inf))
                #         i = sum((1 for e in v.in_edges if e.weight != np.inf))
                #         map[row].append(o+i)
        print('b')
        if len(changed_edges) != 0:
            print('computing paths - park')
            t = time.time()
            for car in self.cars:
                if car.parked:
                    continue
                car.prm.update_d_star(changed_edges, car.current_vertex)
                car.prm.d_star.compute_shortest_path(car.current_vertex)
                car.calculations_clock = 0
            print('all paths computed in ', time.time() - t)
        print('c')

        p.stepSimulation()

        self.run_time += 1
        print('d')

        for car in self.cars:
            car.update_state()
        print('e')

        if len(self.graph.deleted_edges) != 0 and self.run_time % consts.calculate_d_star_time == 0:
            print("computing paths - wall")
            t = time.time()
            for car in self.cars:
                car.prm.update_d_star(self.graph.deleted_edges, car.current_vertex)
                car.prm.d_star.compute_shortest_path(car.current_vertex)
                car.calculations_clock = 0
            print("all paths computed in ", time.time() - t)
            self.graph.deleted_edges.clear()
        print('f')

        if self.run_time >= consts.max_time:
            print(f"out of time in {self.maze_title}")
            for idx, car in enumerate(self.cars):
                car.trace.append(car.center_pos)
                if car.finished:
                    car.trace.append(car.end_point)
            return True

        crashed = any(car.crashed for car in self.cars)
        finished = all(car.finished for car in self.cars)

        if not (crashed or finished):
            return False
        for idx, car in enumerate(self.cars):
            car.trace.append(car.center_pos)
            if car.finished:
                car.trace.append(car.end_point)
            elif car.crashed:
                plt.scatter(
                    car.trace[-1][0], car.trace[-1][1], label=f"crash car {idx}"
                )

        if crashed:
            print(f"crashed {self.maze_title} - time {self.run_time}")
            return True
        if finished:
            print(f"finished {self.maze_title}" f" - time {self.run_time}")
            return True
        return False


# for testing:
def main():
    t0 = time.time()
    stop = False
    maze = mazes.default_data_set[2]
    env = Env(maze)
    while not stop:
        stop = env.step()
    print(f"total time: {time.time() - t0}")
    p.disconnect()
    if consts.drawing:
        env.segments_partial_map.plot(env.ax)
        for idx, car in enumerate(env.cars):
            plt.plot([a for a, _ in car.trace], [a for _, a in car.trace], label=f"actual car {idx}")
        plt.title(f'{maze["title"]} - time {env.run_time}')
        ax = env.ax
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()


if __name__ == "__main__":
    main()
