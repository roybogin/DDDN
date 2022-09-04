import os
import time
from typing import Set, List, Dict

import pybullet as p
import pybullet_data as pd
from gym.utils import seeding

import PRM
import consts
import d_star
import map_create
import mazes
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
        self.ax = plt.gca()  # pyplot to draw and debug
        plt_size = consts.size_map_quarter + 1  # pyplot size
        plt.axis([-plt_size, plt_size, -plt_size, plt_size])
        self.maze_title = maze['title']

        self.run_time = None  # time of the run

        self.maze = None  # walls of the maze - list of points
        self.borders = None  # the maze borders - object IDs

        # TODO: maybe generate and copy:
        self.initial_prm = None

        # self.initial_prm = PRM.PRM(
        #     (
        #         int((2 * consts.size_map_quarter) // consts.vertex_offset),
        #         int((2 * consts.size_map_quarter) // consts.vertex_offset),
        #     )
        # )

        # self.generate_graph()

        self.obstacles = []  # list of obstacle IDs in pybullet
        self.bodies = []  # list of all collision body IDs in pybullet

        self.maze = maze['walls']

        self.start_env()
        positions = maze['positions']
        self.number_of_cars = len(positions)
        # TODO: split to generating cars and to placing cars for pybullet speed
        self.cars: List[Car] = [Car(i, positions[i]) for i in range(self.number_of_cars)]
        self.reset()
        for car in self.cars:
            car.after_py_bullet()

    def generate_graph(self):
        print("generating graph")
        self.initial_prm.generate_graph()

        print(self.initial_prm.graph.n, self.initial_prm.graph.e)

    def start_env(self):
        """
        start the pybullet environment and create the car
        :return:
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
        adds the boarders to the maze
        """
        self.borders = map_create.create_poly_wall(
            consts.map_borders, epsilon=consts.epsilon, client=p
        )

    def remove_all_bodies(self):
        """
        removes all collision bodies from the map
        :return:
        """
        for body in self.bodies:
            p.removeBody(body)
        self.bodies = []

    def reset(self):
        """
        resets the environment
        """
        self.remove_all_bodies()
        self.add_borders()

        self.run_time = 0

        self.obstacles = map_create.create_map(self.maze, epsilon=consts.epsilon, client=p)
        self.bodies = self.borders + self.obstacles

        for car in self.cars:
            car.bodies = self.bodies
            car.borders = self.borders

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

    # TODO: handle finishing the maze in all various ways
    def step(self):
        """
        runs the simulation one step
        :return: (next observation, reward, did the simulation finish, info)
        """

        if consts.print_runtime and self.run_time % 500 == 0:
            print("time:", self.run_time)

        # updating target velocity and steering angle
        for car in self.cars:
            car.step()

        p.stepSimulation()

        self.run_time += 1

        for car in self.cars:
            car.scan()

        # TODO: call for cars and analyze results. till "self.trace.append(self.end_point)"
        if self.run_time >= consts.max_time:
            print(
                f"out of time in {self.maze_title}"
            )
            for idx, car in enumerate(self.cars):
                car.trace.append(car.center_pos)
                plt.plot(
                    [a for a, _ in car.trace],
                    [a for _, a in car.trace],
                    label=f"actual path car {idx}",
                )
                plt.scatter(car.center_pos[0], car.center_pos[1], c="red")
            plt.title(f"{self.maze_title} - time {self.run_time}")

            # self.segments_partial_map.plot(self.ax)

            return True

        crashed = any(car.crashed for car in self.cars)
        finished = all(car.finished for car in self.cars)

        if not (crashed or finished):
            return False
        for idx, car in enumerate(self.cars):
            car.trace.append(car.center_pos)
            if finished:
                car.trace.append(car.end_point)

        # TODO: make a finishing function that prints stats at the end of maze:
        plt.title(f"{self.maze_title} - time {self.run_time}")
        # self.segments_partial_map.plot(self.ax)

        if crashed:
            print(
                f"crashed {self.maze_title}"
                # f" - distance is {dist(self.center_pos, self.end_point)}"
                f" - time {self.run_time}"
            )
            return True
        if finished:
            print(f"finished {self.maze_title}" f" - time {self.run_time}")
            return True
        return False


# for testing:
def main():
    t0 = time.time()
    stop = False
    env = Env(mazes.default_data_set[0])
    while not stop:
        stop = env.step()
    print(f"total time: {time.time() - t0}")
    p.disconnect()
    for idx, car in enumerate(env.cars):
        plt.plot(
            [a for a, _ in car.trace],
            [a for _, a in car.trace],
            label=f"actual path car {idx}",
        )
        plt.scatter(car.trace[-1][0], car.trace[-1][1], c="red")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
