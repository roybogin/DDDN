import os
import time
from typing import Set

import pybullet as p
import pybullet_data as pd
from gym.utils import seeding

import PRM
import d_star
import map_create
import mazes
from consts import Direction
from helper import *
from scan_to_map import Map


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


class Car:
    def __init__(self, number):
        super(Car, self).__init__()
        self.ax = plt.gca()  # pyplot to draw and debug
        self.car_number = number

        self.is_backwards_driving = False
        self.need_recalculate_path = False

        self.current_vertex = None
        self.next_vertex = None
        self.prev_vertex = []

        self.calculations_clock = None

        self.action = None
        self.trace = None  # the trace of the car's paths, for plotting
        self.need_recalculate_path = False

        self.calculations_clock = None

        self.rotation = None  # rotation of the car (in radians)
        self.base_pos = None  #  the position of the car according to bullet, not used
        self.center_pos = None  # the real center of the car
        self.start_point = None  # starting point of the map

        self.finished = None  # did the car get to the goal
        self.crashed = None  # did the car crash
        self.swivel = None  # swivel of the car - angle of steering wheel

        self.end_point = None  # end point of the map

        self.segments_partial_map: Map | None = None
        self.hits = None

        # TODO: copy from env:

        self.prm = PRM.PRM(
            (
                int((2 * consts.size_map_quarter) // consts.vertex_offset),
                int((2 * consts.size_map_quarter) // consts.vertex_offset),
            )
        )

        self.generate_graph()

        self.car_model = None  # pybullet ID of the car
        self.wheels = None  # pybullet ID of the wheels for setting speed
        self.steering = None  # pybullet ID of the wheels for steering

        # TODO: we only want end and start:
        self.maze, self.end_point, self.start_point = self.get_new_maze()

        # real initialization
        self.end_point = self.prm.set_end(np.array(self.end_point[:2]))
        self.current_vertex = self.prm.get_closest_vertex(
            self.start_point, 0, Direction.FORWARD
        )
        print(f"new end is {self.end_point}")
        self.start_point = [self.current_vertex.pos[0], self.current_vertex.pos[1], 0]
        self.center_pos = self.current_vertex.pos

        self.next_vertex = None

        self.car_model, self.wheels, self.steering = self.create_car_model()

        # TODO: should that be here?
        self.reset()

        self.scan_environment()

        self.prm.init_d_star(self.current_vertex)
        self.prm.d_star.compute_shortest_path(self.current_vertex)
        self.prm.draw_path(self.current_vertex)

    def generate_graph(self):
        print("generating graph")
        self.prm.generate_graph()

        print(self.prm.graph.n, self.prm.graph.e)

    # TODO: make me pretty please, pretty please <3:
    def scan_environment(self):
        """
        scans the environment and updates the discovery values
        :return:
        """
        directions = [
            2 * np.pi * i / consts.ray_amount for i in range(consts.ray_amount)
        ]
        new_map_discovered = self.discovered
        vertex_removal_radius = math.ceil(0.4 / consts.vertex_offset)
        edge_removal_radius = np.ceil(self.prm.res / consts.vertex_offset)
        problematic_vertices: Set[PRM.Vertex] = set()
        problematic_edges: Set[PRM.Edge] = set()
        new_segments = []
        for i, direction in enumerate(directions):

            did_hit, start, end = self.ray_cast(
                self.car_model,
                [0, 0, 0.5],
                [
                    -consts.ray_length * np.cos(direction),
                    -consts.ray_length * np.sin(direction),
                    0,
                ],
            )
            if did_hit:
                self.hits[i].append((end[0], end[1]))
                if len(self.hits[i]) == consts.max_hits_before_calculation:
                    self.segments_partial_map.add_points_to_map(self.hits[i])
                    self.hits[i] = []
                    new = self.segments_partial_map.new_segments
                    new_segments += new
                    for segment in new:
                        for point in segment:
                            for block in block_options(
                                map_index_from_pos(point),
                                vertex_removal_radius,
                                np.shape(self.discovered),
                            ):
                                for angle in self.prm.vertices[block[0]][block[1]]:
                                    for vertex in angle:
                                        if (
                                            vertex
                                            and not self.segments_partial_map.check_state(
                                                vertex
                                            )
                                        ):
                                            if self.prm.remove_vertex(vertex):
                                                self.need_recalculate_path = True
            add_discovered_matrix(new_map_discovered, start, end)
        self.discovered = new_map_discovered
        for segment in new_segments:
            for point in segment:
                for block in block_options(
                    map_index_from_pos(point),
                    edge_removal_radius,
                    np.shape(self.discovered),
                ):
                    for angle in self.prm.vertices[block[0]][block[1]]:
                        problematic_vertices.update(angle)

        for vertex in problematic_vertices:
            if vertex is None:
                continue
            for edge in vertex.in_edges | vertex.out_edges:
                if (
                    edge.src in problematic_vertices
                    and edge.dst in problematic_vertices
                ):
                    problematic_edges.add(edge)
        for segment in new_segments:
            for i in range(len(segment) - 1):
                for edge in problematic_edges:
                    if (
                        edge.active
                        and distance_between_lines(
                            segment[i], segment[i + 1], edge.src.pos, edge.dst.pos
                        )
                        < consts.width + 2 * consts.epsilon
                    ):
                        if self.prm.remove_edge(edge):
                            edge.active = False
                            self.need_recalculate_path = True

    def reset(self):
        """
        resets the environment
        """
        self.calculations_clock = 0
        self.trace = []
        self.calculations_clock = 0
        self.hits = [[] for _ in range(consts.ray_amount)]

        self.segments_partial_map = Map([consts.map_borders.copy()])

        self.swivel = 0
        self.rotation = 0

        self.finished = False
        self.crashed = False
        self.last_speed = 0

        self.set_car_position(self.start_point)

    def ray_cast(self, car, offset, direction):
        """
        generates a raycast in a given direction
        :param car: car ID
        :param offset: offset from the car to start the raycast
        :param direction: direction of ray
        :return: (did the ray collide with an obstacle, start position of the ray, end position of the ray)
        """
        pos, quaternions = p.getBasePositionAndOrientation(car)
        euler = p.getEulerFromQuaternion(quaternions)
        x = math.cos(euler[2])
        y = math.sin(euler[2])
        offset = [
            x * offset[0] - y * offset[1],
            x * offset[1] + y * offset[0],
            offset[2],
        ]
        direction = [
            x * direction[0] - y * direction[1],
            x * direction[1] + y * direction[0],
            direction[2],
        ]
        start = add_lists([pos, offset])
        end = add_lists([pos, offset, direction])
        ray_test = p.rayTest(start, end)
        if ray_test[0][3] == (0, 0, 0) and ray_test[0][0] not in self.borders:
            return False, start[:2], end[:2]
        else:
            return True, start[:2], ray_test[0][3]

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

    def step(self):
        """
        runs the simulation one step
        :return: (next observation, reward, did the simulation finish, info)
        """

        if self.next_vertex and dist(self.center_pos, self.next_vertex.pos) <= 0.05:
            print("got to", self.center_pos, self.rotation)
            self.calculations_clock = 0
        if self.calculations_clock == 100:
            self.calculations_clock = 0

        if self.calculations_clock == 0:
            self.trace.append(self.center_pos)
            self.next_vertex = self.prm.next_in_path(self.current_vertex)

        if self.calculations_clock == 0:
            self.trace.append(self.center_pos)
            next_vertex = self.initial_prm.next_in_path(self.current_vertex)
            if next_vertex is None:
                if (not self.is_backwards_driving) or dist(
                    self.center_pos, self.next_vertex.pos
                ) <= 0.05:
                    self.next_vertex = self.prev_vertex.pop()
                    self.is_backwards_driving = True
                    print("popped")
            else:
                print("forward")
                self.next_vertex = next_vertex
                self.is_backwards_driving = False

        if self.calculations_clock % consts.calculate_action_time == 0:
            transformed = self.initial_prm.transform_by_values(
                self.center_pos, self.rotation, self.next_vertex
            )
            x_tag, y_tag = transformed[0][0], transformed[0][1]

            radius = np.sqrt(self.initial_prm.radius_x_y_squared(x_tag, y_tag))
            delta = np.sign(y_tag) * np.arctan(consts.length / radius)

            rotation = [delta, delta]

            rotation = np.array(rotation)

            self.action = [np.sign(x_tag) / (1 + 4 * abs(delta)), rotation]

        self.calculations_clock += 1

        wanted_speed = self.action[0] * consts.max_velocity
        wanted_steering_angle = self.action[1]
        wanted_steering_angle = np.sign(wanted_steering_angle) * np.minimum(
            np.abs(wanted_steering_angle), consts.max_steer
        )
        if abs(wanted_speed) > consts.max_velocity:
            wanted_speed = consts.max_velocity * np.sign(wanted_speed)

        # moving
        for wheel in self.wheels:
            p.setJointMotorControl2(
                self.car_model,
                wheel,
                p.VELOCITY_CONTROL,
                targetVelocity=wanted_speed,
                force=consts.max_force,
            )

        for steer, angle in zip(self.steering, wanted_steering_angle):
            p.setJointMotorControl2(
                self.car_model, steer, p.POSITION_CONTROL, targetPosition=angle,
            )

    # TODO : doc
    # TODO: handle finishing the maze in all various ways
    def scan(self):
        # updating map;
        self.base_pos, quaternions = p.getBasePositionAndOrientation(self.car_model)
        self.rotation = p.getEulerFromQuaternion(quaternions)[2]
        self.center_pos = PRM.pos_to_car_center(
            np.array(self.base_pos[:2]), self.rotation
        )

        # checking if collided or finished
        if self.check_collision(self.car_model, self.bodies):
            self.crashed = True
        if dist(self.center_pos, self.end_point) < consts.min_dist_to_target:
            self.finished = True
        # # getting values for NN
        if self.base_pos[2] > 0.1:
            self.crashed = True

        self.base_pos = self.base_pos[:2]

        # saving for later
        swivel_states = p.getJointStates(self.car_model, self.steering)
        angles = [state[0] for state in swivel_states]
        cot_delta = (1 / np.tan(angles[0]) + 1 / np.tan(angles[1])) / 2
        self.swivel = np.arctan(1 / cot_delta)

        prev_vertex = self.current_vertex
        self.current_vertex = self.initial_prm.get_closest_vertex(
            self.center_pos, self.rotation
        )
        if self.current_vertex != prev_vertex and not self.is_backwards_driving:
            self.prev_vertex.append(prev_vertex)

        self.scan_environment()

        if (
            self.need_recalculate_path
            and self.calculations_clock % consts.calculate_d_star_time == 0
        ):
            self.prm.d_star.k_m += d_star.h(prev_vertex, self.current_vertex)
            for edge in self.prm.deleted_edges:
                u = edge.src
                rhs = self.prm.d_star.rhs
                rhs[u] = min(rhs[u], self.prm.d_star.g[u])
                self.prm.d_star.update_vertex(u)
            # TODO: fix the code, bogin knows what to do:
            print(f"car number {self.number} recalc path, pos:", self.center_pos)
            self.prm.d_star.compute_shortest_path(self.current_vertex)

        # TODO: check for each car, and report to the env
        if not (self.crashed or self.finished):
            return False
        self.trace.append(self.center_pos)
        if self.finished:
            self.trace.append(self.end_point)

        plt.plot(
            [a for a, _ in self.trace], [a for _, a in self.trace], label="actual path"
        )
        plt.title(f"maze {self.maze_idx} - time {self.run_time}")
        plt.legend()
        plt.show()
        self.segments_partial_map.show()

        if self.crashed:
            print(
                f"crashed maze {self.maze_idx}"
                f" - distance is {dist(self.center_pos, self.end_point)}"
                f" - time {self.run_time}"
            )
            return True
        if self.finished:
            print(f"finished maze {self.maze_idx}" f" - time {self.run_time}")
            return True
        return False

    def set_car_position(self, position):
        """
        sets the car in a position
        :param position: position to place the car at
        :return:
        """
        base_position = list(PRM.car_center_to_pos(np.array(position), 0)) + [0]
        p.resetBasePositionAndOrientation(
            self.car_model, base_position, [0.0, 0.0, 0.0, 1.0]
        )
        p.resetBaseVelocity(self.car_model, [0, 0, 0], [0, 0, 0])

    def create_car_model(self):
        """
        create the car in pybullet
        :return: the car ID and important joint IDs for steering and setting speed
        """
        car = p.loadURDF(os.path.join(pd.getDataPath(), "racecar/racecar.urdf"))
        inactive_wheels = [3, 5, 7]
        wheels = [2]

        for wheel in inactive_wheels:
            p.setJointMotorControl2(
                car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0
            )

        steering = [4, 6]
        # TODO: maybe immidatiatly set the right pos and rot
        p.resetBasePositionAndOrientation(car, [0, 0, 0], [0, 0, 0, 1])

        return car, wheels, steering
