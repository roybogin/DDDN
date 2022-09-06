import os
import time
from typing import Set, Dict, List

import pybullet as p
import pybullet_data as pd
# from gym.utils import seeding

import PRM
import consts
import d_star
import map_create
import mazes
from WeightedGraph import WeightedGraph, Edge
from helper import *
from scan_to_map import Map


class Car:
    def __init__(self, index: int, positions: Dict, prm: PRM, segments_map: Map):
        super(Car, self).__init__()
        if consts.drawing:
            self.ax = plt.gca()  # pyplot to draw and debug
        self.car_number = index
        self.borders = None
        self.bodies = None

        self.parked = False
        self.changed_edges: Set[Edge] = set()

        self.is_backwards_driving = False
        self.backward_driving_counter = 0   # counter for how many steps we were driving backwards

        self.action = None
        self.trace = []  # the trace of the car's paths, for plotting

        self.calculations_clock = 0

        self.rotation = positions["rotation"]  # rotation of the car (in radians)
        self.base_pos = None  # the position of the car according to bullet, not used
        self.start_point = positions["start"]  # starting point of the map

        self.finished = False  # did the car get to the goal
        self.crashed = False  # did the car crash
        self.swivel = 0  # swivel of the car - angle of steering wheel

        self.end_point = positions["end"]  # end point of the map

        self.segments_partial_map: Map = segments_map

        self.hits = [[] for _ in range(consts.ray_amount)]
        map_length = int((2 * consts.size_map_quarter) // consts.vertex_offset)
        self.map_shape = (map_length, map_length)

        self.prm = PRM.PRM(self.map_shape, prm)

        # initialize in prm

        self.current_vertex = None
        self.next_vertex = None
        self.prev_vertex = []
        self.next_vertex = None

        self.end_point = self.prm.set_end(np.array(self.end_point[:2]))
        self.current_vertex = self.prm.get_closest_vertex(
            self.start_point, self.rotation
        )
        print(f"new end is {self.end_point}")
        self.start_point = [self.current_vertex.pos[0], self.current_vertex.pos[1], 0]
        self.center_pos = self.current_vertex.pos

        self.car_model = None  # pybullet ID of the car
        self.wheels = None  # pybullet ID of the wheels for setting speed
        self.steering = None  # pybullet ID of the wheels for steering
        self.cars = None # pybullet ID's for all the cars in the environment

    def pybullet_init(self):
        """
        initiations done after the pybullet server has started
        """
        self.car_model, self.wheels, self.steering = self.create_car_model()

        self.scan_environment()

        self.prm.init_d_star(self.current_vertex)
        self.prm.d_star.compute_shortest_path(self.current_vertex)
        if consts.drawing:
            self.prm.draw_path(self.current_vertex, idx=f"car {self.car_number}")
        print(self.prm.end.pos)

    def set_cars(self, cars):
        self.cars = cars

    def generate_graph(self):
        print("generating graph")
        self.prm.generate_graph()

        print(self.prm.graph.n, self.prm.graph.e)

    def remove_vertices(self, new):
        """
        this function removes all vertices not viable by the addition of a new list of segments
        :param new: the new segments to take into account
        """
        vertex_removal_radius = math.ceil(0.3 / consts.vertex_offset)
        for segment in new:
            if len(segment) == 1:
                for block in block_options(
                    map_index_from_pos(segment[0]), vertex_removal_radius, self.map_shape
                ):
                    for vertex in self.prm.vertices[block[0]][block[1]]:
                        if vertex and dist(vertex.pos, segment[0]) < consts.width:
                            self.prm.remove_vertex(vertex)
            else:
                for i in range(len(segment) - 1):
                    rect = get_wall(segment[i], segment[i+1], 0.3)
                    x_min = max(consts.amount_vertices_from_edge, min([map_index_from_pos(point)[0] for point in rect]))
                    x_max = min(len(self.prm.vertices) - consts.amount_vertices_from_edge, max([map_index_from_pos(point)[0] for point in rect]))
                    y_min = max(consts.amount_vertices_from_edge, min([map_index_from_pos(point)[1] for point in rect]))
                    y_max = min(len(self.prm.vertices) - consts.amount_vertices_from_edge, max([map_index_from_pos(point)[1] for point in rect]))
                    for x in range(x_min, x_max+1):
                        for y in range(y_min, y_max+1):
                            if is_in_rect(rect, (x, y)):
                                for angle in range(consts.directions_per_vertex):
                                    v = self.prm.vertices[x][y][angle]
                                    self.prm.remove_vertex(v)

    def remove_edges(self, new_segments, deactivate=False):
        """
        given walls in segments we calculate which walls we need to remove from the graph
        deactivate indicates if we want to deactivate rather than remove
        """
        edge_removal_radius = np.ceil(self.prm.res / consts.vertex_offset)
        problematic_vertices: Set[PRM.Vertex] = set()
        problematic_edges: Set[PRM.Edge] = set()
        for segment in new_segments:
            for point in segment:
                for block in block_options(
                    map_index_from_pos(point), edge_removal_radius, self.map_shape
                ):
                    problematic_vertices.update(self.prm.vertices[block[0]][block[1]])

        for vertex in problematic_vertices:
            if vertex is None:
                continue
            for edge in vertex.in_edges | vertex.out_edges:
                if (
                    edge.src in problematic_vertices
                    and edge.dst in problematic_vertices
                ):
                    problematic_edges.add(edge)
        for edge in problematic_edges:
            changed_edge = False
            for segment in new_segments:
                if changed_edge:
                    break
                if len(segment) == 1:
                    if not deactivate:
                        if edge.active and perpendicularDistance(segment[0], edge.src.pos, edge.dst.pos) < \
                                consts.width + 2 * consts.epsilon:
                            self.prm.remove_edge(edge)
                            changed_edge = True
                    else:
                        if edge.active and perpendicularDistance(segment[0], edge.src.pos, edge.dst.pos) < \
                                consts.width + 1 * consts.epsilon:
                            edge.weight = np.inf
                            self.changed_edges.add(edge)
                            edge.parked_cars += 1
                            changed_edge = True
                else:
                    for i in range(len(segment) - 1):
                        if not deactivate:
                            if edge.active and distance_between_lines(segment[i], segment[i + 1], edge.src.pos, edge.dst.pos) < \
                                    consts.width + 2 * consts.epsilon:
                                self.prm.remove_edge(edge)
                                changed_edge = True
                                break   # exit for over i
                        else:
                            if edge.active and distance_between_lines(segment[i], segment[i + 1], edge.src.pos, edge.dst.pos) < \
                                    consts.width + 1 * consts.epsilon:
                                edge.weight = np.inf
                                self.changed_edges.add(edge)
                                edge.parked_cars += 1
                                changed_edge = True
                                break  # exit for over i

    def scan_environment(self):
        """
        scans the environment and updates the discovery values
        """
        old_graph_sizes = (self.prm.graph.n, self.prm.graph.e)
        directions = [
            2 * np.pi * i / consts.ray_amount for i in range(consts.ray_amount)
        ]
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
                    self.remove_vertices(new)
        self.remove_edges(new_segments)
        return old_graph_sizes != (self.prm.graph.n, self.prm.graph.e)  # we removed new edges or vertices

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

    def deactivate_edges(self):
        points_to_check = [(self.center_pos[0] - 1/2 * consts.width, self.center_pos[1] - 1/2 * consts.length),
                           (self.center_pos[0] - 1/2 * consts.width, self.center_pos[1] + 1/2 * consts.length),
                           (self.center_pos[0] + 1/2 * consts.width, self.center_pos[1] + 1/2 * consts.length),
                           (self.center_pos[0] + 1/2 * consts.width, self.center_pos[1] - 1/2 * consts.length),
                           (self.center_pos[0] - 1/2 * consts.width, self.center_pos[1] - 1/2 * consts.length),
                           ]
        horizontal_line = [(self.center_pos[0] - 1/2 * consts.width, self.center_pos[1]),
                           (self.center_pos[0] + 1/2 * consts.width, self.center_pos[1])]
        vertical_line = [(self.center_pos[0], self.center_pos[1] - 1/2 * consts.length),
                           (self.center_pos[0], self.center_pos[1] + 1/2 * consts.length)]
        points_to_check = [self.prm.rotate_angle(np.array(point), self.rotation) for point in points_to_check]
        self.remove_edges([points_to_check, horizontal_line, vertical_line], True)

    def step(self):
        """
        this function is called each frame,
        with the known graph vertex the car is on, and next vertex for the car to get to,
        the function gets the next action to make, and performes it on the pybullet server.
        """
        needs_parking = False
        for number in range(self.car_number):
            other = self.cars[number]
            if other:
                distance_from_car = dist(self.center_pos, other.center_pos)
                if (self.parked and distance_from_car < 2 * consts.minimum_car_dist) or distance_from_car < consts.minimum_car_dist:
                    needs_parking = True

        changed_parking = False

        if needs_parking and not self.parked:
            self.parked = True
            self.deactivate_edges()
            changed_parking = True
            print('parking')
        if not needs_parking and self.parked:
            changed_parking = True
            self.parked = False
            self.calculations_clock = 0  # need to do calculations now
            for edge in self.changed_edges:
                edge.parked_cars -= 1
                if edge.parked_cars == 0:
                    edge.weight = edge.original_weight

        if self.parked or changed_parking:
            self.action = [0, [0, 0]]
            # moving
            for wheel in self.wheels:
                p.setJointMotorControl2(self.car_model, wheel, p.VELOCITY_CONTROL, targetVelocity=0,
                                        force=consts.max_force)

            for steer in self.steering:
                p.setJointMotorControl2(self.car_model, steer, p.POSITION_CONTROL, targetPosition=0)

            # NOTE: calculation clock is irrelevant
            return changed_parking

        if self.next_vertex and dist(self.center_pos, self.next_vertex.pos) <= 0.05:
            print("got to", self.center_pos, self.rotation)
            self.calculations_clock = 0
        if self.calculations_clock == consts.reset_count_time:
            self.calculations_clock = 0

        if self.calculations_clock == 0:
            self.trace.append(self.center_pos)
            next_vertex = self.prm.next_in_path(self.current_vertex)
            if self.is_backwards_driving:
                self.backward_driving_counter -= 1
            if next_vertex is None or self.backward_driving_counter > 0:
                if (not self.is_backwards_driving) or dist(
                    self.center_pos, self.next_vertex.pos
                ) <= 0.1:
                    self.next_vertex = self.prev_vertex.pop()
                    self.is_backwards_driving = True
                    print("popped")
                    self.backward_driving_counter = 2
            else:
                print("forward")
                self.next_vertex = next_vertex
                self.is_backwards_driving = False
                self.backward_driving_counter = 0

        if self.calculations_clock % consts.calculate_action_time == 0:
            transformed_vertex = self.prm.transform_by_values(
                self.center_pos, self.rotation, self.next_vertex
            )
            x_tag, y_tag = transformed_vertex[0][0], transformed_vertex[0][1]

            radius = np.sqrt(self.prm.radius_x_y_squared(x_tag, y_tag))
            delta = np.sign(y_tag) * np.arctan(consts.length / radius)

            rotation = [delta, delta]

            rotation = np.array(rotation)

            self.action = [np.sign(x_tag) / (1 + 4 * abs(delta)), rotation]

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
                    self.car_model, steer, p.POSITION_CONTROL, targetPosition=angle
                )
        self.calculations_clock += 1
        return False

    def update_state(self):
        """
        updates the state of the car afrer a step,
        updates variables and checks if the car collided with something, or got to it's goal.
        returns true if this car is done
        """
        # updating map;
        self.base_pos, quaternions = p.getBasePositionAndOrientation(self.car_model)
        self.rotation = p.getEulerFromQuaternion(quaternions)[2]
        self.center_pos = PRM.pos_to_car_center(
            np.array(self.base_pos[:2]), self.rotation
        )
        # checking if collided or finished
        if self.check_collision(self.car_model, self.bodies + [other.car_model for other in self.cars if other]):
            self.crashed = True
        if dist(self.center_pos, self.end_point) < consts.min_dist_to_target:
            self.finished = True
        if self.base_pos[2] > 0.1:
            self.crashed = True

        self.base_pos = self.base_pos[:2]

        # saving for later
        # swivel_states = p.getJointStates(self.car_model, self.steering)
        # angles = [state[0] for state in swivel_states]
        # cot_delta = (1 / np.tan(angles[0]) + 1 / np.tan(angles[1])) / 2
        # self.swivel = np.arctan(1 / cot_delta)
        prev_vertex = self.current_vertex
        self.current_vertex = self.prm.get_closest_vertex(self.center_pos, self.rotation)
        if self.current_vertex != prev_vertex and not self.is_backwards_driving:
            self.prev_vertex.append(prev_vertex)
        self.scan_environment()
        return self.crashed or self.finished

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
        base_position = list(
            PRM.car_center_to_pos(np.array(self.start_point[:2]), self.rotation)
        ) + [0]
        # TODO: maybe immidatiatly set the right pos and rot
        p.resetBasePositionAndOrientation(
            car, base_position, p.getQuaternionFromEuler([0, 0, self.rotation])
        )

        return car, wheels, steering
