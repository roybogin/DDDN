import os
from typing import Set, Dict, Sequence, Optional, List, Tuple

import pybullet as p
import pybullet_data as pd

import PRM
from WeightedGraph import Edge, Vertex
from helper import *
from scan_to_map import Map


class Car:
    def __init__(self, index: int, positions: Dict, prm: PRM, segments_map: Map):
        """
        creates a car instance for the maze
        :param index: index of the car in the environment
        :param positions: dictionary that states the initial position, end position and initial rotation of the car (
        in the specified format from the user guide)
        :param prm: shared prm object that the cars get as input to create copies
        :param segments_map: shared map for the cars that will be updated when the car encounters walls
        """
        super(Car, self).__init__()
        if consts.drawing:
            self.ax = plt.gca()  # pyplot to draw and debug
            self.trace: List = []  # the trace of the car's paths, for plotting

        self.car_number: int = index
        self.borders = None  # TODO:typing and type explain
        self.bodies = None  # TODO:typing and type explain

        self.parked: bool = False  # is the car currently parked # TODO: change name to is_parked
        self.changed_edges: Set[Edge] = set()   # set of edges that were affected by the car parking

        self.is_backwards_driving: bool = False  # is the car currently driving backwards

        self.action: Optional[Sequence[float, np.ndarray]] = None  # current action for the car and contains two items -
        # first is a value representing car speed relative the maximum speed and second is a numpy array for the
        # rotation of each wheel

        self.calculations_clock: int = 0  # internal clock for specific calculations like wall avoiding walls

        self.rotation: float = positions["rotation"]  # rotation of the car (in radians)
        self.base_pos: Optional[Sequence[float]] = None  # the position of the car according to bullet, not used
        self.start_point: Sequence[float] = positions["start"]  # starting point of the map - will be rounded to
        # nearest vertex

        self.finished: bool = False  # did the car get to the goal
        self.crashed: bool = False  # did the car crash

        self.segments_partial_map: Map = segments_map   # a shared map for all cars that contains the discovered
        # obstacles

        self.hits: List = [[] for _ in range(consts.ray_amount)]    # list of hits from the ray casters on the cars
        # for wall detection

        map_length = int((2 * consts.size_map_quarter) // consts.vertex_offset)  # amount of vertices in each row\col
        self.map_shape = (map_length, map_length)  # shape of the vertices on the map

        self.prm = PRM.PRM(self.map_shape, prm)  # prm object for running D* lite

        self.next_vertex: Optional[Vertex] = None  # next vertex in the car's path
        self.prev_vertex: List[Vertex] = []  # stack that contains the vertices in the car's path for backtracking

        self.end_point: Sequence[float] = self.prm.set_end(np.array(positions["end"][:2]))  # the end goal for the
        # car - is rounded to the nearest vertex
        self.current_vertex: Vertex = self.prm.get_closest_vertex(self.start_point, self.rotation)  # current vertex in
        # the prm that matches the car (the closest one)
        self.start_point: Sequence[float] = [self.current_vertex.pos[0], self.current_vertex.pos[1], 0]  # rounded start position
        self.center_pos: Sequence[float] | np.ndarray = self.current_vertex.pos   # position of car center

        self.car_model: Optional[int] = None  # pybullet ID of the car
        self.wheels: Optional[list] = None  # pybullet ID of the wheels for setting speed
        self.steering: Optional[list] = None  # pybullet ID of the wheels for steering
        self.cars: Optional[list] = None  # car objects for all the cars in the environment

    def after_pybullet_init(self):
        """
        initiations done after the pybullet server has started
        """
        self.car_model, self.wheels, self.steering = self.create_car_model()

        self.scan_environment()

        self.prm.init_d_star(self.current_vertex)
        self.prm.d_star.compute_shortest_path(self.current_vertex)
        if consts.drawing:
            self.prm.draw_path(self.current_vertex, idx=f"car {self.car_number}")

    def set_cars(self, cars: List):
        """
        copy the car list
        :param cars: list of cars from environment
        """
        self.cars = cars

    def add_obstacles(self, index):
        """
        TODO: doc this, idk wtf this do and add typing
        """
        vertex_removal_radius = math.ceil(0.4 / consts.vertex_offset)
        self.segments_partial_map.add_points_to_map(self.hits[index])
        self.hits[index] = []
        new = self.segments_partial_map.new_segments
        for segment in new:
            for point in segment:
                for block in block_options(
                    map_index_from_pos(point), vertex_removal_radius, self.map_shape
                ):
                    for vertex in self.prm.vertices[block[0]][block[1]]:
                        if vertex and dist(vertex.pos, point) < consts.width:
                            self.prm.remove_vertex(vertex)
        return new

    def remove_edges(self, new_segments, deactivate: bool = False):  # TODO: typing
        """
        given walls in segments we calculate which walls we need to remove from the graph
        deactivate indicates if we want to deactivate rather than remove
        """
        edge_removal_radius = np.ceil(self.prm.res / consts.vertex_offset)
        problematic_vertices: Set[Vertex] = set()
        problematic_edges: Set[Edge] = set()
        for segment in new_segments:
            for point in segment:
                for block in block_options(map_index_from_pos(point), edge_removal_radius, self.map_shape):
                    problematic_vertices.update(self.prm.vertices[block[0]][block[1]])

        for vertex in problematic_vertices:
            if vertex is None:
                continue
            for edge in vertex.in_edges | vertex.out_edges:
                if edge.src in problematic_vertices and edge.dst in problematic_vertices:
                    problematic_edges.add(edge)
        for segment in new_segments:
            for edge in problematic_edges:
                for i in range(len(segment) - 1):
                    if edge.active and distance_between_lines(segment[i], segment[i + 1], edge.src.pos, edge.dst.pos) < \
                            consts.width:
                        if not deactivate:
                            self.prm.remove_edge(edge)
                        else:
                            edge.weight = np.inf
                            self.changed_edges.add(edge)
                            edge.parked_cars += 1
                        break   # don't check for other segments

    def scan_environment(self):
        """
        scans the environment and updates the discovery values
        """
        directions = [2 * np.pi * i / consts.ray_amount for i in range(consts.ray_amount)]
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
                    new_segments += self.add_obstacles(i)
        self.remove_edges(new_segments)

    def ray_cast(self, car: int, offset: list, direction: list) -> Tuple[bool, list, list]:
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

    def check_collision(self, car_model: int, obstacles: list, margin: float = 0, max_distance: float = 1.0) -> bool:
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

    def deactivate_edges(self): #TODO: docstring
        points_to_check = [(self.center_pos[0] - 1/2 * consts.width, self.center_pos[1] - 1/2 * consts.length),
                           (self.center_pos[0] - 1/2 * consts.width, self.center_pos[1] + 1/2 * consts.length),
                           (self.center_pos[0] + 1/2 * consts.width, self.center_pos[1] + 1/2 * consts.length),
                           (self.center_pos[0] + 1/2 * consts.width, self.center_pos[1] - 1/2 * consts.length),
                           (self.center_pos[0] - 1/2 * consts.width, self.center_pos[1] - 1/2 * consts.length),
                           ]
        points_to_check = [self.prm.rotate_angle(np.array(point), self.rotation) for point in points_to_check]
        self.remove_edges([points_to_check], True)

    def step(self) -> bool:
        """
        this function is called each frame,
        with the known graph vertex the car is on, and next vertex for the car to get to,
        the function gets the next action to make, and performs it on the pybullet server.
        :return: has the parking state changed (started or finished parking)
        """
        needs_parking = False
        for number in range(self.car_number):   # check if we need to park because car is to close to us
            other = self.cars[number]
            distance_from_car = dist(self.center_pos, other.center_pos)
            if (self.parked and distance_from_car < 2 * consts.minimum_car_dist) or \
                    distance_from_car < consts.minimum_car_dist:
                needs_parking = True
                break

        changed_parking = False

        if needs_parking and not self.parked:   # deactivate close edges so cars won't come near
            self.parked = True
            self.deactivate_edges()
            changed_parking = True
            print('parking')
        if not needs_parking and self.parked:   # unpark the car and free the edges
            changed_parking = True
            self.parked = False
            self.calculations_clock = 0  # need to do calculations now
            for edge in self.changed_edges:
                edge.parked_cars -= 1
                if edge.parked_cars == 0:
                    edge.weight = edge.original_weight

        if self.parked or changed_parking:  # don't move if parking
            self.action = [0, [0, 0]]
            for wheel in self.wheels:
                p.setJointMotorControl2(self.car_model, wheel, p.VELOCITY_CONTROL, targetVelocity=0,
                                        force=consts.max_force)
            for steer in self.steering:
                p.setJointMotorControl2(self.car_model, steer, p.POSITION_CONTROL, targetPosition=0)
            # NOTE: calculation clock is irrelevant
            return changed_parking

        if self.next_vertex and dist(self.center_pos, self.next_vertex.pos) <= 0.05:    # get new next vertex
            print("got to", self.center_pos, self.rotation)
            self.calculations_clock = 0
        if self.calculations_clock == consts.reset_count_time:  # reset calculation clock by time
            self.calculations_clock = 0

        if self.calculations_clock == 0:
            if consts.drawing:
                self.trace.append(self.center_pos)
            next_vertex = self.prm.next_in_path(self.current_vertex)    # calculate next vertex
            if next_vertex is None:  # try to return backwards if we can't drive forwards
                if (not self.is_backwards_driving) or dist(self.center_pos, self.next_vertex.pos) <= 0.1:
                    self.next_vertex = self.prev_vertex.pop()
                    self.is_backwards_driving = True
                    print("popped")
            else:
                print("forward")
                self.next_vertex = next_vertex
                self.is_backwards_driving = False

        if self.calculations_clock % consts.calculate_action_time == 0: # calculate needed action to get to the next
            # vertex
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

    def update_state(self) -> bool:
        """
        updates the state of the car after a step,
        updates variables and checks if the car collided with something, or got to it's goal.
        :return: true if this car is done
        """
        # updating map;
        self.base_pos, quaternions = p.getBasePositionAndOrientation(self.car_model)
        self.rotation = p.getEulerFromQuaternion(quaternions)[2]
        self.center_pos = PRM.pos_to_car_center(np.array(self.base_pos[:2]), self.rotation)

        # checking if collided or finished
        if self.check_collision(self.car_model, self.bodies + [other.car_model for other in self.cars]):
            self.crashed = True
        if dist(self.center_pos, self.end_point) < consts.min_dist_to_target:
            self.finished = True
        if self.base_pos[2] > 0.1:
            self.crashed = True

        self.base_pos = self.base_pos[:2]

        prev_vertex = self.current_vertex
        self.current_vertex = self.prm.get_closest_vertex(self.center_pos, self.rotation)
        if self.current_vertex != prev_vertex and not self.is_backwards_driving:    # TODO: increase distance or time for this
            self.prev_vertex.append(prev_vertex)
        self.scan_environment()
        return self.crashed or self.finished

    def create_car_model(self) -> Tuple[int, List[int], List[int]]:
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
        p.resetBasePositionAndOrientation(
            car, base_position, p.getQuaternionFromEuler([0, 0, self.rotation])
        )

        return car, wheels, steering
