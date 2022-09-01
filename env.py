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


class CarEnv:
    def __init__(self, index, seed):
        super(CarEnv, self).__init__()
        self.trace = None
        self.need_recalculate = False
        self.current_vertex = None
        self.next_vertex = None
        self.count = None
        self.direction = None
        self.index = index  # index of environment in multiprocessing
        self.maze_idx = None  # index of the maze we chosen TODO: delete after generalizing
        self.np_random = None  # object for randomizing values according to a seed
        self.rotation_trig = None  # trigonometry values on the car orientation
        self.speed = None  # speed of the car
        self.velocity = None  # velocity of the car (speed in both axes)
        self.angular_velocity = None  # angular velocity of the car
        self.rotation = None  # rotation of the car (in radians)
        self.base_pos = None
        self.center_pos = None
        self.start_point = None  # starting point of the map
        self.discovered = None  # binary matrix which shows what regions of the map were seen by the car
        self.discrete_partial_map = None  # the map as perceived by the car - 0 means unexplored or empty and 1 means that the map has
        # a wall
        self.bodies = None  # list of collision bodies the maze walls and perimeter
        self.last_speed = None  # speed of the car in the previous frame
        self.run_time = None  # time of the run
        self.finished = None  # did the car get to the goal
        self.crashed = None  # did the car crash
        self.swivel = None  # swivel of the car - angle of steering wheel
        self.acceleration = None  # acceleration of the car (difference in speed)
        self.end_point = None  # end point of the map
        self.maze = None  # walls of the maze - list of points
        self.borders = None  # the maze borders - object IDs
        self.curr_goal = None  # goal for now (close to current position)
        self.segments_partial_map: Map | None = None
        self.hits = None

        self.prm = PRM.PRM((int((2 * consts.size_map_quarter) // consts.vertex_offset),
                            int((2 * consts.size_map_quarter) // consts.vertex_offset)))

        self.generate_graph()

        self.car_model = None  # pybullet ID of the car
        self.wheels = None  # pybullet ID of the wheels for setting speed
        self.steering = None  # pybullet ID of the wheels for steering
        self.obstacles = []  # list of obstacle IDs in pybullet
        self.bodies = []  # list of all collision body IDs in pybullet
        self.seed(seed)
        self.maze, self.end_point, self.start_point = self.get_new_maze()

        # real initialization
        print(self.end_point)
        self.end_point = self.prm.set_end(np.array(self.end_point[:2]))
        self.current_vertex = self.prm.get_closest_vertex(self.start_point, 0)
        print(f'new end is {self.end_point}')
        self.start_point = [self.current_vertex.pos[0], self.current_vertex.pos[1], 0]
        self.center_pos = self.current_vertex.pos

        self.next_vertex = None

        self.start_env()
        self.reset()

        self.scan_environment()

        self.prm.init_d_star(self.current_vertex)
        self.prm.d_star.compute_shortest_path(self.current_vertex)
        self.prm.draw_path(self.current_vertex)

    def generate_graph(self):
        print("generating graph")
        self.prm.generate_graph()

        print(self.prm.graph.n, self.prm.graph.e)

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

        self.car_model, self.wheels, self.steering = self.create_car_model()

    def add_borders(self):
        """
        adds the boarders to the maze
        """
        self.borders = map_create.create_poly_wall(consts.map_borders, epsilon=consts.epsilon, client=p)

    def remove_all_bodies(self):
        """
        removes all collision bodies from the map
        :return:
        """
        for body in self.bodies:
            p.removeBody(body)
        self.bodies = []

    def scan_environment(self):
        """
        scans the environment and updates the discovery values
        :return:
        """
        directions = [2 * np.pi * i / consts.ray_amount for i in range(consts.ray_amount)]
        new_map_discovered = self.discovered
        vertex_removal_radius = math.ceil(0.4 / consts.vertex_offset)
        edge_removal_radius = np.ceil(self.prm.res / consts.vertex_offset)
        problematic_vertices: Set[PRM.Vertex] = set()
        problematic_edges: Set[PRM.Edge] = set()
        new_segments = []
        for i, direction in enumerate(directions):

            did_hit, start, end = self.ray_cast(
                self.car_model, [0, 0, 0.5],
                [-consts.ray_length * np.cos(direction), -consts.ray_length * np.sin(direction), 0]
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
                            for block in block_options(map_index_from_pos(point), vertex_removal_radius,
                                                       np.shape(self.discovered)):
                                for vertex in self.prm.vertices[block[0]][block[1]]:
                                    if vertex and not self.segments_partial_map.check_state(vertex):
                                        if self.prm.remove_vertex(vertex):
                                            self.need_recalculate = True
            add_discovered_matrix(new_map_discovered, start, end)
        self.discovered = new_map_discovered
        for segment in new_segments:
            for point in segment:
                for block in block_options(map_index_from_pos(point), edge_removal_radius, np.shape(self.discovered)):
                    problematic_vertices.update(self.prm.vertices[block[0]][block[1]])

        for vertex in problematic_vertices:
            if vertex is None:
                continue
            for edge in vertex.in_edges | vertex.out_edges:
                if edge.src in problematic_vertices and edge.dst in problematic_vertices:
                    problematic_edges.add(edge)
        for segment in new_segments:
            for i in range(len(segment) - 1):
                for edge in problematic_edges:
                    if edge.active and distance_between_lines(segment[i], segment[i + 1], edge.src.pos, edge.dst.pos) < \
                            consts.width + 2 * consts.epsilon:
                        if self.prm.remove_edge(edge):
                            edge.active = False
                            self.need_recalculate = True
        # if self.run_time % 20 == 0 and self.need_recalculate:
        #     print('recalc', self.center_pos, self.prm.graph.n, self.prm.graph.e)
        #     self.prm.dijkstra(self.prm.end)
        #     self.count = 0
        #     print('distance to end is', self.prm.distances[self.current_vertex])
        #     self.need_recalculate = False


    def reset(self):
        """
        resets the environment
        """
        self.trace = []
        self.count = 0
        self.hits = [[] for _ in range(consts.ray_amount)]
        self.remove_all_bodies()
        self.add_borders()

        self.segments_partial_map = Map([consts.map_borders.copy()])

        self.swivel = 0
        self.speed = 0
        self.velocity = [0, 0]
        self.angular_velocity = [0, 0]
        self.acceleration = 0
        self.rotation = 0
        self.run_time = 0

        self.finished = False
        self.crashed = False
        self.last_speed = 0

        self.obstacles = map_create.create_map(self.maze, self.end_point, epsilon=consts.epsilon, client=p)
        self.bodies = self.borders + self.obstacles
        self.discrete_partial_map = [[0 for _ in range(int((2 * consts.size_map_quarter) // consts.vertex_offset))] for
                                     _ in
                                     range(int((2 * consts.size_map_quarter) // consts.vertex_offset))]
        for i in range(int((2 * consts.size_map_quarter) // consts.vertex_offset)):
            self.discrete_partial_map[i][0] = 1
            self.discrete_partial_map[0][i] = 1
            self.discrete_partial_map[i][int((2 * consts.size_map_quarter) // consts.vertex_offset) - 1] = 1
            self.discrete_partial_map[int((2 * consts.size_map_quarter) // consts.vertex_offset) - 1][i] = 1
        self.set_car_position(self.start_point)
        self.discovered = [
            [0 for _ in range(int((2 * consts.size_map_quarter) // consts.vertex_offset))]
            for _ in range(int((2 * consts.size_map_quarter) // consts.vertex_offset))
        ]
        self.rotation_trig = [np.cos(self.rotation), np.sin(self.rotation)]



        return self.get_observation()

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

    def seed(self, seed=None):
        """
        set a seed for the randomness
        :param seed: the np.random seed
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        print(seed)
        return [seed]

    def check_collision(self, car_model, obstacles, margin=0,
                        max_distance=1.0):
        """
        did the car collide with an obstacle
        :param car_model: car ID
        :param obstacles: list of body IDs to check collision of the car with
        :param margin: margin of error for collision - if the distance is smaller than the margin - the car collided
        :param max_distance: distance from the car to search for collisions in
        :return: did the car collide with an obstacle
        """
        for ob in obstacles:
            closest_points = p.getClosestPoints(car_model, ob,
                                                distance=max_distance)
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

        if consts.print_runtime and self.run_time % 500 == 0:
            print('time:', self.run_time, 'pos:', self.center_pos)

        if self.next_vertex and dist(self.center_pos, self.next_vertex.pos) <= 0.05:
            print("got to", self.center_pos, self.rotation)
            self.count = 0
        if self.count == 100:
            self.count = 0

        if self.count == 0:
            self.trace.append(self.center_pos)
            self.next_vertex = self.prm.next_in_path(self.current_vertex)

        self.count += 1

        transformed = self.prm.transform_by_values(self.center_pos, self.rotation, self.next_vertex)
        x_tag, y_tag = transformed[0][0], transformed[0][1]

        radius = np.sqrt(self.prm.radius_x_y_squared(x_tag, y_tag))
        delta = np.sign(y_tag) * np.arctan(consts.length / radius)

        '''rad_1 = np.sqrt(radius ** 2 - consts.a_2 ** 2)
        delta_inner = np.arctan(consts.length/(rad_1 - consts.width/2))
        delta_outer = np.arctan(consts.length/(rad_1 + consts.width/2))

        if y_tag >= 0:
            rotation = [delta_inner, delta_outer]
        else:
            rotation = [-delta_outer, -delta_inner]'''
        rotation = [delta, delta]

        rotation = np.array(rotation)

        # print(self.car_center, self.next_vertex.pos, self.end_point)

        action = [np.sign(x_tag) / (1 + 4 * abs(delta)), rotation]

        # updating target velocity and steering angle
        wanted_speed = action[0] * consts.max_velocity
        wanted_steering_angle = action[1]
        wanted_steering_angle = np.sign(wanted_steering_angle) * np.minimum(np.abs(wanted_steering_angle),
                                                                            consts.max_steer)
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
                self.car_model,
                steer,
                p.POSITION_CONTROL,
                targetPosition=angle,
            )
        p.stepSimulation()

        self.run_time += 1

        # updating map;
        self.base_pos, quaternions = p.getBasePositionAndOrientation(self.car_model)
        self.rotation = p.getEulerFromQuaternion(quaternions)[2]
        self.center_pos = PRM.pos_to_car_center(np.array(self.base_pos[:2]), self.rotation)

        self.velocity, self.angular_velocity = p.getBaseVelocity(self.car_model)
        self.velocity = self.velocity[:2]
        self.angular_velocity = self.angular_velocity[:2]

        # checking if collided or finished
        if self.check_collision(self.car_model, self.bodies):
            self.crashed = True
        if dist(self.center_pos, self.end_point) < consts.min_dist_to_target:
            self.finished = True
        # # getting values for NN
        if self.base_pos[2] > 0.1:
            self.crashed = True

        self.base_pos = self.base_pos[:2]

        self.last_speed = self.speed
        self.speed = norm(self.velocity)
        self.acceleration = self.speed - self.last_speed

        self.rotation_trig = [np.cos(self.rotation), np.sin(self.rotation)]

        # saving for later
        swivel_states = p.getJointStates(self.car_model, self.steering)
        angles = [state[0] for state in swivel_states]
        cot_delta = (1 / np.tan(angles[0]) + 1 / np.tan(angles[1])) / 2
        self.swivel = np.arctan(1 / cot_delta)

        prev_vertex = self.current_vertex
        self.current_vertex = self.prm.get_closest_vertex(self.center_pos, self.rotation)

        self.scan_environment()

        if self.need_recalculate:
            self.prm.d_star.k_m += d_star.h(prev_vertex, self.current_vertex)
            for edge in self.prm.deleted_edges:
                u = edge.src
                rhs = self.prm.d_star.rhs
                rhs[u] = min(rhs[u], self.prm.d_star.g[u])
                self.prm.d_star.update_vertex(u)

            print('recalc path, pos:', self.center_pos)
            self.prm.d_star.compute_shortest_path(self.current_vertex)

        if self.run_time >= consts.max_time:
            print(
                f"out of time in maze {self.maze_idx}"
                f" - distance is {dist(self.center_pos, self.end_point)}")
            p.disconnect()
            self.trace.append(self.center_pos)
            plt.plot([a for a, _ in self.trace], [a for _, a in self.trace], label='actual path')
            plt.scatter(self.center_pos[0], self.center_pos[1], c='red')
            plt.title(f'maze {self.maze_idx} - time {self.run_time}')
            # self.prm.draw_path(self.current_vertex, ' end')
            plt.legend()
            plt.show()
            self.segments_partial_map.show()

            return True

        if not (self.crashed or self.finished):
            return False
        self.trace.append(self.center_pos)
        if self.finished:
            self.trace.append(self.end_point)

        plt.plot([a for a, _ in self.trace], [a for _, a in self.trace], label='actual path')
        plt.title(f'maze {self.maze_idx} - time {self.run_time}')
        plt.legend()
        plt.show()
        self.segments_partial_map.show()

        if self.crashed:
            print(
                f"crashed maze {self.maze_idx}"
                f" - distance is {dist(self.center_pos, self.end_point)}"
                f" - time {self.run_time}")
            p.disconnect()
            return True
        if self.finished:
            print(
                f"finished maze {self.maze_idx}"
                f" - time {self.run_time}")
            p.disconnect()
            return True
        return False

    def get_observation(self):
        """
        get the current observation
        :return: dictionary matching the observation
        """
        return None

    def set_car_position(self, position):
        """
        sets the car in a position
        :param position: position to place the car at
        :return:
        """
        base_position = list(PRM.car_center_to_pos(np.array(position), 0)) + [0]
        p.resetBasePositionAndOrientation(self.car_model, base_position, [0.0, 0.0, 0.0, 1.0])
        p.resetBaseVelocity(self.car_model, [0, 0, 0], [0, 0, 0])

    def create_car_model(self):
        """
        create the car in pybullet
        :return: the car ID and important joint IDs for steering and setting speed
        """
        car = p.loadURDF(
            os.path.join(pd.getDataPath(), "racecar/racecar.urdf")
        )
        inactive_wheels = [3, 5, 7]
        wheels = [2]

        for wheel in inactive_wheels:
            p.setJointMotorControl2(
                car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0
            )

        steering = [4, 6]

        p.resetBasePositionAndOrientation(car, [0, 0, 0], [0, 0, 0, 1])

        return car, wheels, steering

    def get_new_maze(self):
        """
        gets a new maze for the run
        :return: maze (a set of polygonal lines), a start_point and end_point(3D vectors)
        """
        self.maze_idx = self.np_random.randint(0, len(mazes.empty_set))
        self.maze_idx = 'with small block'
        maze, start, end = mazes.default_data_set[1] # mazes.empty_set[self.maze_idx] #
        return maze, end, start


def main():
    t0 = time.time()
    stop = False
    env = CarEnv(0, consts.seed)
    while not stop:
        stop = env.step()
    print(f'total time: {time.time() - t0}')


if __name__ == "__main__":
    main()
