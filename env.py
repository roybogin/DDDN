import os
from typing import Set

import pybullet as p
import pybullet_data as pd
from gym import spaces
from gym.utils import seeding

import PRM
import map_create
import mazes
from helper import *
from scan_to_map import Map


def make_env(index, seed=None):
    """
    Utility function for multiprocessing env.
    :param index: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :return: function that creates the environment
    """

    return lambda: CarEnv(index, seed)
    # def _init():
    #     env = CarEnv(index, seed)
    #     return env
    #
    # return _init


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
    def __init__(self, index, seed, size=10):
        super(CarEnv, self).__init__()
        self.car_center = None
        self.trace = None
        self.current_vertex = None
        self.next_vertex = None
        self.count = None
        self.end_vertex = None
        self.direction = None
        self.distances_from_pos = None
        self.index = index  # index of environment in multiprocessing
        self.maze_idx = None  # index of the maze we chosen TODO: delete after generalizing
        self.np_random = None  # object for randomizing values according to a seed
        self.initial_distance_to_target = None  # the initial distance between the car and target
        self.total_score = None  # total score for the run
        self.rotation_trig = None  # trigonometry values on the car orientation
        self.speed = None  # speed of the car
        self.velocity = None  # velocity of the car (speed in both axes)
        self.angular_velocity = None  # angular velocity of the car
        self.rotation = None  # rotation of the car (in radians)
        self.pos = None  # position of the car on the map
        self.start_point = None  # starting point of the map
        self.discovered = None  # binary matrix which shows what regions of the map were seen by the car
        self.discrete_partial_map = None  # the map as perceived by the car - 0 means unexplored or empty and 1 means that the map has
        # a wall
        self.bodies = None  # list of collision bodies the maze walls and perimeter
        self.last_speed = None  # speed of the car in the previous frame
        self.run_time = None  # time of the run
        self.finished = None  # did the car get to the goal
        self.new_discovered = None  # new indices that were discovered by the car
        self.crashed = None  # did the car crash
        self.swivel = None  # swivel of the car - angle of steering wheel
        self.acceleration = None  # acceleration of the car (difference in speed)
        self.end_point = None  # end point of the map
        self.maze = None  # walls of the maze - list of points
        self.borders = None  # the maze borders - object IDs
        self.curr_goal = None  # goal for now (close to current position)
        self.distances_to_end = None  # minimum distance in blocks (in the maze) to the end for each block
        self.map_changed = None  # did the perceived map change (we need to recalculate the distances)
        self.prev_pos = None
        self.segments_partial_map: Map | None = None
        self.scanned_indices = None  # new indices since scan
        self.hits = None

        self.prm = PRM.PRM((int((2 * consts.size_map_quarter) // consts.vertex_offset), int((2 * consts.size_map_quarter) // consts.vertex_offset)))

        self.generate_graph()

        '''structure of an observation
                "position": 2,
                "goal": 2,
                "velocity": 2,
                "angular_velocity": 2,
                "swivel": 1,
                "rotation_trigonometry": 2,
                "acceleration": 1
                '''
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(-size, size, shape=(2,), dtype=np.float32),
                "goal": spaces.Box(-size, size, shape=(2,), dtype=np.float32),
                "velocity": spaces.Box(
                    -consts.max_velocity, consts.max_velocity, shape=(2,), dtype=np.float32
                ),
                "angular_velocity": spaces.Box(
                    -consts.max_velocity, consts.max_velocity, shape=(2,), dtype=np.float32
                ),
                "swivel": spaces.Box(
                    -consts.max_steer, consts.max_steer, shape=(1,), dtype=np.float32
                ),
                "rotation_trigonometry": spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                "acceleration": spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        self.car_model = None  # pybullet ID of the car
        self.wheels = None  # pybullet ID of the wheels for setting speed
        self.steering = None  # pybullet ID of the wheels for steering
        self.obstacles = []  # list of obstacle IDs in pybullet
        self.bodies = []  # list of all collision body IDs in pybullet
        self.seed(seed)
        self.maze, self.end_point, self.start_point = self.get_new_maze()

        print(self.end_point)
        self.end_point = self.prm.set_end(np.array(self.end_point[:2]))
        print(f'new end is {self.end_point}')
        self.prm.dijkstra(self.prm.end)

        print(len([v for v in self.prm.graph.vertices if self.prm.distances[v][0] == np.inf]))
        self.current_vertex = self.prm.get_closest_vertex(np.array(self.start_point[:2]), 0)
        current_car_pos = PRM.car_center_to_pos(self.current_vertex.pos, 0)
        self.start_point = [current_car_pos[0], current_car_pos[1], 0]

        self.prm.draw_path(self.current_vertex)

        self.next_vertex = None

        self.start_env()
        self.reset()

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

    def calculate_next_goal(self):
        pass

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
        need_recalculate = False
        directions = [2 * np.pi * i / consts.ray_amount for i in range(consts.ray_amount)]
        new_map_discovered = self.discovered
        vertex_removal_radius = math.ceil(0.4 / consts.vertex_offset)
        edge_removal_radius = np.ceil(self.prm.res / consts.vertex_offset)
        problematic_vertices: Set[PRM.Vertex] = set()
        problematic_edges: Set[PRM.Edge] = set()
        new_segments = []
        for direction in directions:

            did_hit, start, end = self.ray_cast(
                self.car_model, [0, 0, 0.5],
                [-consts.ray_length * np.cos(direction), -consts.ray_length * np.sin(direction), 0]
            )
            if did_hit:
                self.hits.append((end[0], end[1]))
                if len(self.hits) == consts.max_hits_before_calculation:
                    self.segments_partial_map.add_points_to_map(self.hits)
                    self.hits = []
                    new = self.segments_partial_map.new_segments
                    new_segments += new
                    for segment in new:
                        for point in segment:
                            for block in block_options(map_index_from_pos(point), vertex_removal_radius,
                                                       np.shape(self.discovered)):
                                for vertex in self.prm.vertices[block[0]][block[1]]:
                                    if vertex and not self.segments_partial_map.check_state(vertex):
                                        if self.prm.remove_vertex(vertex):
                                            print('remove vertex')
                                            need_recalculate = True
            self.new_discovered = add_discovered_matrix(new_map_discovered, start, end)
        self.discovered = new_map_discovered
        for segment in new_segments:
            for point in segment:
                for block in block_options(map_index_from_pos(point), edge_removal_radius, np.shape(self.discovered)):
                    problematic_vertices.update(self.prm.vertices[block[0]][block[1]])

        for vertex in problematic_vertices:
            for edge in vertex.edges:
                if edge.v1 in problematic_vertices and edge.v2 in problematic_vertices:
                    problematic_edges.add(edge)
        if len(problematic_edges) != 0:
            print('problematic')
        for segment in new_segments:
            if len(segment) == 1:
                for edge in problematic_edges:
                    if perpendicularDistance(segment[0], edge.v1.pos, edge.v2.pos) < consts.width + 2 * consts.epsilon:
                        if self.prm.graph.remove_edge(edge):
                            print('remove edge')
                            need_recalculate = True
            else:
                for i in range(len(segment) - 1):
                    for edge in problematic_edges:
                        if distance_between_lines(segment[i], segment[i+1], edge.v1.pos, edge.v2.pos) < consts.width + 2 * consts.epsilon:
                            if self.prm.graph.remove_edge(edge):
                                print('remove edge')
                                need_recalculate = True
        if need_recalculate:
            print('recalc')
            self.prm.dijkstra(self.prm.end)


    def reset(self):
        """
        resets the environment
        """
        self.trace = []
        self.count = 0
        self.hits = []
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

        self.initial_distance_to_target = dist(self.start_point[:2], self.end_point[:2])
        self.finished = False
        self.crashed = False
        self.pos = self.start_point
        self.prev_pos = self.start_point[:2]
        self.last_speed = 0
        self.total_score = 0

        self.obstacles = map_create.create_map(self.maze, self.end_point, epsilon=consts.epsilon, client=p)
        self.bodies = self.borders + self.obstacles
        self.discrete_partial_map = [[0 for _ in range(int((2 * consts.size_map_quarter) // consts.vertex_offset))] for _ in
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

        self.map_changed = True
        self.calculate_next_goal()
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
        if ray_test[0][3] == (0, 0, 0):
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

        self.scan_environment()

        if consts.print_runtime:
            print(self.run_time)

        self.car_center = PRM.pos_to_car_center(np.array(self.pos[:2]), self.rotation)

        if self.next_vertex and dist(self.car_center, self.next_vertex.pos) <= 0.05:
            print("got to", self.car_center, self.rotation)
            self.count = 0
        if self.count == 100:
            self.count = 0

        if self.count == 0:
            self.trace.append(self.car_center)
            self.next_vertex = self.prm.next_in_path(self.car_center, self.rotation)
            if not self.next_vertex:
                self.next_vertex = self.current_vertex

        self.count += 1

        transformed = self.prm.transform_by_values(self.car_center, self.rotation, self.next_vertex)
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

        self.scan_environment()

        # updating map;
        self.pos, quaternions = p.getBasePositionAndOrientation(self.car_model)

        self.velocity, self.angular_velocity = p.getBaseVelocity(self.car_model)
        self.velocity = self.velocity[:2]
        self.angular_velocity = self.angular_velocity[:2]

        # checking if collided or finished
        if self.check_collision(self.car_model, self.bodies):
            self.crashed = True
        if dist(self.car_center, self.end_point) < consts.min_dist_to_target:
            self.finished = True
        # # getting values for NN
        if self.pos[2] > 0.1:
            self.crashed = True

        self.pos = self.pos[:2]
        self.rotation = p.getEulerFromQuaternion(quaternions)[2]

        self.last_speed = self.speed
        self.speed = norm(self.velocity)
        self.acceleration = self.speed - self.last_speed

        self.rotation_trig = [np.cos(self.rotation), np.sin(self.rotation)]

        # saving for later
        swivel_states = p.getJointStates(self.car_model, self.steering)
        angles = [state[0] for state in swivel_states]
        cot_delta = (1/np.tan(angles[0]) + 1/np.tan(angles[1])) / 2
        self.swivel = np.arctan(1/cot_delta)

        score = 0
        self.total_score += score

        self.prev_pos = self.pos

        self.calculate_next_goal()

        if not (self.crashed or self.finished):
            return self.get_observation(), score, False, {}
        self.trace.append(self.car_center)
        if self.finished:
            self.trace.append(self.end_point)

        plt.plot([a for a, _ in self.trace], [a for _, a in self.trace], c='orange', label='actual path')
        plt.title(f'maze {self.maze_idx} - time {self.run_time}')
        plt.legend()
        plt.show()
        # self.segments_partial_map.show()

        if self.crashed:
            print(
                f"crashed maze {self.maze_idx}"
                f" - distance is {dist(self.pos, self.end_point)}"
                f" - time {self.run_time}")
            p.disconnect()
            exit(666)
        if self.finished:
            print(
                f"finished maze {self.maze_idx}"
                f" - time {self.run_time}")
            p.disconnect()
            exit(0)
        return self.get_observation(), score, True, {}

    def get_observation(self):
        """
        get the current observation
        :return: dictionary matching the observation
        """
        observation = {
            "position": np.array(self.pos[:2], dtype=np.float32),
            "goal": np.array(self.curr_goal, dtype=np.float32),
            "velocity": np.array(self.velocity, dtype=np.float32),
            "angular_velocity": np.array(self.angular_velocity, dtype=np.float32),
            "swivel": np.array([self.swivel], dtype=np.float32),
            "rotation_trigonometry": np.array(self.rotation_trig, dtype=np.float32),
            "acceleration": np.array([self.acceleration], dtype=np.float32),
        }
        return observation

    def set_car_position(self, position):
        """
        sets the car in a position
        :param position: position to place the car at
        :return:
        """
        p.resetBasePositionAndOrientation(self.car_model, position, [0.0, 0.0, 0.0, 1.0])
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
        self.maze_idx = 5
        maze, start, end = mazes.empty_set[self.maze_idx]
        return maze, end, start


def main():
    env = CarEnv(0, consts.seed)
    while True:
        env.step()


if __name__ == "__main__":
    main()
