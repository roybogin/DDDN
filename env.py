import glob
import os
from datetime import datetime
from typing import Callable, Any, List

import gym
import pybullet as p
import pybullet_data as pd
from gym import spaces
from gym.utils import seeding
from pybullet_utils import bullet_client
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

import map_create
import mazes
from helper import *


def make_env(index: int, seed: int | None = None) -> Callable:
    """
    Utility function for multiprocessing env.
    :param index: index of the subprocess
    :param seed: the initial seed for RNG
    :return: function that creates the environment
    """

    return lambda: CarEnv(index, seed)
    # def _init():
    #     env = CarEnv(index, seed)
    #     return env
    #
    # return _init


def add_discovered_matrix(discovered_matrix: consts.binary_matrix, start: consts.vector, end: consts.vector) -> float:
    """
    updates the discovered matrix by drawing a line on it matching the endpoints of the raycast
    :param discovered_matrix: matrix that represents the discovered areas by the car
    :param start: start of the raycast
    :param end: end of the raycast
    :return: the percentage of discovered area
    """
    x0, y0 = map_index_from_pos(start)
    x1, y1 = map_index_from_pos(end)
    plot_line(x0, y0, x1, y1, discovered_matrix)

    return (
            sum([sum(discovered_matrix[i]) for i in
                 range(len(discovered_matrix))])
            / len(discovered_matrix) ** 2
    )


# car api to use with the SAC algorithm
class CarEnv(gym.Env):
    def __init__(self, index: int, seed: int | None):
        super(CarEnv, self).__init__()
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
        self.map = None  # the map as perceived by the car - 0 means unexplored or empty and 1 means that the map has
        # a wall
        self.bodies = None  # list of collision bodies the maze walls and perimeter
        self.last_speed = None  # speed of the car in the previous frame
        self.run_time = None  # time of the run
        self.finished = None  # did the car get to the goal
        self.map_discovered = None  # percentage of the map that was discovered by the car
        self.crashed = None  # did the car crash
        self.swivel = None  # swivel of the car - angle of steering wheel
        self.acceleration = None  # acceleration of the car (difference in speed)
        self.end_point = None  # end point of the map
        self.maze = None  # walls of the maze - list of points
        self.borders = None  # the maze borders - object IDs
        self.p1 = None  # separate pybullet client for multiprocessing
        self.curr_goal = None   # goal for now (close to current position)
        self.distances_to_end = None  # minimum distance in blocks (in the maze) to the end for each block
        self.map_changed = None  # did the perceived map change (we need to recalculate the distances)
        self.prev_pos = None

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
                "position": spaces.Box(-consts.size_map_quarter, consts.size_map_quarter, shape=(2,), dtype=np.float32),
                "goal": spaces.Box(-consts.size_map_quarter, consts.size_map_quarter, shape=(2,), dtype=np.float32),
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
        self.start_env()
        self.seed(seed)
        self.reset()

    def start_env(self) -> None:
        """
        start the pybullet environment and create the car
        :return:
        """
        self.p1 = bullet_client.BulletClient()
        self.p1.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.p1.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self.p1.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.p1.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        self.p1.setAdditionalSearchPath(pd.getDataPath())
        self.p1.setGravity(0, 0, -10)

        self.p1.setRealTimeSimulation(consts.use_real_time)
        self.p1.setTimeStep(consts.time_step)
        self.p1.loadSDF(os.path.join(pd.getDataPath(), "stadium.sdf"))
        self.p1.resetDebugVisualizerCamera(
            cameraDistance=consts.cameraDistance,
            cameraYaw=consts.cameraYaw,
            cameraPitch=consts.cameraPitch,
            cameraTargetPosition=consts.cameraTargetPosition,
        )

        self.car_model, self.wheels, self.steering = self.create_car_model()

    def calculate_next_goal(self):
        end_block = map_index_from_pos(self.end_point)
        curr_block = map_index_from_pos(self.pos)
        if end_block == curr_block or dist(self.pos, self.end_point) <= consts.block_size/2:
            self.curr_goal = self.end_point
            return
        if self.map_changed:
            self.distances_to_end = calculate_distances(self.map, end_block)
        self.map_changed = False  # no need to recalculate the distances
        neighbors = get_neighbors(curr_block, np.shape(self.map))
        next_block = min(neighbors, key=lambda idx: (self.distances_to_end[idx], dist(self.pos, pos_from_map_index(
            idx))))
        self.curr_goal = pos_from_map_index(next_block)

    def add_borders(self) -> None:
        """
        adds the boarders to the maze
        """
        self.borders = map_create.create_poly_wall(consts.map_borders, epsilon=0.1, client=self.p1)

    def remove_all_bodies(self) -> None:
        """
        removes all collision bodies from the map
        :return:
        """
        for body in self.bodies:
            self.p1.removeBody(body)
        self.bodies = []

    def scan_environment(self) -> None:
        """
        scans the environment and updates the discovery values
        :return:
        """
        # TODO: get rid of new_map_discovered? - doesn't copy
        directions = [2 * np.pi * i / consts.ray_amount for i in range(consts.ray_amount)]
        new_map_discovered = self.discovered
        for direction in directions:

            did_hit, start, end = self.ray_cast(
                self.car_model, [0, 0, 0.5],
                [-consts.ray_length * np.cos(direction), -consts.ray_length * np.sin(direction), 0]
            )
            if did_hit:
                x1, y1 = map_index_from_pos(end)
                if self.map[x1][y1] != 1:
                    self.map_changed = True
                self.map[x1][y1] = 1
            self.map_discovered = add_discovered_matrix(new_map_discovered, start, end)

        self.discovered = new_map_discovered

    def reset(self) -> dict:
        """
        resets the environment
        :return: The observation for the environment
        """
        self.remove_all_bodies()
        self.add_borders()

        self.maze, self.end_point, self.start_point = self.get_new_maze()

        self.swivel = 0
        self.speed = 0
        self.velocity = [0, 0]
        self.angular_velocity = [0, 0]
        self.acceleration = 0
        self.rotation = 0
        self.run_time = 0

        self.initial_distance_to_target = dist(self.start_point[:2], self.end_point[:2])
        self.map_discovered = 0
        self.finished = False
        self.crashed = False
        self.pos = self.start_point
        self.prev_pos = self.start_point[:2]
        self.last_speed = 0
        self.total_score = 0

        self.obstacles = map_create.create_map(self.maze, self.end_point, epsilon=0.1, client=self.p1)
        self.bodies = self.borders + self.obstacles
        self.map = [[0 for _ in range(int((2 * consts.size_map_quarter) // consts.block_size))] for _ in
                    range(int((2 * consts.size_map_quarter) // consts.block_size))]
        for i in range(int((2 * consts.size_map_quarter) // consts.block_size)):
            self.map[i][0] = 1
            self.map[0][i] = 1
            self.map[i][int((2 * consts.size_map_quarter) // consts.block_size) - 1] = 1
            self.map[int((2 * consts.size_map_quarter) // consts.block_size) - 1][i] = 1
        self.set_car_position(self.start_point)
        self.discovered = [
            [0 for _ in range(int((2 * consts.size_map_quarter) // consts.block_size))]
            for _ in range(int((2 * consts.size_map_quarter) // consts.block_size))
        ]
        self.rotation_trig = [np.cos(self.rotation), np.sin(self.rotation)]

        self.scan_environment()

        self.map_changed = True
        self.calculate_next_goal()

        return self.get_observation()

    def ray_cast(self, car: int, offset: consts.vector, direction: consts.vector) -> Tuple[bool, consts.vector,
                                                                                           consts.vector]:
        """
        generates a raycast in a given direction
        :param car: car ID
        :param offset: offset from the car to start the raycast
        :param direction: direction of ray
        :return: (did the ray collide with an obstacle, start position of the ray, end position of the ray)
        """
        pos, quaternions = self.p1.getBasePositionAndOrientation(car)
        euler = self.p1.getEulerFromQuaternion(quaternions)
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
        ray_test = self.p1.rayTest(start, end)
        if ray_test[0][3] == (0, 0, 0):
            return False, start[:2], end[:2]
        else:
            return True, start[:2], ray_test[0][3]

    def seed(self, seed: int | None = None) -> List[int | None]:
        """
        set a seed for the randomness
        :param seed: the np.random seed
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calculate_reward(self) -> float:
        """
        calculate the reward of the step
        :return: the reward for the step
        """
        reward = (
                consts.TIME_PENALTY +
                self.crashed * consts.CRASH_PENALTY * (1 - 0.5 * self.run_time / consts.max_time) +
                self.finished * consts.FINISH_REWARD +
                consts.GOAL_DIST_REWARD * (1 - dist(self.pos, self.curr_goal)/dist(self.prev_pos, self.curr_goal))
        )
        return reward

    def check_collision(self, car_model: int, obstacles: List[int], margin: float = 0,
                        max_distance: float = 1.0) -> bool:
        """
        did the car collide with an obstacle
        :param car_model: car ID
        :param obstacles: list of body IDs to check collision of the car with
        :param margin: margin of error for collision - if the distance is smaller than the margin - the car collided
        :param max_distance: distance from the car to search for collisions in
        :return: did the car collide with an obstacle
        """
        for ob in obstacles:
            closest_points = self.p1.getClosestPoints(car_model, ob,
                                                      distance=max_distance)
            closest_points = [
                a for a in closest_points if not (a[1] == a[2] == car_model)
            ]
            if len(closest_points) != 0:
                distance = np.min([pt[8] for pt in closest_points])
                if distance < margin:
                    return True
        return False

    def step(self, action: Tuple[float]) -> Tuple[dict, float, bool, Any]:
        """
        runs the simulation one step
        :param action: the action to preform (tuple of speed change and steer change)
        :return: (next observation, reward, did the simulation finish, info)
        """
        if consts.print_runtime:
            print(self.run_time)

        change_steering_angle = action[0] * consts.max_steer
        change_target_velocity = action[1] * consts.max_velocity

        # updating target velocity and steering angle
        wanted_speed = self.speed + change_target_velocity * consts.speed_scalar
        wanted_steering_angle = self.swivel + change_steering_angle * consts.steer_scalar
        if abs(wanted_steering_angle) > consts.max_steer:
            wanted_steering_angle = consts.max_steer * np.sign(wanted_steering_angle)
        if abs(wanted_speed) > consts.max_velocity:
            wanted_speed = consts.max_velocity * np.sign(wanted_speed)

        # moving
        for wheel in self.wheels:
            self.p1.setJointMotorControl2(
                self.car_model,
                wheel,
                self.p1.VELOCITY_CONTROL,
                targetVelocity=wanted_speed,
                force=consts.max_force,
            )

        for steer in self.steering:
            self.p1.setJointMotorControl2(
                self.car_model,
                steer,
                self.p1.POSITION_CONTROL,
                targetPosition=wanted_steering_angle,
            )
        self.p1.stepSimulation()

        self.run_time += 1

        self.scan_environment()

        # updating map;
        self.pos, quaternions = self.p1.getBasePositionAndOrientation(self.car_model)

        self.velocity, self.angular_velocity = p.getBaseVelocity(self.car_model)
        self.velocity = self.velocity[:2]
        self.angular_velocity = self.angular_velocity[:2]

        # checking if collided or finished
        if self.check_collision(self.car_model, self.bodies):
            self.crashed = True
        if dist(self.pos, self.end_point) < consts.min_dist_to_target:
            self.finished = True
        # # getting values for NN
        if self.pos[2] > 0.1:
            self.crashed = True

        self.pos = self.pos[:2]
        self.rotation = self.p1.getEulerFromQuaternion(quaternions)[2]

        self.last_speed = self.speed
        self.speed = norm(self.velocity)
        self.acceleration = self.speed - self.last_speed

        self.rotation_trig = [np.cos(self.rotation), np.sin(self.rotation)]

        # saving for later
        swivel_states = self.p1.getJointStates(self.car_model, self.steering)
        self.swivel = sum((state[0] for state in swivel_states)) / len(swivel_states)  # average among wheels

        score = self.calculate_reward()
        self.total_score += score

        self.prev_pos = self.pos

        self.calculate_next_goal()

        if self.run_time >= consts.max_time:
            print(
                f"time's up maze {self.maze_idx}"
                f" - distance is {dist(self.pos, self.end_point)}"
                f" - total score is {self.total_score}"
                f" - initial distance {dist(self.start_point[:2], self.end_point[:2])}"
                f" - time {self.run_time}")
            return self.get_observation(), score, True, {}

        if not (self.crashed or self.finished):
            return self.get_observation(), score, False, {}

        if self.crashed:
            print(
                f"crashed maze {self.maze_idx}"
                f" - distance is {dist(self.pos, self.end_point)}"
                f" - total score is {self.total_score}"
                f" - initial distance {dist(self.start_point[:2], self.end_point[:2])}"
                f" - time {self.run_time}")
        if self.finished:
            print(
                f"finished maze {self.maze_idx}"
                f" - total score is {self.total_score}"
                f" - initial distance {dist(self.start_point[:2], self.end_point[:2])}"
                f" - time {self.run_time}")
        return self.get_observation(), score, True, {}

    def get_observation(self) -> dict:
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

    def set_car_position(self, position: consts.vector) -> None:
        """
        sets the car in a position
        :param position: position to place the car at
        :return:
        """
        self.p1.resetBasePositionAndOrientation(self.car_model, position, [0, 0, 0, 1])
        self.p1.resetBaseVelocity(self.car_model, [0, 0, 0], [0, 0, 0])

    def create_car_model(self) -> Tuple[int, List[int], List[int]]:
        """
        create the car in pybullet
        :return: the car ID and important joint IDs for steering and setting speed
        """
        car = self.p1.loadURDF(
            os.path.join(pd.getDataPath(), "racecar/racecar.urdf")
        )
        inactive_wheels = [3, 5, 7]
        wheels = [2]

        for wheel in inactive_wheels:
            self.p1.setJointMotorControl2(
                car, wheel, self.p1.VELOCITY_CONTROL, targetVelocity=0, force=0
            )

        steering = [4, 6]

        self.p1.resetBasePositionAndOrientation(car, [0, 0, 0], [0, 0, 0, 1])

        return car, wheels, steering

    def get_new_maze(self) -> Tuple[list, consts.vector, consts.vector]:
        """
        gets a new maze for the run
        :return: maze (a set of polygonal lines), a start_point and end_point(3D vectors)
        """
        self.maze_idx = self.np_random.randint(0, len(mazes.empty_set))
        maze, start, end = mazes.empty_set[self.maze_idx]
        return maze, end, start


def save_model(model_to_save: consts.network_model, format_str: str, suffix: str = '') -> None:
    """
    saves the model with to the location returned by the given format string (formats the time of the run's end)
    :param model_to_save: The ddpg model that needs to be saved
    :param format_str: A format string for the saved file - will be passed to strftime
    :param suffix: sting to add at the end of the file name
    :return:
    """
    try:
        os.mkdir("results_alt")
    except:
        pass
    curr_time = datetime.now().strftime(format_str)
    filename = f'run-{curr_time}{suffix}'
    print(f"saving as {filename}")
    model_to_save.save(f"results_alt/{filename}")


def get_model(env: consts.stable_baselines_env, should_load: bool, filename: str, verbose: bool = True) -> consts.network_model:
    """
    Returns a model from a save file or creates a new one
    :param env: Gym environment for the model
    :param should_load: Do we want to load a model or create a new one
    :param filename: Wanted model filename - None if we want the last file in results
    :param verbose: do we want  to print the file used for loading (might be
    different from filename if the given file name doesn't exist)
    :return: The created model
    """
    if should_load:
        if filename is not None and os.path.exists(filename):
            # file exists and will be loaded
            if verbose:
                print(f'loading file: {filename}')
            model = SAC.load(filename, env, train_freq=1, gradient_steps=2,
                             verbose=1)
            return model

        # searching for latest file
        list_of_files = glob.glob("results_alt/*.zip")
        if len(list_of_files) != 0:  # there is an existing file
            latest_file = max(list_of_files, key=os.path.getctime)
            if verbose:
                print(f'loading file: {latest_file}')
            model = SAC.load(latest_file, env, train_freq=1, gradient_steps=2,
                             verbose=1)
            return model

    # creating new model
    if verbose:
        print("Creating a new model")
    model = SAC("MultiInputPolicy", env, train_freq=1, gradient_steps=2, verbose=1, buffer_size=10000)
    return model


def evaluate(model: consts.network_model, env: consts.stable_baselines_env) -> float:
    """
    Evaluating the model
    :param env: Gym environment for the model
    :param model: The model to evaluate
    :return: the mean in the evaluations
    """
    env.reset()
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=3, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    return mean_reward


def main():
    env = SubprocVecEnv([make_env(i) for i in range(consts.num_processes)])
    print("loading")
    model = get_model(env, consts.is_model_load, consts.loaded_model_path)
    print("training")
    while True:
        model.learn(total_timesteps=consts.checkpoint_steps)
        reward = evaluate(model, env)
        save_model(model, "%m_%d-%H_%M_%S", suffix=f'${str(int(reward))}')


if __name__ == "__main__":
    main()
