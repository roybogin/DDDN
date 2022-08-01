import glob
import math
import os
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data as pd
from gym import spaces
from matplotlib.colors import ListedColormap
from pybullet_utils import bullet_client
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

import consts
import map_create
import scan_to_map
from scan_to_map import dist


def make_env(index, seed=0):
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


# car api to use with the DDPG algorithm
def get_new_maze():
    """
    returns a maze (a set of polygonal lines), a start_point and end_point(3D vectors)
    """
    start = [0, 0, 0]
    end = [0, 1, 0]
    maze = []
    return maze, end, start


def draw_discovered_matrix(discovered_matrix):
    cmap = ListedColormap(["b", "g"])
    matrix = np.array(discovered_matrix, dtype=np.uint8)
    plt.imshow(matrix, cmap=cmap)


def add_discovered_list(discovered_matrix, start, end):
    x0 = int((start[0] + consts.size_map_quarter) / consts.block_size)
    y0 = int((start[1] + consts.size_map_quarter) / consts.block_size)
    x1 = int((end[0] + consts.size_map_quarter) / consts.block_size)
    y1 = int((end[1] + consts.size_map_quarter) / consts.block_size)

    plot_line(x0, y0, x1, y1, discovered_matrix)

    return (
            sum([sum(discovered_matrix[i]) for i in
                 range(len(discovered_matrix))])
            / len(discovered_matrix) ** 2
    )


# car api to use with the DDPG algorithm
class CarEnv(gym.Env):
    def __init__(self, index, seed, size=10):
        super(CarEnv, self).__init__()
        self.speed = None
        self.velocity = None
        self.angular_velocity = None
        self.rotation = None
        self.pos = None
        self.start_point = None
        self.discovered = None
        self.map = None
        self.bodies = None
        self.hits = None
        self.last_speed = None
        self.last_pos = None
        self.time = None
        self.finished = None
        self.map_discovered = None
        self.crushed = None
        self.min_distance_to_target = None
        self.distance_covered = None
        self.swivel = None
        self.steeringAngle = None
        self.targetVelocity = None
        self.acceleration = None
        self.end_point = None
        self.maze = None
        self.borders = None
        self.p1 = None
        self.index = index
        self.seed = seed
        self.discovery_difference = 0
        self.wanted_observation = {
            "position": 2,
            "goal": 2,
            "velocity": 2,
            "angular_velocity": 2,
            "swivel": 1,
            "rotation_trigonometry": 2,
            "acceleration": 1,
            "map": int((2 * consts.size_map_quarter + 1) // consts.block_size) ** 2,
            "discovered": int((2 * consts.size_map_quarter + 1) // consts.block_size) ** 2}
        self.observation_len = sum(self.wanted_observation.values())

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
                # "time": spaces.Box(0, consts.max_time, shape=(1,), dtype=int)
                "map": spaces.Box(0, 1, shape=(int((2 * consts.size_map_quarter + 1) // consts.block_size),
                                               int((2 * consts.size_map_quarter + 1) // consts.block_size)),
                                  dtype=np.uint8),
                "discovered": spaces.Box(0, 1, shape=(int((2 * consts.size_map_quarter + 1) // consts.block_size),
                                                      int((2 * consts.size_map_quarter + 1) // consts.block_size)),
                                         dtype=np.uint8)
            }
        )
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        self.min_dist_to_target = consts.min_dist_to_target
        self.max_hits_before_calculation = consts.max_hits_before_calculation
        self.env_size = size
        self.car_model = None
        self.wheels = None
        self.steering = None
        self.obstacles = []
        self.start_env()
        self.reset()

    def start_env(self):
        self.p1 = bullet_client.BulletClient(p.DIRECT)
        self.p1.setAdditionalSearchPath(pd.getDataPath())
        self.p1.setGravity(0, 0, -10)

        self.p1.setRealTimeSimulation(consts.use_real_time)
        self.p1.loadSDF(os.path.join(pd.getDataPath(), "stadium.sdf"))
        self.p1.resetDebugVisualizerCamera(
            cameraDistance=consts.cameraDistance,
            cameraYaw=consts.cameraYaw,
            cameraPitch=consts.cameraPitch,
            cameraTargetPosition=consts.cameraTargetPosition,
        )
        self.p1.configureDebugVisualizer(self.p1.COV_ENABLE_GUI, 0)
        self.add_borders()
        self.car_model, self.wheels, self.steering = self.create_car_model()

    def add_borders(self):
        """
        adds the boarders to the maze
        """
        self.borders = map_create.create_poly_wall(consts.map_borders, epsilon=0.1)

    def remove_all_obstacles(self):
        for obstacle in self.obstacles:
            self.p1.removeBody(obstacle)
        self.obstacles = []

    def reset(self):
        """
        resets the environment
        options can be used to specify "how to reset" (like with an empty maze/one obstacle etc.)
        """
        self.remove_all_obstacles()

        self.maze, self.end_point, self.start_point = get_new_maze()

        self.targetVelocity = 0
        self.steeringAngle = 0
        self.swivel = 0
        self.speed = 0
        self.velocity = [0, 0]
        self.angular_velocity = [0, 0]
        self.acceleration = 0
        self.rotation = 0
        self.distance_covered = 0
        self.min_distance_to_target = dist(self.start_point[:2], self.end_point[:2])
        self.map_discovered = 0
        self.finished = False
        self.time = 0
        self.crushed = False
        self.hits = []
        self.last_pos = self.start_point
        self.pos = self.start_point
        self.last_speed = 0

        self.obstacles = map_create.create_map(self.maze, self.end_point, epsilon=0.1)
        self.bodies = self.borders + self.obstacles
        self.map = [[0 for _ in range(int((2 * consts.size_map_quarter + 1) // consts.block_size))] for _ in
                    range(int((2 * consts.size_map_quarter + 1) // consts.block_size))]
        for i in range(int((2 * consts.size_map_quarter + 1) // consts.block_size)):
            self.map[i][0] = 1
            self.map[0][i] = 1
            self.map[i][int((2 * consts.size_map_quarter + 1) // consts.block_size) - 1] = 1
            self.map[int((2 * consts.size_map_quarter + 1) // consts.block_size) - 1][i] = 1
        self.set_car_position(self.start_point)
        self.discovered = [
            [0 for _ in range(int((2 * consts.size_map_quarter + 1) // consts.block_size))]
            for _ in range(int((2 * consts.size_map_quarter + 1) // consts.block_size))
        ]
        self.discovery_difference = 0
        return self.get_observation()

    def ray_cast(self, car, offset, direction):
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

    def print_reward_breakdown(self):
        print("got to target", self.finished)
        print("min_distance_to_target", self.min_distance_to_target)
        print("crushed", self.crushed)
        print("map_discovered", self.map_discovered)
        print("distance_covered", self.distance_covered)
        print("time", self.time)
        draw_discovered_matrix(self.discovered)
        draw_discovered_matrix(self.map)

    def calculate_reward(self):
        reward = (
                self.time * consts.TIME_PENALTY + 
                self.crushed * consts.CRUSH_PENALTY + 
                self.finished * consts.FINISH_REWARD +
                self.map_discovered * consts.DISCOVER_REWARD
        )
        return reward

    def check_collision(self, car_model, obstacles, margin=0,
                        max_distance=1.0):
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

    def step(self, action):
        """
        the step function, gets an action (tuple of speed change and steer change)
        runs the simulation one step and returns the reward, the observation and if we are done.
        """
        if consts.print_runtime:
            print(self.time)

        # updating map;
        self.velocity, self.angular_velocity = p.getBaseVelocity(self.car_model)
        self.velocity = self.velocity[:2]
        self.angular_velocity = self.angular_velocity[:2]

        directions = [2 * np.pi * i / consts.ray_amount for i in range(consts.ray_amount)]
        new_map_discovered = self.discovered
        amount_discovered = self.map_discovered
        for direction in directions:

            did_hit, start, end = self.ray_cast(
                self.car_model, [0, 0, 0],
                [-consts.ray_length * np.cos(direction), -consts.ray_length * np.sin(direction), 0]
            )
            if did_hit:
                x1 = int((end[0] + consts.size_map_quarter) / consts.block_size)
                y1 = int((end[1] + consts.size_map_quarter) / consts.block_size)
                self.map[x1][y1] = 1
            self.map_discovered = add_discovered_list(new_map_discovered, start, end)

        self.discovery_difference = self.map_discovered - amount_discovered
        self.discovered = new_map_discovered

        # checking if collided or finished
        if self.check_collision(self.car_model, self.bodies):
            self.crushed = True
        if scan_to_map.dist(self.last_pos, self.end_point) < self.min_dist_to_target:
            self.finished = True
        # # getting values for NN
        self.pos, quaternions = self.p1.getBasePositionAndOrientation(self.car_model)
        if self.pos[2] > 0.1:
            self.crushed = True

        self.pos = self.pos[:2]
        self.rotation = self.p1.getEulerFromQuaternion(quaternions)[2]
        self.speed = norm(self.velocity)
        self.acceleration = self.speed - self.last_speed

        change_steering_angle = action[0] * consts.max_steer
        change_target_velocity = action[1] * consts.max_velocity

        # updating target velocity and steering angle
        self.targetVelocity += change_target_velocity * consts.speed_scalar
        self.steeringAngle += change_steering_angle * consts.steer_scalar
        if abs(self.steeringAngle) > consts.max_steer:
            self.steeringAngle = consts.max_steer * self.steeringAngle / abs(
                self.steeringAngle)
        if abs(self.targetVelocity) > consts.max_velocity:
            self.targetVelocity = consts.max_velocity * self.targetVelocity / abs(
                self.targetVelocity)

        # saving for later
        self.swivel = self.steeringAngle
        self.last_pos = self.pos
        self.last_speed = self.speed

        # moving
        for wheel in self.wheels:
            self.p1.setJointMotorControl2(
                self.car_model,
                wheel,
                self.p1.VELOCITY_CONTROL,
                targetVelocity=self.targetVelocity,
                force=consts.max_force,
            )

        for steer in self.steering:
            self.p1.setJointMotorControl2(
                self.car_model,
                steer,
                self.p1.POSITION_CONTROL,
                targetPosition=self.steeringAngle,
            )

        self.time += 1
        self.distance_covered += self.speed
        self.min_distance_to_target = min(
            self.min_distance_to_target, dist(self.pos, self.end_point[:2])
        )

        if self.crushed or self.finished or self.time >= consts.max_time:
            if consts.print_reward_breakdown:
                self.print_reward_breakdown()
            if self.finished:
                print("finished")
            elif self.crushed:
                print("crashed")
            else:
                print(f"time's up - minimal distance is {self.min_distance_to_target}")
            return self.get_observation(), self.calculate_reward(), True, {}

        self.p1.stepSimulation()

        return self.get_observation(), self.calculate_reward(), False, {}

    def get_observation(self):

        observation = {
            "position": np.array(self.pos[:2], dtype=np.float32),
            "goal": np.array(self.end_point[:2], dtype=np.float32),
            "velocity": np.array(self.velocity, dtype=np.float32),
            "angular_velocity": np.array(self.angular_velocity, dtype=np.float32),
            "swivel": np.array([self.swivel], dtype=np.float32),
            "rotation_trigonometry": np.array([np.sin(self.rotation), np.cos(self.rotation)], dtype=np.float32),
            "acceleration": np.array([self.acceleration], dtype=np.float32),
            # "time": np.array([self.time], dtype=int),
            "map": np.array(self.map, dtype=np.uint8),
            "discovered": np.array(self.discovered, dtype=np.uint8)

        }
        return observation

    def set_car_position(self, starting_point):
        self.p1.resetBasePositionAndOrientation(self.car_model, starting_point, [0, 0, 0, 1])
        self.p1.resetBaseVelocity(self.car_model, [0, 0, 0], [0, 0, 0])

    def create_car_model(self):
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


def add_lists(lists):
    ret = [0, 0, 0]
    for lst in lists:
        for i in range(len(lst)):
            ret[i] += lst[i]
    return ret


def plot_line_low(x0, y0, x1, y1, discovered_matrix):
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    d = (2 * dy) - dx
    y = y0

    for x in range(x0, x1 + 1):
        if x < len(discovered_matrix) and y < len(discovered_matrix):
            discovered_matrix[x][y] = 1
        else:
            pass
            # print("illegal", x, y)
        if d > 0:
            y = y + yi
            d = d + (2 * (dy - dx))
        else:
            d = d + 2 * dy


def plot_line_high(x0, y0, x1, y1, discovered_matrix):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx

    d = (2 * dx) - dy
    x = x0

    for y in range(y0, y1 + 1):
        if x < len(discovered_matrix) and y < len(discovered_matrix):
            discovered_matrix[x][y] = 1
        else:
            # print("illegal", x, y)
            pass
        if d > 0:
            x = x + xi
            d = d + (2 * (dx - dy))
        else:
            d = d + 2 * dx


def plot_line(x0, y0, x1, y1, discovered_matrix):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            plot_line_low(x1, y1, x0, y0, discovered_matrix)
        else:
            plot_line_low(x0, y0, x1, y1, discovered_matrix)
    else:
        if y0 > y1:
            plot_line_high(x1, y1, x0, y0, discovered_matrix)
        else:
            plot_line_high(x0, y0, x1, y1, discovered_matrix)


def norm(a):
    return math.sqrt(sum((x ** 2 for x in a)))


def save_model(model_to_save, format_str, suffix=''):
    """
    saves the model with to the location returned by the given format string (formats the time of the run's end)
    :param model_to_save: The ddpg model that needs to be saved
    :param format_str: A format string for the saved file - will be passed to strftime
    :param suffix: sting to add at the end of the file name
    :return:
    """
    try:
        os.mkdir("results")
    except:
        pass
    curr_time = datetime.now().strftime(format_str)
    model_to_save.save(f"results/run-{curr_time}{suffix}")


def get_model(env, should_load, filename, verbose=True):
    """
    Returns a DDPG model from a save file or creates a new one
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
        list_of_files = glob.glob("results/*.zip")
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


def evaluate(model, env):
    """
    Evaluating the model
    :param env: Gym environment for the model
    :param model: The model to evaluate
    :return: the mean reward in the evaluations
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
    print("first eval")
    evaluate(model, env)
    print("training")
    total_runtime = 0
    while consts.train_steps < 0 or total_runtime < consts.train_steps:
        model.learn(total_timesteps=consts.checkpoint_steps)
        reward = evaluate(model, env)
        save_model(model, "%m_%d-%H_%M_%S", suffix=f'${str(int(reward))}')
        total_runtime += consts.checkpoint_steps
    reward = evaluate(model, env)
    save_model(model, "%m_%d-%H_%M_%S", suffix=f'${str(int(reward))}')


if __name__ == "__main__":
    main()
