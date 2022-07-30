import pybullet as p
import os
import pybullet_data

import consts
import map_create
from scan_to_map import Map, dist
import matplotlib.pyplot as plt
import numpy as np
import math
import scan_to_map
import glob

import gym


from matplotlib.colors import ListedColormap

from gym.envs.registration import register
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime


# car api to use with the DDPG algorithem
class CarEnv(gym.Env):
    def __init__(self, size=10):
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
        self.map_discovered = 0
        self.crushed = None
        self.min_distance_to_target = None
        self.distance_covered = None
        self.swivel = None
        self.steeringAngle = None
        self.targetVelocity = None
        self.acceleration = None
        self.end_point = None
        self.maze = None
        self.done = None
        self.borders = None
        self.col_id = None
        self.total_runtime = 0
        self.discovery_difference = 0
        self.wanted_observation = {
            "position": 2,
            "goal": 2,
            "velocity": 2,
            "angular_velocity": 2,
            "swivel": 1,
            "rotation": 1,
            "acceleration": 1,
            "map":  int((2 * consts.size_map_quarter + 1) // consts.block_size) ** 2,
            "discovered":  int((2 * consts.size_map_quarter + 1) // consts.block_size) ** 2}
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
                "rotation": spaces.Box(-360, 360, shape=(1,), dtype=np.float32),
                "acceleration": spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32),
                # "time": spaces.Box(0, consts.max_time, shape=(1,), dtype=int)
                "map":        spaces.Box(0, 1, shape=( int((2 * consts.size_map_quarter + 1) // consts.block_size),  int((2 * consts.size_map_quarter + 1) // consts.block_size)), dtype=np.uint8),
                "discovered": spaces.Box(0, 1, shape=( int((2 * consts.size_map_quarter + 1) // consts.block_size),  int((2 * consts.size_map_quarter + 1) // consts.block_size)), dtype=np.uint8)
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
        col_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=col_id)

        p.resetSimulation()
        p.setGravity(0, 0, -10)

        p.setRealTimeSimulation(consts.use_real_time)
        p.loadSDF(os.path.join(pybullet_data.getDataPath(), "stadium.sdf"))
        p.resetDebugVisualizerCamera(
            cameraDistance=consts.cameraDistance,
            cameraYaw=consts.cameraYaw,
            cameraPitch=consts.cameraPitch,
            cameraTargetPosition=consts.cameraTargetPosition,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.add_borders()
        self.car_model, self.wheels, self.steering = self.create_car_model()
        self.col_id = col_id

    def get_new_maze(self):
        """
        returns a maze (a set of polygonal lines), a start_point and end_point(3D vectors)
        """
        start = [0, 0, 0]
        end = [2, 1, 0]
        maze = []
        return maze, end, start

    def add_borders(self):
        """
        adds the boarders to the maze
        """
        self.borders = map_create.create_poly_wall(consts.map_borders, epsilon=0.1)

    def remove_all_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)
        self.obstacles = []

    def reset(self):
        """
        resets the environment
        options can be used to specify "how to reset" (like with an empty maze/one obstacle etc)
        """
        self.done = False
        self.remove_all_obstacles()

        self.maze, self.end_point, self.start_point = self.get_new_maze()

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
        self.map = [[0 for _ in range(int((2 * consts.size_map_quarter + 1) // consts.block_size))] for _ in range(int((2 * consts.size_map_quarter + 1) // consts.block_size))]
        for i in range(int((2 * consts.size_map_quarter + 1) // consts.block_size)):
            self.map[i][0] = 1
            self.map[0][i] = 1
            self.map[i][int((2 * consts.size_map_quarter + 1) // consts.block_size) - 1] = 1
            self.map[int((2 * consts.size_map_quarter + 1) // consts.block_size) - 1][i] = 1 
        self.set_car_position(self.start_point)
        self.discovered = [[0 for _ in range(int((2 * consts.size_map_quarter + 1) // consts.block_size))] for _ in
                           range(int((2 * consts.size_map_quarter + 1) // consts.block_size))]
        self.discovery_difference = 0
        return self.get_observation()

    def ray_cast(self, car, offset, direction):
        pos, quat = p.getBasePositionAndOrientation(car)
        euler = p.getEulerFromQuaternion(quat)
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
        start = addLists([pos, offset])
        end = addLists([pos, offset, direction])
        ray_test = p.rayTest(start, end)
        if ray_test[0][3] == (0, 0, 0):
            return False, start[:2], end[:2]
        else:
            return True, start[:2], ray_test[0][3]

    def print_reward_breakdown(self):
        print("got to target", self.finished)
        print("min_distance_to_target", self.min_distance_to_target)
        print("crushed", self.crushed)
        print("map_discoverd", self.map_discovered)
        print("distance_covered", self.distance_covered)
        print("time", self.time)
        self.draw_discovered_matrix(self.discovered)
        self.draw_discovered_matrix(self.map)

    def reset_runtime(self):
        self.total_runtime = 0

    def draw_discovered_matrix(self, discovered_matrix):
        cmap = ListedColormap(["b", "g"])
        matrix = np.array(discovered_matrix, dtype=np.uint8)
        plt.imshow(matrix, cmap=cmap)

    def add_disovered_list(self, discovered_matrix, start, end):
        x0 = int((start[0] + consts.size_map_quarter) / consts.block_size)
        y0 = int((start[1] + consts.size_map_quarter) / consts.block_size)
        x1 = int((end[0] + consts.size_map_quarter) / consts.block_size)
        y1 = int((end[1] + consts.size_map_quarter) / consts.block_size)

        plot_line(x0, y0, x1, y1, discovered_matrix)

        return (
            sum([sum(discovered_matrix[i]) for i in range(len(discovered_matrix))])
            / len(discovered_matrix) ** 2
        )

    def calculate_reward(self):
        reward = (
            (self.speed * consts.DISTANCE_REWARD)
            + (self.discovery_difference * consts.EXPLORATION_REWARD)
            + (self.min_distance_to_target * consts.MIN_DIST_PENALTY)
        )
        return reward

    def check_collision(self, car_model, obstacles, col_id, margin=0, max_distance=1.0):
        for ob in obstacles:
            closest_points = p.getClosestPoints(car_model, ob, distance=max_distance)
            closest_points = [
                a for a in closest_points if not (a[1] == a[2] == car_model)
            ]
            if len(closest_points) != 0:
                dist = np.min([pt[8] for pt in closest_points])
                if dist < margin:
                    return True
        return False

    def step(self, action):
        """
        the step function, gets an action (tuple of speedchange and steerchange)
        runs the simulation one step and returns the reward, the observation and if we are done.
        """
        if consts.print_runtime:
            print(self.time, self.total_runtime)
        if self.crushed or self.finished or self.time == consts.max_time:
            if consts.print_reward_breakdown:
                self.print_reward_breakdown()
            if self.finished:
                print("finished")
                return self.get_observation(), consts.END_REWARD, True, {}
            if self.crushed:
                print("crashed")
                return self.get_observation(), consts.CRUSH_PENALTY, True, {}
            return self.get_observation(), 0, True, {}

        # updating map;
        self.velocity, self.angular_velocity = p.getBaseVelocity(self.car_model)
        self.velocity = self.velocity[:2]
        self.angular_velocity = self.angular_velocity[:2]

        directions = [2*np.pi * i / consts.ray_amount for i in range(consts.ray_amount)]
        new_map_discovered = self.discovered
        amount_discovered = self.map_discovered
        for direction in directions:

            did_hit, start, end = self.ray_cast(
                self.car_model, [0, 0, 0], [-consts.ray_length*np.cos(direction), -consts.ray_length*np.sin(direction), 0]
            )
            if did_hit:
                x1 = int((end[0] + consts.size_map_quarter) / consts.block_size)
                y1 = int((end[1] + consts.size_map_quarter) / consts.block_size)
                self.map[x1][y1] = 1
            self.map_discovered = self.add_disovered_list(new_map_discovered, start, end)

        self.discovery_difference = self.map_discovered - amount_discovered

        # checking if collided or finished
        if self.check_collision(self.car_model, self.bodies, self.col_id):
            self.crushed = True
        if scan_to_map.dist(self.last_pos, self.end_point) < self.min_dist_to_target:
            self.finished = True
        # # getting values for NN
        self.pos, quat = p.getBasePositionAndOrientation(self.car_model)
        if self.pos[2] > 0.1:
            self.crushed = True

        self.pos = self.pos[:2]
        self.rotation = p.getEulerFromQuaternion(quat)[2]
        self.speed = norm(self.velocity)
        self.acceleration = self.speed - self.last_speed

        changeSteeringAngle = action[0] * consts.max_steer
        changeTargetVelocity = action[1] * consts.max_velocity

        # updating target velocity and steering angle
        self.targetVelocity += changeTargetVelocity * consts.speed_scalar
        self.steeringAngle += changeSteeringAngle * consts.steer_scalar
        if abs(self.steeringAngle) > consts.max_steer:
            self.steeringAngle = (
                consts.max_steer * self.steeringAngle / abs(self.steeringAngle)
            )
        if abs(self.targetVelocity) > consts.max_velocity:
            self.targetVelocity = (
                consts.max_velocity * self.targetVelocity / abs(self.targetVelocity)
            )

        # saving for later
        self.swivel = self.steeringAngle
        self.last_pos = self.pos
        self.last_speed = self.speed

        # moving
        for wheel in self.wheels:
            p.setJointMotorControl2(
                self.car_model,
                wheel,
                p.VELOCITY_CONTROL,
                targetVelocity=self.targetVelocity,
                force=consts.max_force,
            )

        for steer in self.steering:
            p.setJointMotorControl2(
                self.car_model,
                steer,
                p.POSITION_CONTROL,
                targetPosition=self.steeringAngle,
            )

        self.time += 1
        self.total_runtime += 1
        self.distance_covered += self.speed
        self.min_distance_to_target = min(
            self.min_distance_to_target, dist(self.pos, self.end_point[:2])
        )
        p.stepSimulation()

        return self.get_observation(), self.calculate_reward(), self.done, {}

    def get_observation(self):

        observation = {
            "position": np.array(self.pos[:2], dtype=np.float32),
            "goal": np.array(self.end_point[:2], dtype=np.float32),
            "velocity": np.array(self.velocity, dtype=np.float32),
            "angular_velocity": np.array(self.angular_velocity, dtype=np.float32),
            "swivel": np.array([self.swivel], dtype=np.float32),
            "rotation": np.array([self.rotation], dtype=np.float32),
            "acceleration": np.array([self.acceleration], dtype=np.float32),
            # "time": np.array([self.time], dtype=int),
            "map": np.array(self.map, dtype=np.uint8),
            "discovered": np.array(self.discovered, dtype=np.uint8)

        }
        return observation

    def set_car_position(self, starting_point):
        p.resetBasePositionAndOrientation(self.car_model, starting_point, [0, 0, 0, 1])
        p.resetBaseVelocity(self.car_model, [0, 0, 0], [0, 0, 0])


    def create_car_model(self):
        car = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf")
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



def addLists(lists):
    ret = [0, 0, 0]
    for l in lists:
        for i in range(len(l)):
            ret[i] += l[i]
    return ret


def plot_line_low(x0, y0, x1, y1, discovered_matrix):
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    D = (2 * dy) - dx
    y = y0

    for x in range(x0, x1 + 1):
        if x < len(discovered_matrix) and y < len(discovered_matrix):
            discovered_matrix[x][y] = 1
        else:
            pass
            # print("illegal", x, y)
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy


def plot_line_high(x0, y0, x1, y1, discovered_matrix):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx

    D = (2 * dx) - dy
    x = x0

    for y in range(y0, y1 + 1):
        if x < len(discovered_matrix) and y < len(discovered_matrix):
            discovered_matrix[x][y] = 1
        else:
            # print("illegal", x, y)
            pass
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx


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
    curr_time = datetime.now().strftime(format_str)
    model_to_save.save(f"results/{curr_time}{suffix}")


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
                print(f"loading file: {filename}")
            model = DDPG.load(filename, env)
            return model

        # searching for latest file
        list_of_files = glob.glob("results/*.zip")
        if len(list_of_files) != 0:  # there is an existing file
            latest_file = max(list_of_files, key=os.path.getctime)
            if verbose:
                print(f"loading file: {latest_file}")
            model = DDPG.load(latest_file, env)
            return model

    # creating new model
    if verbose:
        print("Creating a new model")
    model = DDPG("MultiInputPolicy", env, buffer_size=10000)
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
    env = CarEnv()
    print("loading")
    model = get_model(env, consts.is_model_load, consts.loaded_model_path)
    print("first eval")
    evaluate(model, env)
    print("training")
    env.reset_runtime()
    while consts.train_steps < 0 or env.total_runtime < consts.train_steps:
        model.learn(total_timesteps=consts.checkpoint_steps)
        reward = evaluate(model, env)
        save_model(model, "%m_%d-%H_%M_%S", suffix=f'${str(int(reward))}')
    reward = evaluate(model, env)
    save_model(model, "%m_%d-%H_%M_%S", suffix=f'${str(int(reward))}')


if __name__ == "__main__":
    main()
