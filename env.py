import pybullet as p
import os
import pybullet_data as pd
from pybullet_utils import bullet_client

import consts
import map_create
from scan_to_map import Map, dist
import matplotlib.pyplot as plt
import numpy as np
import math
import scan_to_map
import gym

from matplotlib.colors import ListedColormap

from gym.envs.registration import register
from gym import spaces

def make_env(index, seed=0):
    """
    Utility function for multiprocessed env.
    :param seed: (int) the inital seed for RNG
    :param index: (int) index of the subprocess
    """
    def _init():
        env = CarEnv(index, seed)
        return env
    return _init


# car api to use with the DDPG algorithem
class CarEnv(gym.Env):
    def __init__(self, index, seed, size=10):
        super(CarEnv, self).__init__()
        self.speed = None
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
        self.done = None
        self.borders = None
        self.p1 = None
        self.index = index
        self.seed = seed
        self.wanted_observation = {'position': 2,
                                   'goal': 2,
                                   'speed': 1,
                                   'swivel': 1,
                                   'rotation': 1,
                                   'acceleration': 1,
                                   'time': 1}  # TODO: add map
        self.observation_len = sum(self.wanted_observation.values())

        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(-size, size, shape=(2,), dtype=np.float32),
                "goal": spaces.Box(-size, size, shape=(2,), dtype=np.float32),
                "speed": spaces.Box(0, consts.max_velocity, shape=(1,),
                                    dtype=np.float32),
                "swivel": spaces.Box(
                    -consts.max_steer, consts.max_steer, shape=(1,), dtype=np.float32
                ),
                "rotation": spaces.Box(-360, 360, shape=(1,), dtype=np.float32),
                "acceleration": spaces.Box(-1000, 1000, shape=(1,),
                                           dtype=np.float32),
                "time": spaces.Box(0, consts.max_time, shape=(1,), dtype=int)
                # TODO: add map
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

    def get_new_maze(self):
        """
        returns a maze (a set of polygonal lines), a start_point and end_point(3D vectors)
        """
        start = [0, 0, 0]
        end = [0, 1, 0]
        maze = []
        return maze, end, start

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
        options can be used to specify "how to reset" (like with an empty maze/one obstacle etc)
        """
        self.done = False
        self.remove_all_obstacles()

        self.maze, self.end_point, self.start_point = self.get_new_maze()

        self.targetVelocity = 0
        self.steeringAngle = 0
        self.swivel = 0
        self.speed = 0
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

        self.obstacles = map_create.create_map(self.maze, self.end_point,
                                               epsilon=0.1)
        self.bodies = self.borders + self.obstacles
        self.map = Map([consts.map_borders.copy()])
        self.set_car_position(self.start_point)
        self.discovered = [
            [
                0
                for x in range(
                int((2 * consts.size_map_quarter + 1) // consts.block_size)
            )
            ]
            for y in
            range(int((2 * consts.size_map_quarter + 1) // consts.block_size))
        ]
        return self.get_observation()

    def ray_cast(self, car, offset, direction):
        pos, quat = self.p1.getBasePositionAndOrientation(car)
        euler = self.p1.getEulerFromQuaternion(quat)
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
        ray_test = self.p1.rayTest(start, end)
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
        self.maself.p1.show()
        self.maself.p1.show()

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
                sum([sum(discovered_matrix[i]) for i in
                     range(len(discovered_matrix))])
                / len(discovered_matrix) ** 2
        )

    def calculate_reward(self):
        reward = (self.speed * consts.DISTANCE_REWARD) + (
                self.discovery_difference * consts.EXPLORATION_REWARD
        )
        return reward

    def check_collision(self, car_model, obstacles, margin=0,
                        max_distance=1.0):
        for ob in obstacles:
            closest_points = self.p1.getClosestPoints(car_model, ob, distance=max_distance)
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
        print(self.time)
        if self.crushed or self.finished or self.time == consts.max_time:
            if consts.print_reward_breakdown:
                self.print_reward_breakdown()
            if self.finished:
                return self.get_observation(), consts.END_REWARD, True, {}
            if self.crushed:
                return self.get_observation(), consts.CRUSH_PENALTY, True, {}
            return self.get_observation(), 0, True, {}

        # updating map
        did_hit, start, end = self.ray_cast(
            self.car_model, [0, 0, 0], [-consts.ray_length, 0, 0]
        )
        if did_hit:
            self.hits.append((end[0], end[1]))
            if len(self.hits) == self.max_hits_before_calculation:
                self.map.add_points_to_map(self.hits)
                self.hits = []

        new_map_discovered = self.add_disovered_list(self.discovered, start, end)
        self.discovery_difference = new_map_discovered - self.map_discovered
        self.map_discovered = new_map_discovered

        # checking if collided or finished
        if self.check_collision(self.car_model, self.bodies):
            self.crushed = True
        if scan_to_map.dist(self.last_pos, self.end_point) < self.min_dist_to_target:
            self.finished = True
        # # getting values for NN
        self.pos, quat = self.p1.getBasePositionAndOrientation(self.car_model)
        if self.pos[2] > 0.1:
            self.crushed = True

        self.pos = self.pos[:2]
        self.rotation = self.p1.getEulerFromQuaternion(quat)[2]
        self.speed = norm(self.pos, self.last_pos)
        self.acceleration = self.speed - self.last_speed

        changeSteeringAngle = action[0] * consts.max_steer
        changeTargetVelocity = action[1] * consts.max_velocity

        # updating target velocity and steering angle
        self.targetVelocity += changeTargetVelocity * consts.speed_scalar
        self.steeringAngle += changeSteeringAngle * consts.steer_scalar
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
                self.car_model, steer, self.p1.POSITION_CONTROL,
                targetPosition=self.steeringAngle
            )

        self.time += 1
        self.distance_covered += self.speed
        self.min_distance_to_target = min(
            self.min_distance_to_target, dist(self.pos, self.end_point[:2])
        )
        self.p1.stepSimulation()

        return self.get_observation(), self.calculate_reward(), self.done, {}

    def get_observation(self):

        observation = {"position": np.array(self.pos[:2], dtype=np.float32),
                       "goal": np.array(self.end_point[:2], dtype=np.float32),
                       "speed": np.array([self.speed], dtype=np.float32),
                       "swivel": np.array([self.swivel], dtype=np.float32),
                       "rotation": np.array([self.rotation], dtype=np.float32),
                       "acceleration": np.array([self.acceleration], dtype=np.float32),
                       "time": np.array([self.time], dtype=int)
                       }
        return observation

    def set_car_position(self, starting_point):
        self.p1.resetBasePositionAndOrientation(self.car_model, starting_point,
                                          [0, 0, 0, 1])

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


def norm(a1, a2):
    return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a1, a2))))


if __name__ == "__main__":
    from stable_baselines3 import DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    env = SubprocVecEnv([make_env(i) for i in range(consts.num_processes)])
    model = DDPG("MultiInputPolicy", env, train_freq=1, gradient_steps=2, verbose=1)
    model.learn(total_timesteps=consts.max_time)
