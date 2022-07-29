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
import gym
import spinup


from matplotlib.colors import ListedColormap
from gym.utils.env_checker import check_env

from gym.envs.registration import register
from gym import spaces


# car api to use with the DDPG algorithem
class CarEnv(gym.Env):
    def __init__(self, size=10):

        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(-size, size, shape=(2,), dtype=float),
                "goal": spaces.Box(-size, size, shape=(2,), dtype=float),
                "speed": spaces.Box(0, consts.max_velocity, shape=(1,), dtype=float),
                "swivel": spaces.Box(
                    -consts.max_steer, consts.max_steer, shape=(1,), dtype=float
                ),
                "swivel": spaces.Box(
                    -consts.max_steer, consts.max_steer, shape=(1,), dtype=float
                ),
                "rotation": spaces.Box(-360, 360, shape=(2,), dtype=float),
                "acceleration": spaces.Box(-1000, 1000, shape=(1,), dtype=float),
                "time": spaces.Box(0, consts.max_time, shape=(1,), dtype=int),
                "map": spaces.Box(
                    0, consts.max_time, shape=(1,), dtype=int
                ),  # very much wrong
            }
        )
        self.action_space = spaces.Dict(
            {
                "steerChange": spaces.Box(-1, 1, shape=(1,), dtype=float),
                "velocityChange": spaces.Box(-1, 1, shape=(1,), dtype=float),
            }
        )

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
        end = [0, 1, 0]
        maze = []
        return (maze, end, start)

    def add_borders(self):
        """
        adds the boarders to the maze
        """
        self.borders = map_create.create_poly_wall(consts.map_borders, epsilon=0.1)

    def remove_all_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)
        self.obstacles = []

    def reset(self, seed=None, options=dict()):
        """
        resets the environment
        options can be used to specify "how to reset" (like with an empty maze/one obstacle etc)
        """
        super().reset(seed=seed)

        self.done = False
        self.remove_all_obstacles()

        self.maze, self.end_point, self.start_point = self.get_new_maze()

        self.targetVelocity = 0
        self.steeringAngle = 0
        self.swivel = 0
        self.distance_covered = 0
        self.min_distance_to_target = dist(self.start_point[:2], self.end_point[:2])
        self.map_discovered = 0
        self.finished = False
        self.time = 0
        self.crushed = False
        self.hits = []
        self.last_pos = self.start_point
        self.last_speed = 0

        self.obstacles = map_create.create_map(self.maze, self.end_point, epsilon=0.1)
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
            for y in range(int((2 * consts.size_map_quarter + 1) // consts.block_size))
        ]

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

    def print_reward_breakdown(
        self,
        distance_covered,
        map_discovered,
        finished,
        time,
        crushed,
        min_distance_to_target,
        discovered,
        map,
    ):
        print("got to target", finished)
        print("min_distance_to_target", min_distance_to_target)
        print("crushed", crushed)
        print("map_discoverd", map_discovered)
        print("distance_covered", distance_covered)
        print("time", time)
        self.draw_discovered_matrix(discovered)
        map.show()

    def draw_discovered_matrix(self, discovered_matrix):
        cmap = ListedColormap(["b", "g"])
        matrix = np.array(discovered_matrix, dtype=np.uint8)
        plt.imshow(matrix, cmap=cmap)

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
        reward = (self.speed * consts.DISTANCE_REWARD) + (
            self.discovery_difference * consts.EXPLORATION_REWARD
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

        if crushed or finished:
            if consts.print_reward_breakdown:
                self.print_reward_breakdown()
            if finished:
                return self.get_observation(), consts.END_REWARD, True
            if crushed:
                return self.get_observation(), consts.CRUSH_PENALTY, True

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
        if self.check_collision(self.car_model, self.bodies, self.col_id):
            crushed = True
        if scan_to_map.dist(self.last_pos, self.end_point) < self.min_dist_to_target:
            finished = True
        # # getting values for NN
        self.pos, quat = p.getBasePositionAndOrientation(self.car_model)
        if self.pos[2] > 0.1:
            crushed = True

        self.pos = self.pos[:2]
        self.rotation = p.getEulerFromQuaternion(quat)[2]
        self.speed = norm(self.pos, self.last_pos)
        self.acceleration = self.speed - self.last_speed

        changeSteeringAngle = action["steerChange"]
        changeTargetVelocity = action["VelocityChange"]

        # updating target velocity and steering angle
        targetVelocity += changeTargetVelocity * consts.speed_scalar
        steeringAngle += changeSteeringAngle * consts.steer_scalar
        if abs(steeringAngle) > consts.max_steer:
            steeringAngle = consts.max_steer * steeringAngle / abs(steeringAngle)
        if abs(targetVelocity) > consts.max_velocity:
            targetVelocity = consts.max_velocity * targetVelocity / abs(targetVelocity)

        # saving for later
        self.swivel = steeringAngle
        self.last_pos = self.pos
        self.last_speed = self.speed

        # moving
        for wheel in self.wheels:
            p.setJointMotorControl2(
                self.car_model,
                wheel,
                p.VELOCITY_CONTROL,
                targetVelocity=targetVelocity,
                force=consts.max_force,
            )

        for steer in self.steering:
            p.setJointMotorControl2(
                self.car_model, steer, p.POSITION_CONTROL, targetPosition=steeringAngle
            )

        self.time += 1
        self.distance_covered += self.speed
        min_distance_to_target = min(
            min_distance_to_target, dist(self.pos, self.end_point[:2])
        )
        p.stepSimulation()

        return self.get_observation(), self.calculate_reward(), self.done

    def get_observation(self):
        return {
            "position": self.pos,
            "goal": self.end_point[:2],
            "speed": self.speed,
            "swivel": self.swivel,
            "swivel": self.rotation,
            "rotation": self.rotation,
            "acceleration": self.acceleration,
            "time": self.time,
            "map": self.map.segment_representation_as_points(),
        }

    def set_car_position(self, starting_point):
        p.resetBasePositionAndOrientation(self.car_model, starting_point, [0, 0, 0, 1])

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


def norm(a1, a2):
    return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a1, a2))))


if __name__ == "__main__":
    env = CarEnv()
    check_env(env)
    var = spinup.ddpg_pytorch(CarEnv)
    type(var)
