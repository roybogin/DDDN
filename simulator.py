import pybullet as p
import os
import pybullet_data
import map_create
import math
import scan_to_map
import numpy as np
import consts
from scan_to_map import Map, dist
import time as t
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


def addLists(lists):
    ret = [0, 0, 0]
    for l in lists:
        for i in range(len(l)):
            ret[i] += l[i]
    return ret


def multiply_list_by_scalar(list, scalar):
    return [scalar * elem for elem in list]


def start_simulation():
    if consts.is_visual:
        cid = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

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

    return col_id


def create_car_model(starting_point):
    car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"))
    inactive_wheels = [3, 5, 7]
    wheels = [2]

    for wheel in inactive_wheels:
        p.setJointMotorControl2(
            car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0
        )

    steering = [4, 6]

    p.resetBasePositionAndOrientation(car, starting_point, [0, 0, 0, 1])

    return car, wheels, steering


def ray_cast(car, offset, direction):
    pos, quat = p.getBasePositionAndOrientation(car)
    euler = p.getEulerFromQuaternion(quat)
    x = math.cos(euler[2])
    y = math.sin(euler[2])
    offset = [x * offset[0] - y * offset[1], x * offset[1] + y * offset[0], offset[2]]
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


def check_collision(car_model, obstacles, col_id, margin=0, max_distance=1.0):
    for ob in obstacles:
        closest_points = p.getClosestPoints(car_model, ob, distance=max_distance)
        closest_points = [a for a in closest_points if not (a[1] == a[2] == car_model)]
        if len(closest_points) != 0:
            dist = np.min([pt[8] for pt in closest_points])
            if dist < margin:
                return True
    return False


def norm(a1, a2):
    return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a1, a2))))


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


def add_disovered_list(discovered_matrix, start, end):
    x0 = int((start[0] + consts.size_map_quarter) / consts.block_size)
    y0 = int((start[1] + consts.size_map_quarter) / consts.block_size)
    x1 = int((end[0] + consts.size_map_quarter) / consts.block_size)
    y1 = int((end[1] + consts.size_map_quarter) / consts.block_size)

    plot_line(x0, y0, x1, y1, discovered_matrix)

    return (
        sum([sum(discovered_matrix[i]) for i in range(len(discovered_matrix))])
        / len(discovered_matrix) ** 2
    )


def draw_discovered_matrix(discovered_matrix):
    cmap = ListedColormap(["b", "g"])
    matrix = np.array(discovered_matrix, dtype=np.uint8)
    plt.imshow(matrix, cmap=cmap)


def print_reward_breakdown(
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
    draw_discovered_matrix(discovered)
    map.show()


def run_sim(car_brain, steps, maze, starting_point, end_point):
    targetVelocity = 0
    steeringAngle = 0
    swivel = 0
    distance_covered = 0
    min_distance_to_target = dist(starting_point[:2], end_point[:2])
    map_discovered = 0
    finished = False
    time = 0
    crushed = False
    max_hits_before_calculation = 10
    hits = []

    col_id = start_simulation()
    bodies = map_create.create_map(maze, end_point, epsilon=0.1)
    map = Map([consts.map_borders.copy()])
    car_model, wheels, steering = create_car_model(starting_point)
    last_pos = starting_point
    last_speed = 0

    discovered = [
        [0 for x in range(int((2 * consts.size_map_quarter + 1) // consts.block_size))]
        for y in range(int((2 * consts.size_map_quarter + 1) // consts.block_size))
    ]

    for i in range(steps):
        if consts.record:
            log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, consts.video_name)
        if consts.debug_sim:
            print("step", i)
            print("pos", pos)
            print("last_pos", last_pos)
            print("speed", speed)
            print("acceleration", acceleration)
            print("swivel:", steeringAngle)
            print("targetVelocity", targetVelocity)
            print()

        if crushed or finished:
            p.disconnect()
            if consts.print_reward_breakdown:
                print_reward_breakdown(
                    distance_covered,
                    map_discovered,
                    finished,
                    time,
                    crushed,
                    min_distance_to_target,
                    discovered,
                    map,
                )
            return (
                distance_covered,
                map_discovered,
                finished,
                time,
                crushed,
                min_distance_to_target,
            )

        # updating map
        did_hit, start, end = ray_cast(car_model, [0, 0, 0], [-consts.ray_length, 0, 0])
        if did_hit:
            hits.append((end[0], end[1]))
            if len(hits) == max_hits_before_calculation:
                map.add_points_to_map(hits)
                hits = []

        map_discovered = add_disovered_list(discovered, start, end)

        # checking if collided or finished
        if check_collision(car_model, bodies, col_id):
            crushed = True
        if scan_to_map.dist(last_pos, end_point) < consts.min_dist_to_target:
            finished = True
        # getting values for NN
        pos, quat = p.getBasePositionAndOrientation(car_model)
        if pos[2] > 0.1:
            crushed = True

        pos = pos[:2]
        rotation = p.getEulerFromQuaternion(quat)[2]
        speed = norm(pos, last_pos)
        acceleration = speed - last_speed

        changeTargetVelocity, changeSteeringAngle = car_brain.forward(
            [
                pos,
                end_point[:2],
                speed,
                swivel,
                rotation,
                acceleration,
                time,
                map.segment_representation_as_points(),
            ]
        )
        # updating target velocity and steering angle
        targetVelocity += changeTargetVelocity * consts.speed_scalar
        steeringAngle += changeSteeringAngle * consts.steer_scalar
        if abs(steeringAngle) > consts.max_steer:
            steeringAngle = consts.max_steer * steeringAngle / abs(steeringAngle)
        if abs(targetVelocity) > consts.max_velocity:
            targetVelocity = consts.max_velocity * targetVelocity / abs(targetVelocity)

        # saving for later
        swivel = steeringAngle
        last_pos = pos
        last_speed = speed

        # moving
        for wheel in wheels:
            p.setJointMotorControl2(
                car_model,
                wheel,
                p.VELOCITY_CONTROL,
                targetVelocity=targetVelocity,
                force=consts.max_force,
            )

        for steer in steering:
            p.setJointMotorControl2(
                car_model, steer, p.POSITION_CONTROL, targetPosition=steeringAngle
            )

        time += 1
        distance_covered += speed
        min_distance_to_target = min(min_distance_to_target, dist(pos, end_point[:2]))
        p.stepSimulation()
    if consts.record:
        p.stopStateLogging(log_id)
    p.disconnect()

    if consts.print_reward_breakdown:
        print_reward_breakdown(
            distance_covered,
            map_discovered,
            finished,
            time,
            crushed,
            min_distance_to_target,
            discovered,
            map,
        )
    map.show()
    return (
        distance_covered,
        map_discovered,
        finished,
        time,
        crushed,
        min_distance_to_target,
    )
