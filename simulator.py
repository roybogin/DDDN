import pybullet as p
import os
import pybullet_data
import map_create
import math
import scan_to_map
import numpy as np
import consts
from scan_to_map import Map
import time as t


def addLists(lists):
    ret = [0, 0, 0]
    for l in lists:
        for i in range(len(l)):
            ret[i] += l[i]
    return ret


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
    return p.rayTest(addLists([pos, offset]), addLists([pos, offset, direction]))[0][3]


def check_collision(car_model, obstacles, col_id, margin=0, max_distance=1.0):
    for ob in obstacles:
        closest_points = p.getClosestPoints(
            car_model, ob, distance=max_distance, physicsClientId=col_id
        )
        closest_points = [a for a in closest_points if not (a[1] == a[2] == car_model)]
        if len(closest_points) != 0:
            dist = np.min([pt[8] for pt in closest_points])
            if dist < margin:
                return True
    return False


def norm(a1, a2):
    return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a1, a2))))


def run_sim(car_brain, steps, maze, starting_point, end_point):
    targetVelocity = 0
    steeringAngle = 0
    swivel = 0
    distance_covered = 0
    map_discovered = 0
    finished = False
    time = 0
    crushed = False
    max_hits_before_calculation = 10
    hits = []
    last_speed = 0
    col_id = start_simulation()
    bodies = map_create.create_map(maze, epsilon=0.1)
    map = Map(consts.map_borders.copy())
    car_model, wheels, steering = create_car_model(starting_point)
    last_pos = starting_point
    maze = scan_to_map.Map([])

    for i in range(steps):
        if consts.record:
            log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, consts.video_name)
        if consts.debug_sim:
            pos, qat = p.getBasePositionAndOrientation(car_model)
            if pos[2] > 0.1:
                print("step", i)
                print("pos", pos)
                print("last_pos", last_pos)
                print("speed", speed)
                print("acceleration", acceleration)
                print("swivel:", steeringAngle)
                print("targetVelocity", targetVelocity)
                print()

        if crushed or finished:
            return distance_covered, map_discovered, finished, time, crushed

        # updating map
        hit = ray_cast(car_model, [0, 0, 0], [-consts.ray_length, 0, 0])
        if hit != (0, 0, 0):
            hits.append((hit[0], hit[1]))
            if len(hits) == max_hits_before_calculation:
                map.add_points_to_map(hits)

                hits = []

        map_discovered = map.map_length() - 279

        # checking if collided or finished
        if check_collision(car_model, bodies, col_id):
            crushed = True
        if scan_to_map.dist(last_pos, end_point) < consts.min_dist_to_target:
            finished = True
        # getting values for NN
        pos, quat = p.getBasePositionAndOrientation(car_model)
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
            targetVelocity = consts.max_steer * targetVelocity / abs(targetVelocity)

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
                force=10,
            )

        for steer in steering:
            p.setJointMotorControl2(
                car_model, steer, p.POSITION_CONTROL, targetPosition=steeringAngle
            )

        time += 1
        distance_covered += speed
        p.stepSimulation()
    if consts.record:
        p.stopStateLogging(log_id)
    p.disconnect()
    if consts.print_reward_breakdown:
        print("map_discoverd", map_discovered)
        print("distance_covered", distance_covered)
        print("time", time)
        map.show()

    return distance_covered, map_discovered, finished, time, crushed
