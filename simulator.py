import pybullet as p
import os
import pybullet_data
import map_create
import math
import scan_to_map
import numpy as np


def addLists(lists):
    ret = [0, 0, 0]
    for l in lists:
        for i in range(len(l)):
            ret[i] += l[i]
    return ret


def start_simulation():
    cid = p.connect(p.SHARED_MEMORY)

    col_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

    if cid < 0:
        p.connect(p.GUI)

    p.resetSimulation()
    p.setGravity(0, 0, -10)
    useRealTimeSim = 0

    p.setRealTimeSimulation(useRealTimeSim)
    # p.loadURDF("plane.urdf")
    p.loadSDF(os.path.join(pybullet_data.getDataPath(), "stadium.sdf"))

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

    return car, wheel, steering


def ray_cast(p, car, offset, direction):
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


def check_collision(car_model, obstacles, col_id, margin=0):

    ds = compute_min_distance(car_model, obstacles, col_id)
    return (ds < margin).any()


def compute_min_distance(car_model, obstacles, col_id, max_distance=1.0):
    distances = []
    for ob in obstacles:
        closest_points = p.getClosestPoint(
            car_model.body_uid,
            ob.body_uid,
            distance=max_distance,
            linkIndexA=car_model.link_uid,
            linkIndexB=ob.link_uid,
            physicsClientId=col_id,
        )
    if len(closest_points) == 0:
        distances.append(max_distance)
    else:
        distances.append(np.min([pt[8] for pt in closest_points]))
    return np.array(distances)


def run_sim(car_brain, steps, map, starting_point, end_point):

    swivel = 0
    distance_covered = 0
    map_discovered = 0
    finished = False
    time = 0
    crushed = False
    min_dist_to_target = 0.1
    max_hits_before_calculation = 10
    hits = []

    start_simulation()
    bodies = map_create.create_map()
    car_model, wheels, steering = create_car_model(starting_point)
    bodies.append(car_model)
    map = scan_to_map.map([])

    for i in range(steps):
        if crushed or finished:
            return distance_covered, map_discovered, finished, time, crushed

        hit = ray_cast(p, car_model)
        if hit != (0, 0, 0):
            hits.append((hit[0], hit[1]))
            if len(hits) == max_hits_before_calculation:
                map.add_points_to_map(hits)
                hits = []

        pos, quat = p.getBasePositionAndOrientation(car_model)
        pos = pos[0:1]
        rotation = p.getEulerFromQuaternion(quat)[2]

        # if this does not work, there is a function that returns velocity, then we can compute norm
        speed = p.getSpeed(car_model)

        # if this does not work, can use speed - last_speed... should be okay too
        acceleration = p.getAcceleration(car_model)
        targetVelocity, steeringAngle = car_brain.forward(
            pos, end_point, speed, swivel, rotation, acceleration, map
        )

        last_speed = speed

        if scan_to_map.dist(pos, end_point) < min_dist_to_target:
            finished = True

        if check_collision():
            pass

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
        swivel = steeringAngle

        p.stepSimulation()

    return distance_covered, map_discovered, finished, time, crushed
