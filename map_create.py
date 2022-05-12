import math
import pybullet as p
import pybullet_data
import os


def create_wall(pos, orientation, length, width):

    boxHalfLength = length / 2
    boxHalfWidth = width / 2
    boxHalfHeight = 0.5
    body = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight]
    )
    mass = 10000
    visualShapeId = -1
    # block creation
    block = p.createMultiBody(
        10000,
        body,
        -1,
        pos,
        orientation,
    )
    return block


def distance(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def create_poly_wall(poly, epsilon):
    length = distance(poly[0], poly[1]) + 2 * epsilon
    width = 2 * epsilon
    prev_angle = math.atan2(poly[1][1] - poly[0][1], poly[1][0] - poly[0][0])
    euler = [0, 0, prev_angle]
    orientation = p.getQuaternionFromEuler(euler)
    pos = [(poly[0][0] + poly[1][0]) / 2, (poly[0][1] + poly[1][1]) / 2, 0.5]
    create_wall(pos, orientation, length, width)
    walls = []
    for i in range(1, len(poly) - 1):
        length = distance(poly[i], poly[i + 1]) + epsilon
        angle = math.atan2(poly[i + 1][1] - poly[i][1], poly[i + 1][0] - poly[i][0])
        diff = angle - prev_angle
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff <= -math.pi:
            diff += 2 * math.pi
        euler = [0, 0, angle]
        orientation = p.getQuaternionFromEuler(euler)

        length_from = 0
        if diff == math.pi:
            continue
        if diff == 0 or diff == math.pi / 2 or diff == -math.pi / 2:
            length_from = epsilon

        elif -math.pi / 2 < diff < math.pi / 2:
            if diff > 0:
                length_from = math.sqrt(2) * epsilon * math.cos(math.pi / 4 - diff)
            else:
                length_from = math.sqrt(2) * epsilon * math.cos(math.pi / 4 + diff)
        else:
            if diff > 0:
                length_from += epsilon / math.tan(math.pi - diff)
                length_from += epsilon / math.sin(math.pi - diff)
            else:
                length_from += epsilon / math.tan(math.pi + diff)
                length_from += epsilon / math.sin(math.pi + diff)

        length -= length_from
        pos = [
            poly[i][0] + math.cos(angle) * (length / 2 + length_from),
            poly[i][1] + math.sin(angle) * (length / 2 + length_from),
            0.5,
        ]
        walls.append(create_wall(pos, orientation, length, width))
        prev_angle = angle
    return walls


def create_map(in_map, epsilon):
    walls = []
    for poly in in_map:
        walls += create_poly_wall(poly, epsilon)
    return walls


def main():

    cid = p.connect(p.SHARED_MEMORY)
    if cid < 0:
        p.connect(p.GUI)

    p.resetSimulation()
    p.setGravity(0, 0, -10)

    useRealTimeSim = 0

    # for video recording (works best on Mac and Linux, not well on Windows)
    # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
    p.setRealTimeSimulation(useRealTimeSim)  # either this
    # p.loadURDF("plane.urdf")
    p.loadSDF(os.path.join(pybullet_data.getDataPath(), "stadium.sdf"))

    car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"))
    inactive_wheels = [3, 5, 7]
    wheels = [2]

    for wheel in inactive_wheels:
        p.setJointMotorControl2(
            car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0
        )

    steering = [4, 6]

    targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -30, 30, 0)
    maxForceSlider = p.addUserDebugParameter("maxForce", 0, 10, 10)
    steeringSlider = p.addUserDebugParameter("steering", -1, 1, 0)
    epsilon = 0.1
    create_poly_wall(
        [
            (-1, 3),
            (0, 10),
            (1, 3),
            (10, 10),
            (3, 1),
            (10, 0),
            (3, -1),
            (10, -10),
            (1, -3),
            (0, -10),
            (-1, -3),
            (-10, -10),
            (-3, -1),
            (-10, 0),
            (-3, 1),
            (-10, 10),
            (-2, 4),
        ],
        epsilon,
    )

    while True:
        maxForce = p.readUserDebugParameter(maxForceSlider)
        targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
        steeringAngle = p.readUserDebugParameter(steeringSlider)

        for wheel in wheels:
            p.setJointMotorControl2(
                car,
                wheel,
                p.VELOCITY_CONTROL,
                targetVelocity=targetVelocity,
                force=maxForce,
            )

        for steer in steering:
            p.setJointMotorControl2(
                car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle
            )
            # if useRealTimeSim == 0:
            #    p.stepSimulation()
            # time.sleep(0.01)
        ## first ball


if __name__ == "__main__":
    main()