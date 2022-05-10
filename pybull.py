import os, inspect
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
parentdir = os.path.join(currentdir, "../gym")


os.sys.path.insert(0, parentdir)

from scan_to_map import Map
import pybullet as p
import pybullet_data
import map_create
import time


def addLists(lists):
    ret = [0, 0, 0]
    for l in lists:
        for i in range(len(l)):
            ret[i] += l[i]
    return ret


def rayCast(p, car, offset, direction):
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


hits = []
map = Map([], 10, 0.4)
cid = p.connect(p.SHARED_MEMORY)
if cid < 0:
    p.connect(p.GUI)

p.resetSimulation()
p.setGravity(0, 0, -10)

useRealTimeSim = 0

map_create.create_poly_wall(
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
    epsilon=0.1,
)

p.setRealTimeSimulation(useRealTimeSim)  # either this
# p.loadURDF("plane.urdf")
p.loadSDF(os.path.join(pybullet_data.getDataPath(), "stadium.sdf"))

car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"))
inactive_wheels = [3, 5, 7]
wheels = [2]

for wheel in inactive_wheels:
    p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

steering = [4, 6]

targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -15, 15, 0)
maxForceSlider = p.addUserDebugParameter("maxForce", 0, 10, 10)
steeringSlider = p.addUserDebugParameter("steering", -1, 1, 0)


count = 0
while True:
    hit = rayCast(p, car, [0, 0, 0], [-10, 0, 0])
    if hit != (0, 0, 0):
        hits.append((hit[0], hit[1]))
        if len(hits) == 500:
            map.add_points_to_map(hits)
            hits = []
            count += 1
            if count == 1:
                print("show")
                map.show()
                count = 0
    maxForce = p.readUserDebugParameter(maxForceSlider)
    targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
    steeringAngle = p.readUserDebugParameter(steeringSlider)
    # print(targetVelocity)

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
    if useRealTimeSim == 0:
        p.stepSimulation()
    # time.sleep(0.01)
