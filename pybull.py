import os, inspect
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
parentdir = os.path.join(currentdir, "../gym")


os.sys.path.insert(0, parentdir)

from scan_to_map import Map
import pybullet as p
import pybullet_data

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
map = Map([], 10, 0.15)
cid = p.connect(p.SHARED_MEMORY)
if cid < 0:
    p.connect(p.GUI)

p.resetSimulation()
p.setGravity(0, 0, -10)

useRealTimeSim = 1


boxHalfLength = 0.05
boxHalfWidth = 1
boxHalfHeight = 0.5
body = p.createCollisionShape(
    p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight]
)
pin = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
wgt = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])

mass = 10000
visualShapeId = -1
nlnk = 2
link_Masses = [0, 1]
linkCollisionShapeIndices = [pin, wgt]
linkVisualShapeIndices = [-1] * nlnk
linkPositions = [[0.0, 0.0, 0.0], [1.0, 0, 0]]
linkOrientations = [[0, 0, 0, 1]] * nlnk
linkInertialFramePositions = [[0, 0, 0]] * nlnk
linkInertialFrameOrientations = [[0, 0, 0, 1]] * nlnk
indices = [0, 1]
jointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
axis = [[0, 0, 1], [0, 1, 0]]
basePosition = [0, 0, 0.5]
baseOrientation = [0, 0, 0, 1]

# block creation
block = p.createMultiBody(
    mass,
    body,
    visualShapeId,
    basePosition,
    baseOrientation,
    linkMasses=link_Masses,
    linkCollisionShapeIndices=linkCollisionShapeIndices,
    linkVisualShapeIndices=linkVisualShapeIndices,
    linkPositions=linkPositions,
    linkOrientations=linkOrientations,
    linkInertialFramePositions=linkInertialFramePositions,
    linkInertialFrameOrientations=linkInertialFrameOrientations,
    linkParentIndices=indices,
    linkJointTypes=jointTypes,
    linkJointAxis=axis,
)


# block set pos and rotation
p.resetBasePositionAndOrientation(block, [1, 1, 0.5], [0, 0, 0, 1])

p.enableJointForceTorqueSensor(block, 0, enableSensor=1)


boxHalfLength = 1
boxHalfWidth = 0.05
boxHalfHeight = 0.5
body = p.createCollisionShape(
    p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight]
)
pin = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
wgt = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])

mass = 10000
visualShapeId = -1
nlnk = 2
link_Masses = [0, 1]
linkCollisionShapeIndices = [pin, wgt]
linkVisualShapeIndices = [-1] * nlnk
linkPositions = [[0.0, 0.0, 0.0], [1.0, 0, 0]]
linkOrientations = [[0, 0, 0, 1]] * nlnk
linkInertialFramePositions = [[0, 0, 0]] * nlnk
linkInertialFrameOrientations = [[0, 0, 0, 1]] * nlnk
indices = [0, 1]
jointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
axis = [[0, 0, 1], [0, 1, 0]]
basePosition = [0.1, 0, 0.5]
baseOrientation = [0, 0, 0, 1]

# block creation
block = p.createMultiBody(
    mass,
    body,
    visualShapeId,
    basePosition,
    baseOrientation,
    linkMasses=link_Masses,
    linkCollisionShapeIndices=linkCollisionShapeIndices,
    linkVisualShapeIndices=linkVisualShapeIndices,
    linkPositions=linkPositions,
    linkOrientations=linkOrientations,
    linkInertialFramePositions=linkInertialFramePositions,
    linkInertialFrameOrientations=linkInertialFrameOrientations,
    linkParentIndices=indices,
    linkJointTypes=jointTypes,
    linkJointAxis=axis,
)


# block set pos and rotation
p.resetBasePositionAndOrientation(block, [-0.1, 1, 0.5], [0, 0, 0, 1])

p.enableJointForceTorqueSensor(block, 0, enableSensor=1)



# for video recording (works best on Mac and Linux, not well on Windows)
# p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
p.setRealTimeSimulation(useRealTimeSim)  # either this
# p.loadURDF("plane.urdf")
p.loadSDF(os.path.join(pybullet_data.getDataPath(), "stadium.sdf"))

car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"))
inactive_wheels = [3, 5, 7]
wheels = [2]

for wheel in inactive_wheels:
    p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

steering = [4, 6]

targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -10, 10, 0)
maxForceSlider = p.addUserDebugParameter("maxForce", 0, 10, 10)
steeringSlider = p.addUserDebugParameter("steering", -0.5, 0.5, 0)


while True:
    hit = rayCast(p, car, [0, 0, 0], [-10, 0, 0])
    if hit != (0, 0, 0):
        hits.append((hit[0], hit[1]))
        print("hit!")
        if len(hits) == 10:
            map.add_points_to_map(hits)
            hits = []
            print("show")
            map.show()

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

    steering
    if useRealTimeSim == 0:
        p.stepSimulation()
    time.sleep(0.01)
