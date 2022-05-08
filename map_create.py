import math
import pybullet as p
import pybullet_data
import os
def create_wall(p, pos, orientation, length, width):

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

def distance(p1, p2):
	return math.sqrt(pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2))

def create_poly_wall(p, poly, epsilon):
	length = distance(poly[0], poly[1])
	width = 2 * epsilon
	angle = math.atan2(poly[1][1] - poly[0][1], poly[1][0] - poly[0][0])
	bottom_left = angle - math.pi * 5 / 4
	if bottom_left <= - math.pi:
		bottom_left  += 2*math.pi
	x_diff = length / 2 * math.cos(angle) + math.sqrt(2) * epsilon * math.cos(bottom_left)
	y_diff = length / 2 * math.sin(angle) + math.sqrt(2) * epsilon * math.sin(bottom_left)
	euler = [0, 0, angle]
	orientation = p.getQuaternionFromEuler(euler)
	pos = [poly[0][0]+x_diff, poly[0][1]+y_diff, 0.5]
	create_wall(p, pos, orientation, length, width)
	for i in range(1, len(poly) - 1):
		length = distance(poly[0], poly[1]) + (1-math.sqrt(2))*epsilon
		width = 2 * epsilon
		angle = math.atan2(poly[i+1][1] - poly[i][1], poly[i+1][0] - poly[i][0])
		bottom_left = angle - math.pi * 5 / 4
		if bottom_left <= - math.pi:
			bottom_left  += 2*math.pi
		x_diff = length / 2 * math.cos(angle) + math.sqrt(2) * epsilon * math.cos(bottom_left) + (math.sqrt(2)) * epsilon * math.cos(angle)
		y_diff = length / 2 * math.sin(angle) + math.sqrt(2) * epsilon * math.sin(bottom_left) + (math.sqrt(2)) * epsilon * math.sin(angle)
		euler = [0, 0, angle]
		orientation = p.getQuaternionFromEuler(euler)
		pos = [poly[i][0]+x_diff, poly[i][1]+y_diff, 0.5]
		create_wall(p, pos, orientation, length, width)


def create_map(p, in_map, epsilon):
	for poly in in_map:
		create_poly_wall(p, poly, epsilon)

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
	    p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

	steering = [4, 6]

	targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -30, 30, 0)
	maxForceSlider = p.addUserDebugParameter("maxForce", 0, 10, 10)
	steeringSlider = p.addUserDebugParameter("steering", -1, 1, 0)
	epsilon = 0.01
	create_poly_wall(p, [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)], epsilon)
	create_poly_wall(p, [(-1,-1), (-2,-2),(-3,3)], epsilon)

	while True:
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
		## first ball


if __name__ == "__main__":
	main()