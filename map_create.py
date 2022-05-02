import math
def create_wall(p, pos, orientation, length, width):

	boxHalfLength = length / 2
	boxHalfWidth = width / 2
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
	basePosition = pos
	baseOrientation = orie\

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
	p.resetBasePositionAndOrientation(block, pos, orientation)

	p.enableJointForceTorqueSensor(block, 0, enableSensor=1)


def distance(p1, p2):
	return math.sqrt(pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2))

def create_poly_wall(p, poly, epsilon):
	length = distance(poly[0], poly[1]) + 2 * epsilon
	width = 2 * epsilon
	angle = math.atan2(poly[1][1] - poly[0][1], poly[1][0] - poly[0][0])
	bottom_left = angle - math.pi * 5 / 4
	if bottom_left <= - math.pi:
		bottom_left  += 2*math.pi
	x_diff = math.sqrt(2) * epsilon * math.cos(bottom_left)
	y_diff = math.sqrt(2) * epsilon * math.sin(bottom_left)
	euler = [0, 0, angle]
	orientation = p.getQuaternionFromEuler(euler)
	pos = [poly[0][0]+x_diff, poly[0][1]+y_diff, 0]
	create_wall(p, pos, orientation, length, width)
	for i in range(1, len(poly) - 1):
		length = distance(poly[0], poly[1]) + (1-math.sqrt(2))*epsilon
		width = 2 * epsilon
		angle = math.atan2(poly[i+1][1] - poly[i][1], poly[i+1][0] - poly[i][0])
		bottom_left = angle - math.pi * 5 / 4
		if bottom_left <= - math.pi:
			bottom_left  += 2*math.pi
		x_diff = math.sqrt(2) * epsilon * math.cos(bottom_left) + (1+math.sqrt(2)) * epsilon * math.cos(angle)
		y_diff = math.sqrt(2) * epsilon * math.sin(bottom_left) + (1+math.sqrt(2)) * epsilon * math.sin(angle)
		euler = [0, 0, angle]
		orientation = p.getQuaternionFromEuler(euler)
		pos = [poly[i][0]+x_diff, poly[i][1]+y_diff, 0]
		create_wall(p, pos, orientation, length, width)


def create_map(p, map, epsilon):
	for poly in 