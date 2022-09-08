# pybullet simulation

import math

import numpy as np

is_visual = not True  # do we want to see visual footage
use_real_time = 0  # is the simulation running in real time - probably should always be 0
time_step = 0.01    # what is a time step in the pybullet simulation

# debugging flags:
debugging = True    # are we debugging the code
drawing = True  # do we want to plot the paths
show_scaned_maze = True
show_projected_path = True
show_actual_path = True
show_goal_starting_points = True
print_runtime = True  # do we want to print the total time of the run


# camera settings for visual pybullet
cameraDistance = 11
cameraYaw = 0
cameraPitch = -89.9
cameraTargetPosition = [0, 0, 0]

max_steer = np.pi / 4   # maximum steering allowed for PRM
max_velocity = 10   # maximum car velocity
max_force = 100  # pybullet force
epsilon = 0.1   # margin for walls and edge removal

min_dist_to_target = 0.5  # distance from target that is treated as success
ray_length = 5  # length of ray for wall detection
ray_amount = 6  # amount of rays
vertex_offset = 0.11    # offset between vertices

directions_per_vertex = 36  # amount of angles allowed for each vertex
amount_vertices_from_edge = math.ceil(0.3 / vertex_offset)  # the offset from possible vertices in the given map, as vertices too close to the wall will crash the car

max_time = int(2.5e4)  # time before forcing a new maze

length = 0.325  # car length
width = 0.2     # car width
a_2 = 0.1477    # a_2 of the car (distance to center of mass)

minimum_car_dist = 1    # minimum distance allowed between cars

max_hits_before_calculation = 10  # amounts of new hits before adding lines to the map
calculate_action_time = 50  # maximum amount of steps to calculate a new action (velocity and wheel rotation)
calculate_d_star_time = 50  # maximum amount of steps to recalculate the d_star in case of edge removal
reset_count_time = 200      # maximum allowed of steps to reset the counter (and also choose a new target vertex)
backwards_driving_steps = 4  # amount of steps we want to drive backwards if needed
scan_time = 1  # time between scans

