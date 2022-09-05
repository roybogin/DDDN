# pybullet simulation
from typing import DefaultDict, Tuple

import numpy as np
import math

is_visual = False  # do we want to see visual footage
use_real_time = 0  # is the simulation running in real time - probably should always be 0
time_step = 0.01    # what is a time step in the pybullet simulation

cameraDistance = 11
cameraYaw = 0
cameraPitch = -89.9
cameraTargetPosition = [0, 0, 0]

# interpreting NN outputs
max_steer = np.pi / 4
max_velocity = 10
max_force = 100
epsilon = 0.1

min_dist_to_target = 0.5  # distance from target that is treated as success
ray_length = 5  # length of ray
ray_amount = 6
size_map_quarter = 8
vertex_offset = 0.11
map_borders = [
    (size_map_quarter, size_map_quarter),
    (size_map_quarter, -size_map_quarter),
    (-size_map_quarter, -size_map_quarter),
    (-size_map_quarter, size_map_quarter),
    (size_map_quarter, size_map_quarter),
]

seed = None  # randomness seed


directions_per_vertex = 36
amount_vertices_from_edge = math.ceil(0.3 / vertex_offset)

max_hits_before_calculation = 10  # amounts of new hits before adding lines to the map
max_time = int(5e4)  # time before forcing a new maze
print_runtime = True  # do we want to print the total time of the run

length = 0.325
width = 0.2
a_2 = 0.1477  # a_2 of the car

minimum_car_dist = 1


calculate_action_time = 50
calculate_d_star_time = 50
reset_count_time = 150
