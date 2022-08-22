# pybullet simulation
from gym.utils import seeding

is_visual = False  # do we want to see visual footage
use_real_time = 0  # is the simulation running in real time - probably should always be 0
time_step = 0.01    # what is a time step in the pybullet simulation

cameraDistance = 11
cameraYaw = -89.9
cameraPitch = -89.9
cameraTargetPosition = [0, 0, 0]

# interpreting NN outputs
speed_scalar = 1
steer_scalar = 0.1
max_steer = 0.3
max_velocity = 30
max_force = 100


min_dist_to_target = 0.5  # distance from target that is treated as success
ray_length = 5  # length of ray
ray_amount = 6
size_map_quarter = 10
block_size = 0.2
map_borders = [
    (size_map_quarter, size_map_quarter),
    (size_map_quarter, -size_map_quarter),
    (-size_map_quarter, -size_map_quarter),
    (-size_map_quarter, size_map_quarter),
    (size_map_quarter, size_map_quarter),
]

seed = None  # randomness seed


max_hits_before_calculation = 10  # amounts of new hits before adding lines to the map
max_time = int(1.5e4)  # time before forcing a new maze
print_runtime = False  # do we want to print the total time of the run


length=0.325
width=0.2
a_2 = 0.1477 # a_2 of the car TODO: fill

sample_amount = 10000

