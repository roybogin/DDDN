import torch

# pybullet simulation
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
print_reward_breakdown = False
size_map_quarter = 10
block_size = 0.2
map_borders = [
    (size_map_quarter, size_map_quarter),
    (size_map_quarter, -size_map_quarter),
    (-size_map_quarter, -size_map_quarter),
    (-size_map_quarter, size_map_quarter),
    (size_map_quarter, size_map_quarter),
]

# reward constants:
DISTANCE_REWARD = 0.05  # reward for high speed
DISCOVER_REWARD = 50    # reward for discovering more of the map
MIN_DIST_PENALTY = -0.005   # reward for getting close to the target
FINISH_REWARD = 1000    # reward for finishing the maze
TIME_PENALTY = -0.01    # reward for finishing in a short time
CRASH_PENALTY = -1000   # reward for not crashing


max_hits_before_calculation = 10  # amounts of new hits before adding lines to the map
max_time = int(1.5e4)  # time before forcing a new maze
print_runtime = False  # do we want to print the total time of the run

is_model_load = True   # do we want to load a model
loaded_model_path = None    # the loaded model filename - None means the latest
checkpoint_steps = int(6e4)  # how many steps we want between checkpointing

num_processes = 4   # amount of processes for multiprocessing
