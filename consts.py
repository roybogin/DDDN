import torch

# pybullet simulation
is_visual = False  # do we want to see visual footage
use_real_time = (
    0  # is the simulation running in real time - probably should always be 0
)
debug_sim = False
map_borders = [[(35, 35), (35, -35), (-35, -35), (-35, 35), (35, 35)]]
min_dist_to_target = 0.5

initial_mutation_density = 0.01  # what percent of weights is mutated (at beginning)
print_reward_breakdown = False

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
ray_length = 10  # length of ray
print_reward_breakdown = False
size_map_quarter = 10
block_size = 0.3
map_borders = [
    (size_map_quarter, size_map_quarter),
    (size_map_quarter, -size_map_quarter),
    (-size_map_quarter, -size_map_quarter),
    (-size_map_quarter, size_map_quarter),
    (size_map_quarter, size_map_quarter),
]

record = True
video_name = "vid.mp4"
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
path_to_save = "saved_run"
path_extentions = ".txt"

# breeding options
breed_same_pair = True  # can a pair be chosen more than once
breed_with_self = False  # can a car be chosen with itself
amount_to_save = 1  # how many of the best to save

## reward constants:
DISTANCE_REWARD = 1.0
EXPLORATION_REWARD = 1.0
MIN_DIST_PENALTY = -10.0
END_REWARD = 10000.0  # 10^5
TIME_PENALTY = -5.0  # at each frame
CRUSH_PENALTY = -1000000  # once

# scores or smthng
best_scores = []  # a list of the best scores, built through simulations
average_scores = []  # averages of populations

max_hits_before_calculation = 10  # amounts of new hits before adding lines to the map
max_time = 10000  # time before forcing a new maze

num_processes = 4