import torch

# pybullet simulation
is_visual = False  # do we want to see visual footage
use_real_time = (
    0  # is the simulation running in real time - probably should always be 0
)
debug_sim = False

max_force = 50
map_borders = [[(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)]]
min_dist_to_target = 0.5

initial_mutation_density = 0.01  # what percent of weights is mutated (at beginning)
print_reward_breakdown = False

cameraDistance = 7
cameraYaw = -89.9
cameraPitch = -89.9
cameraTargetPosition = [0, 0, 0]

# interpreting NN outputs
speed_scalar = 1
steer_scalar = 0.1
max_steer = 0.7
max_velocity = 18


map_borders = [[(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)]]
min_dist_to_target = 0.5  # distance from target that is treated as success
ray_length = 10  # length of ray
print_reward_breakdown = False


record = False
video_name = "vid"
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


# breeding options
breed_same_pair = True  # can a pair be chosen more than once
breed_with_self = False  # can a car be chosen with itself
amount_to_save = 1  # how many of the best to save
path_to_save = "saved_run.txt"


best_scores = []  # a list of the best scores, built through simulations
average_scores = []  # averages of populations
