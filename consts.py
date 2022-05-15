import torch

is_visual = False
use_real_time = 0
debug_sim = False
speed_scalar = 1
steer_scalar = 1
max_steer = 0.7
max_velocity = 18
map_borders = [[(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)]]
min_dist_to_target = 0.5
ray_length = 10
initial_mutation_density = 0.1
print_reward_breakdown = False
cameraDistance = 7
cameraYaw = -89.9
cameraPitch = -89.9
cameraTargetPosition = [0, 0, 0]
record = False
video_name = "vid"
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


# breeding options
breed_same_pair = False
breed_with_self = False
