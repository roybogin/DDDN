# pybullet simulation
is_visual = False   # do we want to see visual footage
use_real_time = 0   # is the simulation running in real time - probably should always be 0
debug_sim = False
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
min_dist_to_target = 0.5    # distance from target that is treated as success
ray_length = 10 # length of ray
print_reward_breakdown = False


record = False
video_name = "vid"
path_to_save = "saved_run.txt"