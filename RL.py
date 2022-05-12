from calendar import EPOCH
import torch
from torch import nn, optim
from collections import namedtuple, deque
from torch.optim.lr_scheduler import MultiStepLR

import random
import math
import matplotlib.pyplot as plt
import simulator

device = "cuda" if torch.cuda.is_available() else "cpu"


## reward constants:
DISTANCE_REWARD = 1.0
EXPLORATION_REWARD = 1.0
END_REWARD = 10000.0  # 10^5
TIME_PENALTY = -5.0  # at each frame
CRUSH_PENALTY = -1000000  # once


def flatten(lst):
    ret = []
    for o in lst:
        if hasattr(o, "__iter__"):
            ret += [a for a in o]
        else:
            ret.append(o)
    return ret


added_params = [
    ("curr_loc", 2),
    ("target_loc", 2),
    ("curr_speed", 1),
    ("current_swivel", 1),
    ("current_orientation", 1),
    ("current_acc", 1),
    ("time", 1),
]

param_size = sum((a[1] for a in added_params))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lines_to_features = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
        )
        self.get_features = torch.max
        self.calculte_move = nn.Sequential(
            nn.Linear(param_size + 1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        lines = torch.FloatTensor(x[-1])
        lines = self.lines_to_features(lines)
        features, _ = self.get_features(lines, dim=0)

        attrib = torch.FloatTensor(flatten(x[:-1]))
        net_inp = torch.cat((attrib, features))

        move = self.calculte_move(net_inp)
        return move.numpy()

    def get_weights(self):
        l1 = [i for i in self.lines_to_features if isinstance(i, nn.Linear)]
        l2 = [i for i in self.calculte_move if isinstance(i, nn.Linear)]
        return l1 + l2


def calculate_score(car, episode_time_length, training_set):
    total_reward = 0
    i = 0
    for map, starting_point, end_point in training_set:
        i += 1
        distance_covered, map_discovered, finished, time, crushed = simulator.run_sim(
            car, episode_time_length, map, starting_point, end_point
        )

        total_reward += (
            (distance_covered * DISTANCE_REWARD)
            + (map_discovered * EXPLORATION_REWARD)
            + (finished * END_REWARD)
            + (time * TIME_PENALTY)
            + (crushed * CRUSH_PENALTY)
        )

        # print(i, "* DISTANCE_REWARD = ", distance_covered * DISTANCE_REWARD)
        # print(i, "EXPLORATION_REWARD = ", map_discovered * DISTANCE_REWARD)
        # print(i, "END_REWARD = ", finished * END_REWARD)
        # print(i, " TIME_PENALTY = ", time * TIME_PENALTY)
        # print(i, "CRUSH_PENALTY = ", crushed * CRUSH_PENALTY)

    # print("total = ", total_reward)
    return total_reward / len(training_set)
