from calendar import EPOCH
import torch
from torch import nn, optim
from collections import namedtuple, deque
from torch.optim.lr_scheduler import MultiStepLR

import random
import math
import matplotlib.pyplot as plt
import simulator
from trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"


## reward constants:
DISTANCE_REWARD = 1.0
EXPLORATION_REWARD = 1.0
END_REWARD = 10000.0  # 10^5
TIME_PENALTY = -5.0  # at each frame
CRUSH_PENALTY = -1000000  # once


added_params = [
    ("curr_loc", 2),
    ("target_loc", 2),
    ("curr_speed", 1),
    ("current_swivel", 1),
    ("current_orientation", 1),
    ("current_acc", 1),
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
        lines = x[-1]
        lines = self.lines_to_features(lines)
        # maybe max_pool_2d kernel = [num, 1]
        features = self.get_features(lines, dim=1)

        move = self.calculte_move(x[:-1] + features)
        return move

    def get_weights(self):
        l1 = [i for i in self.lines_to_features if isinstance(i, nn.Linear)]
        l2 = [i for i in self.calculte_move if isinstance(i, nn.Linear)]
        return l1 + l2


def main():
    torch.set_grad_enabled(False)
    EPOCHS = 1000
    model = NeuralNetwork
    population = 10  # Total Population
    trainer = Trainer(
        model, EPOCHS, population, mutatuion_rate=1, max_iter=100, breed_percent=0.5
    )  # change data


def calculate_score(car, episode_time_length, training_set):
    for map, starting_point, end_point in training_set:
        distance_covered, map_discovered, finished, time, crushed = simulator.run_sim(
            car, episode_time_length, map, starting_point, end_point
        )
        total_reward += distance_covered * DISTANCE_REWARD
        +map_discovered * EXPLORATION_REWARD
        +finished * END_REWARD
        +time * TIME_PENALTY
        +crushed * CRUSH_PENALTY
    return total_reward / len(training_set)


if __name__ == "__main__":
    main()
