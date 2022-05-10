import os
import torch
import numpy as np
from RL import calculate_score
from itertools import combinations


class Trainer:
    def __init__(
        self,
        model,
        epochs,
        population_count,
        mutation_rate,
        breed_percent,
        training_set=None,
        episode_time_length=1000,
    ):
        self.mutation_rate = mutation_rate
        self.model = model
        self.epochs = epochs
        self.population_count = population_count
        self.population = [model() for _ in range(population_count)]
        self.evaluations = None
        self.breed_percent = breed_percent
        self.episode_time_length = episode_time_length
        self.training_set = training_set

    def mutate(self):
        for car in self.population:
            self.mutate_one(car)

    def mutatue_one(self, model):
        for _, v in model.named_parameters():
            v.data = v.data + torch.normal(
                torch.zeros_like(v.data), torch.full(v.data.size, self.mutation_rate)
            )

    def breed(self):
        self.evaluate()
        best_cars = self.evaluations[
            : (int)(self.breed_percent * self.population_count)
        ]
        next_gen = [a[0] for a in best_cars]
        car_cnt = len(next_gen)
        add_amt = self.population_count - car_cnt
        options = list(combinations(range(car_cnt), 2))
        chosen = np.random.choice(options, add_amt)
        self.population = next_gen
        for ind in chosen:
            self.population.append(self.breed_models([next_gen[i] for i in ind]))
        self.mutate()

    def breed_models(self, *args):
        new_model = self.model()
        for k, v in new_model.named_parameters():
            weights = torch.empty(len(args), *v.size)
            for i, car in enumerate(args):
                weights[i] = car[k].data  # probably doesnt work - fuck you dvir
            v.data = weights.mean(dim=0)  # maybe wrong dim - fuck you dvir
        return new_model

    def evaluate(self):
        self.evaluations = []
        for car in self.population:
            score = calculate_score(
                car, self.episode_time_length, self.training_set
            )  # need to implement
            self.evaluations.append((car, score))
        self.evaluations.sort(key=lambda x: x[1], reverse=True)


# def save_checkpoint(self, state):
#     """Save checkpoint."""
#     if not os.path.exists(self.ckptroot):
#         os.makedirs(self.ckptroot)
#
#     torch.save(state, self.ckptroot + 'model-{}.h5'.format(state['epoch']))
