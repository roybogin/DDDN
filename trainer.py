import os
import torch
from RL import calculate_score

class Trainer:
    def __init__(self, model, epochs, population_count, mutation_rate, max_iter):
        self.mutation_rate = mutation_rate
        self.model = model
        self.max_iter = max_iter
        self.epochs = epochs
        self.population_count = population_count
        self.population = [model() for _ in range(population_count)]
        self.evaluations = None
        
    def mutate(self):
        for car in self.population:
            for _, v in car.named_parameters():
                v.data = v.data + torch.normal(torch.zeros_like(v.data), torch.full(v.data.size, self.mutation_rate))
    
    def breed(self):
        self.evaluate()
        

    

    def evaluate(self):
        self.evaluations= []
        for car in self.population:
            score = calculate_score(car)
            self.evaluations.append((car, score))
        self.evaluations.sort(key = lambda x: x[1], reverse = True)

            




# def save_checkpoint(self, state):
#     """Save checkpoint."""
#     if not os.path.exists(self.ckptroot):
#         os.makedirs(self.ckptroot)
#
#     torch.save(state, self.ckptroot + 'model-{}.h5'.format(state['epoch']))
