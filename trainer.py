import os
import torch

class Trainer:
    def __init__(self, model, epochs, population_count, mutation_rate):
        self.mutation_rate = mutation_rate
        self.model = model
        self.epochs = epochs
        self.population_count = population_count
        self.population = [model() for _ in range(population_count)]
        
    def mutate(self):
        for car in self.population:
            for param in car.parameters():
                pass
    
    def breed(self):
        pass

    def evaluate(self):
        pass
            




# def save_checkpoint(self, state):
#     """Save checkpoint."""
#     if not os.path.exists(self.ckptroot):
#         os.makedirs(self.ckptroot)
#
#     torch.save(state, self.ckptroot + 'model-{}.h5'.format(state['epoch']))
