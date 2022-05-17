import torch
import numpy as np
from RL import calculate_score
import itertools
from functools import reduce
from scipy import sparse
from scipy import stats
import scipy
import consts
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        model,  # the NN class
        population_count,   # amount of models each step
        mutation_rate,  # variancce of mutation matrix
        breed_percent,  # percent of population to keep for the next iteration
        training_set=None,
        episode_time_length=1000,   # how many steps for each simulation
    ):
        self.mutation_rate = mutation_rate
        self.model = model
        self.population_count = population_count
        self.population = [model().to(consts.device) for _ in range(population_count)]
        self.evaluations = None
        self.breed_percent = breed_percent
        self.episode_time_length = episode_time_length
        self.training_set = training_set

    def mutation_density(self, size):
        return max(consts.initial_mutation_density, 1/size)

    def mutate(self):
        for car in self.population[1:]:
            self.mutate_one(car)

    def mutate_one(self, model):
        for _, v in model.named_parameters():
            mat_size = list(v.size())
            if len(mat_size) == 1:
                mat_size = [1] + mat_size
            rvs = stats.norm(loc=0, scale=self.mutation_rate).rvs
            mutation = sparse.random(mat_size[0], mat_size[1], self.mutation_density(mat_size[0] * mat_size[1]), data_rvs=rvs).todense()
            mutation_tensor = torch.from_numpy(mutation).squeeze().to(dtype=torch.float32).to(consts.device)
            # print(consts.device_name)
            # print(mutation_tensor.is_cuda)
            v.data = v.data + mutation_tensor.data 
            

    def breed(self):
        self.evaluate()
        best_cars = self.evaluations[
            : (int)(self.breed_percent * self.population_count)
        ]
        next_gen = [a[0] for a in best_cars]
        car_cnt = len(next_gen)
        add_amt = self.population_count - car_cnt
        if consts.duplicate_best:
            add_amt -= 1
        best_scores = np.array([a[1] for a in best_cars])
        best_scores -= np.min(best_scores)
        best_scores /= np.max(best_scores)  # normaize between 0 and 1
        best_scores = scipy.special.softmax(best_scores)
        options = None
        probabilities = None
        
        if consts.breed_with_self:
            options = np.array(list(itertools.combinations_with_replacement(range(car_cnt), 2)))
            probabilities = np.empty(len(options), dtype=np.double) # probability to choose i and j as a pair
            for idx, pair in enumerate(options):
                i = pair[0]
                j = pair[1]
                if i == j:
                    probabilities[idx] = best_scores[i] ** 2
                else:
                    probabilities[idx] = 2 * best_scores[i] * best_scores[j]
        else:
            options = np.array(list(itertools.combinations(range(car_cnt), 2)))
            probabilities = np.empty(len(options), dtype=np.double) # probability to choose i and j as a pair
            for idx, pair in enumerate(options):
                i = pair[0]
                j = pair[1]
                prob1 = best_scores[i]
                prob2 = best_scores[j]
                probabilities[idx] = prob1 * prob2 * (1/(1-prob1) + 1/(1-prob2))


        chosen_idx = np.random.choice(len(options), add_amt, replace=consts.breed_same_pair, p=probabilities)
        chosen = options[chosen_idx]

        self.population = next_gen
        for ind in chosen:
            self.population.append(self.breed_models([next_gen[i] for i in ind]))

        if consts.duplicate_best:
            self.population.append(self.breed_models([self.population[0]]))

        self.mutate()

    def breed_models(self, models):

        def get_module_by_name(module, access_string):
            names = access_string.split(sep=".")
            return reduce(getattr, names, module)

        new_model = self.model().to(consts.device)
        for k, v in new_model.named_parameters():
            indices = np.random.randint(len(models), size=tuple(v.size()))
            new_weight = torch.zeros_like(v.data).to(consts.device)
            for i, car in enumerate(models):
                new_weight += get_module_by_name(
                    car, k
                ).data * (indices == i)
            v.data = new_weight
        return new_model

    def evaluate(self):
        self.evaluations = []
        for car in self.population:
            score = calculate_score(
                car, self.episode_time_length, self.training_set
            )  # need to implement
            self.evaluations.append((car, score))
        # print('mutation of best', self.evaluations[-1][1])
        # print('last oter_best best', self.evaluations[0][1])
        self.evaluations.sort(key=lambda x: x[1], reverse=True)
        print("best:", self.evaluations[0][1])
        lst = [self.evaluations[i][1] for i in range(len(self.evaluations))]
        # print("list:", lst)
        print("average:", sum(lst) / len(lst))
        # plt.plot(
        #     [i + 1 for i in range(len(self.evaluations))],
        #     lst,
        # )
        # plt.show()


# def save_checkpoint(self, state):
#     """Save checkpoint."""
#     if not os.path.exists(self.ckptroot):
#         os.makedirs(self.ckptroot)
#
#     torch.save(state, self.ckptroot + 'model-{}.h5'.format(state['epoch']))
