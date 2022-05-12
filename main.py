from re import I
import torch
from RL import NeuralNetwork
from trainer import Trainer
from RL import calculate_score
import consts
import time
from matplotlib import pyplot as plt

default_data_set = [
    ([[(35,35), (35, -35), (-35,-35), (-35, 35), (34.5, 35)]], [0,0,0], [1,1,0]), #empty
    ([[(35,35), (35, -35), (-35,-35), (-35, 35), (34.5, 35)], [(2.5, 5), (2.5, -2.5)]], [0,0,0], [5,0,0]), #one wall inbetween
    ([[(35,35), (35, -35), (-35,-35), (-35, 35), (34.5, 35)], [(-4, 0), (0, 4)], [(-3,5), (3,5)], [(-7,4), (-3,3)]], [0,0,0], [-4,4,0]), #raish shaped wall
    ([[(35,35), (35, -35), (-35,-35), (-35, 35), (34.5, 35)], [(0,2), (2,-1), (-2,-1), (-4, 0), (-3,-4), (-4,-5)]], [0,0,0], [-4,-4,0]), #some walls in between the path
    ([[(35,35), (35, -35), (-35,-35), (-35, 35), (34.5, 35)]] [(2,1), (6,2), (9,1),(9,-1), (6,-1), (4,-0.5)], [0,0,0], [0,8,0]) # alot of walls around the endpoint that dont really disrupt the path
]

default_training_set = [
    ([[(35,35), (35, -35), (-35,-35), (-35, 35), (34.5, 35)], [(4,0), (0,2), (4,4), (5,4), (5,-4)]], [0,0,0], [4,2,0]), #some walls around the path, rather easy
    ([[(35,35), (35, -35), (-35,-35), (-35, 35), (34.5, 35)], [(6,-6), (6,-2), (2,-6), (0,-4), (0,4), (-2,-2), (-2, 2), (8, 2), (8, -6)]], [0,0,0], [2,-4,0]) # ben hagadol
]
def main():
    torch.set_grad_enabled(False)
    EPOCHS = 1000
    model = NeuralNetwork
    population = 100  # Total Population
    trainer = Trainer(
        model,
        EPOCHS,
        population,
        mutation_rate=1,
        episode_time_length=700,
        breed_percent=0.5,
        training_set=[(([], [0, 0, 0], [1, 1, 0]))],
    )  # change data

    for i in range(3):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(i, "th iteration, time: ", current_time)
        trainer.breed()

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("finished ,time: ", current_time)

    consts.is_visual = True
    plt.plot([1], [1])
    plt.show()  # use this to stop the last simulation
    consts.debug_sim = True
    print(calculate_score(trainer.population[0], 10000, trainer.training_set))


if __name__ == "__main__":
    main()
