from re import I
import torch
from RL import NeuralNetwork
from trainer import Trainer
from RL import calculate_score
import consts
import time
from matplotlib import pyplot as plt


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
        episode_time_length=1000,
        breed_percent=0.5,
        training_set=[(([], [0, 0, 0], [1, 1, 0]))],
    )  # change data

    for i in range(1):
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
    print(calculate_score(trainer.population[0], 1000, trainer.training_set))


if __name__ == "__main__":
    main()
