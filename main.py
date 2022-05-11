from re import I
import torch
from RL import NeuralNetwork
from trainer import Trainer
from RL import calculate_score


def main():
    torch.set_grad_enabled(False)
    EPOCHS = 1000
    model = NeuralNetwork
    population = 10  # Total Population
    trainer = Trainer(
        model,
        EPOCHS,
        population,
        mutation_rate=1,
        episode_time_length=100,
        breed_percent=0.5,
        training_set=[(([], [0, 0, 0], [1, 1, 1]))],
    )  # change data
    car = trainer.population[0]

    for _ in range(10):
        print(calculate_score(car, 100, trainer.training_set))


if __name__ == "__main__":
    main()
