import torch
from RL import NeuralNetwork
from trainer import Trainer


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
    for _ in range(3):
        trainer.breed()


if __name__ == "__main__":
    main()
