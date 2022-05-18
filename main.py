from re import I
import torch
from RL import NeuralNetwork
from trainer import Trainer
from RL import calculate_score
import consts
import time
from matplotlib import pyplot as plt

default_data_set = [
    (
        [[(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)]],
        [0, 0, 0],
        [3, 4, 0],
    ),  # empty
    (
        [
            [(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)],
            [(2.5, 5), (2.5, -2.5)],
        ],
        [0, 0, 0],
        [5, 0, 0],
    ),  # one wall inbetween
    (
        [
            [(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)],
            [(1.5, -1.5), (1.5, 1.5)],
            [(2, 0), (5, 0)],
        ],
        [0, 0, 0],
        [5, 0, 0],
    ),  # T shape with hidden wall
    (
        [
            [(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)],
            [(-4, 0), (0, 4)],
            [(-3, 5), (3, 5)],
            [(-7, 4), (-3, 3)],
        ],
        [0, 0, 0],
        [-4, 4, 0],
    ),  # raish shaped wall
    (
        [
            [(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)],
            [(0, 2), (2, -1), (-2, -1), (-4, 0), (-3, -4), (-4, -5)],
        ],
        [0, 0, 0],
        [-4, -4, 0],
    ),  # some walls in between the path
    (
        [
            [(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)],
            [(2, 1), (6, 2), (9, 1), (9, -1), (6, -1), (4, -0.5)],
        ],
        [0, 0, 0],
        [0, 8, 0],
    ),  # alot of walls around the endpoint that dont really disrupt the path
]

default_training_set = [
    (
        [
            [(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)],
            [(4, 0), (0, 2), (4, 4), (5, 4), (5, -4)],
        ],
        [0, 0, 0],
        [4, 2, 0],
    ),  # some walls around the path, rather easy
    (
        [
            [(35, 35), (35, -35), (-35, -35), (-35, 35), (34.5, 35)],
            [
                (6, -6),
                (6, -2),
                (2, -6),
                (0, -4),
                (0, 4),
                (-2, -2),
                (-2, 2),
                (8, 2),
                (8, -6),
            ],
        ],
        [0, 0, 0],
        [2, -4, 0],
    ),  # ben hagadol
]


def run_full_ses(population=10, epsiode_length=1, maze_index=0, number_of_breeds=1):
    torch.set_grad_enabled(False)
    model = NeuralNetwork
    trainer = Trainer(
        model,
        population,
        mutation_rate=1,
        episode_time_length=epsiode_length,
        breed_percent=0.5,
        training_set=[default_data_set[maze_index]],
    )  # change data
    t = time.localtime()
    start_time = time.strftime("%H:%M:%S", t)
    print("\nRUNNING NEW SIM \n =============================\n\n")
    print("starting time: ", start_time, "\n")

    for i in range(number_of_breeds):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(i, "th iteration, time: ", current_time)
        trainer.breed(i == number_of_breeds - 1)
    t = time.localtime()
    end_time = time.strftime("%H:%M:%S", t)
    print("starting time: ", start_time, "finished", end_time)
    plt.plot([i for i in range(len(consts.best_scores))], consts.best_scores)
    plt.show()
    plt.plot([i for i in range(len(consts.average_scores))], consts.average_scores)
    plt.show()
    return trainer


# shows the runs of the best cars for a given trainer
def get_run_res(trainer, episode_time, number_of_examples):
    plt.plot([1], [1])
    plt.show()  # use this to stop simulation

    consts.is_visual = True
    consts.print_reward_breakdown = True
    print("new run result:\n======================")
    for i in range(number_of_examples):
        print(
            i,
            ":",
            calculate_score(
                trainer.population[i],
                episode_time,
                trainer.training_set,
            ),
            "\n",
        )


def main():
    trainer1 = run_full_ses(
        population=20, epsiode_length=1500, maze_index=2, number_of_breeds=5
    )
    get_run_res(trainer1, trainer1.episode_time_length, 4)


if __name__ == "__main__":
    main()
