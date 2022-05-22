from re import I
import torch
from RL import NeuralNetwork
from trainer import Trainer
from RL import calculate_score
import consts
import time
from matplotlib import pyplot as plt
import mazes


def run_full_ses(
    population=10,
    epsiode_length=1,
    mazes=mazes.default_data_set,
    number_of_breeds=1,
    cars_to_load=None,
):
    torch.set_grad_enabled(False)
    number_of_breeds += 1
    model = NeuralNetwork
    trainer = Trainer(
        model,
        population,
        mutation_rate=1,
        episode_time_length=epsiode_length,
        breed_percent=0.5,
        training_set=mazes,
    )  # change data
    if cars_to_load is not None:
        for idx, num in enumerate(cars_to_load):
            trainer.population[idx].load(num)
    t = time.localtime()
    start_time = time.strftime("%H:%M:%S", t)
    print("\nRUNNING NEW SIM \n =============================\n\n")
    print("starting time: ", start_time, "\n")
    # try:
    for i in range(number_of_breeds):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(i, "th iteration, time: ", current_time)
        trainer.breed(i == number_of_breeds - 1)
    # except Exception as e:
    # print("ERROR : " + str(e))
    # finally:
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
        population=50,
        epsiode_length=1000,
        mazes=mazes.empty_set,
        number_of_breeds=3,
        cars_to_load=["empty." + str(i) for i in range(40)],
    )
    get_run_res(trainer1, trainer1.episode_time_length, 2)
    for i in range(40):
        trainer1.population[i].save("empty." + str(i))


if __name__ == "__main__":
    main()
