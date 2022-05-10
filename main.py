import torch
from RL import NeuralNetwork
from trainer import Trainer

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
    population = 10  # Total Population
    trainer = Trainer(
        model,
        EPOCHS,
        population,
        mutation_rate=1,
        episode_time_length=100,
        breed_percent=0.5,
        training_set=[(([], [0, 0, 0], [1, 1, 0]))],
    )  # change data

    trainer.breed()


if __name__ == "__main__":
    main()
