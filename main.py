import sys
import argparse
import consts
import json
from typing import Dict
import time
from env import Env
from matplotlib import pyplot as plt
import pybullet as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "maze", help="a json file in the right format that describes the simulation."
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="print running simulation information",
    )
    parser.add_argument(
        "-d",
        "--draw",
        action="store_true",
        help="draw a matplotlib plot of the simulation",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="visualize the simulation using pybullet",
    )

    args = parser.parse_args()

    consts.debugging = args.print
    consts.drawing = args.draw
    consts.visualize = args.visualize

    with open(args.maze) as f:
        maze = json.load(f)
        f.close()
    run_sim(maze)


def run_sim(maze: Dict):
    t0 = time.time()
    stop = False
    env = Env(maze)
    while not stop:
        stop = env.step()
    print(f"total time: {time.time() - t0}")
    p.disconnect()
    if consts.drawing:
        env.segments_partial_map.plot(env.ax)
        for idx, car in enumerate(env.cars):
            plt.plot(
                [a for a, _ in car.trace],
                [a for _, a in car.trace],
                label=f"actual car {idx}",
            )
        plt.title(f'{maze["title"]} - time {env.run_time}')
        ax = env.ax
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()


if __name__ == "__main__":
    main()
