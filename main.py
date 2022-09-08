import argparse
import json
import time
from typing import Dict
from scan_to_map import Map
import pybullet as p
from matplotlib import pyplot as plt

import consts
from env import Env


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
    parser.add_argument(
        "-m",
        "--plot_maze",
        action="store_true",
        help="draw the maze map with a matplotlib plot",
    )
    parser.add_argument(
        "-t",
        "--max_time",
        type=int,
        help="the maximum number of ticks before the simulation is stopped.",
    )

    args = parser.parse_args()

    with open(args.maze) as f:
        maze = json.load(f)
        f.close()

    if args.max_time:
        max_time = int(args.max_time)
        consts.max_time = max_time
    else:
        max_time = consts.max_time

    if not is_input_valid(maze, max_time):
        exit(1)

    if args.plot_maze:
        size_map_quarter = maze["size"] / 2

        map_borders = [[
            (size_map_quarter, size_map_quarter),
            (size_map_quarter, -size_map_quarter),
            (-size_map_quarter, -size_map_quarter),
            (-size_map_quarter, size_map_quarter),
            (size_map_quarter, size_map_quarter),
        ]]
        plot_map = Map(maze["walls"] + map_borders, maze["size"])
        ax = plt.gca()
        plot_map.plot(ax)
        plt.show()
        return

    consts.debugging = args.print
    consts.drawing = args.draw
    consts.is_visual = args.visualize

    run_sim(maze)


def is_input_valid(maze: dict, max_time: int) -> bool:
    """
    Check if the input maze can be contained in a square with side length maze["size"].
    :return: True if the input maze can be contained, False otherwise
    """
    size = maze["size"]
    for poly_chain in maze["walls"]:
        for segment in poly_chain:
            for point in segment:
                if abs(point) >= size / 2:
                    print(
                        "invalid input, you might want to increase the size of the maze size."
                    )
                    return False
    for position in maze["positions"]:
        for i in range(2):
            if (
                abs(position["start"][i]) >= size / 2
                or abs(position["end"][i]) >= size / 2
            ):
                print(
                    "invalid input, you might want to increase the size of the maze size."
                )
                return False
        if position["start"][2] != 0 or position["end"][2] != 0:
            print("invalid input, start and endpoints must be with z value 0.")
            return False
    if max_time <= 0:
        print("max time should be a positive integer!")
        return False

    return True


def run_sim(maze: Dict) -> None:
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
            curr_trace = env.traces[idx]
            plt.plot(
                [a for a, _ in curr_trace],
                [a for _, a in curr_trace],
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
