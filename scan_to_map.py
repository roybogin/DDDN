from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
import consts

SAMPLE_DIST = 0.8
testing = True


class Map:
    # map is a list of segments, the obstacles
    def __init__(self, map=[], size=int(consts.size_map_quarter * 1.2), epsilon=0.1):
        # map represented as a list of polygonal chains, each chain is a list of consecutive vertices.
        self.map = map
        self.epsilon = epsilon
        self.size = size

        ##for testing:
        if testing:
            self.points = []
            self.distances = []
            self.new_segments = []
            self.number_of_segment = []

    def show(self):
        segments = self.segment_representation()
        # we can find the left,bottom point with two lines of length epsilon, one going from x1y1 in on the line, and one perpendicular to it.
        # should be a bit more accurate

        # define Matplotlib figure and axis
        fig, ax = plt.subplots()
        plt.axis([-self.size, self.size, -self.size, self.size])

        for segment in segments:
            x1 = segment[0][0]
            x2 = segment[1][0]
            y1 = segment[0][1]
            y2 = segment[1][1]
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            if x1 == x2:
                ax.add_patch(
                    Rectangle(
                        (x1 - self.epsilon, min(y1, y2) - self.epsilon),
                        2 * self.epsilon,
                        2 * self.epsilon + sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                    )
                )

            else:
                m = (y1 - y2) / (x1 - x2)
                theta = np.arctan(m)
                x = x1 - self.epsilon * sqrt(2) * np.cos(theta + np.pi / 4)
                y = y1 - self.epsilon * sqrt(2) * np.sin(theta + np.pi / 4)
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        2 * self.epsilon + sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                        2 * self.epsilon,
                        theta * 180 / np.pi,
                    )
                )
        # drawing the scan points:
        for segment in self.new_segments:
            for point in segment:
                ax.add_patch(Circle(point, 0.05, color="r"))

        # print("number of segments  = ", len(self.segment_representation()))
        self.number_of_segment.append(len(self.segment_representation()))
        plt.show()
        ax.cla()

        # plt.plot(
        #     [i + 1 for i in range(len(self.number_of_segment))], self.number_of_segment
        # )
        # print("number of segments after each addition = ", self.number_of_segment)
        # plt.show()
        # if testing:
        #     print("avrage distance =", sum(self.distances) / len(self.distances))
        #     plt.hist(self.distances)
        #     plt.show()

        return

    def check_batch(self, points):
        segment_representation = self.segment_representation()
        for point in points:
            for segment in segment_representation:
                if perpendicularDistance(point, segment[0], segment[1]) < self.epsilon:
                    return False
        return True
    def add_points_to_map(self, points):
        new_points = []
        if testing:
            self.points += points
            self.new_segments = []
        segment_representation = self.segment_representation()
        for point in points:
            should_add = True
            for segment in segment_representation:
                if perpendicularDistance(point, segment[0], segment[1]) < self.epsilon:
                    should_add = False
            if should_add:
                new_points.append(point)

        if len(segment_representation) == 0:
            new_points = points
        if len(new_points) == 0:
            return
        # dividing the points into segments (for when the samples come from 2 diffrent obstacles):
        segment_to_add = [new_points[0]]
        for i in range(len(new_points) - 1):
            if testing:
                self.distances.append(dist(new_points[i], new_points[i + 1]))
            if dist(new_points[i], new_points[i + 1]) > SAMPLE_DIST:
                new_segment = self.points_to_line(segment_to_add)
                self.new_segments.append(new_segment)
                self.add(new_segment)
                segment_to_add = [new_points[i + 1]]
            else:
                segment_to_add.append(new_points[i + 1])
        new_segment = self.points_to_line(segment_to_add)
        self.new_segments.append(new_segment)
        self.add(new_segment)

    def segment_representation(self):
        segments = []
        for i in range(len(self.map)):
            for j in range(1, len((self.map[i]))):
                segments.append((self.map[i][j - 1], self.map[i][j]))

        return segments

    def map_length(self):
        total_length = 0
        for segment in self.segment_representation():
            total_length += dist(segment[0], segment[1])
        return total_length

    def segment_representation_as_points(self):
        segments = ()
        for i in range(len(self.map)):
            for j in range(1, len((self.map[i]))):
                x1, y1 = self.map[i][j - 1]
                x2, y2 = self.map[i][j]
                segments.append((x1, y1, x2, y2))
        return segments

    def __str__(self):
        out = ""
        for i in range(len(self.map)):
            out += self.polygonal_chain_to_str(self.map[i])
            out += "\n"
        return out

    def polygonal_chain_to_str(self, list_of_points):
        string = "["
        for i in range(len(list_of_points)):
            string += (
                "(" + str(list_of_points[i][0]) + "," + str(list_of_points[i][1]) + ")"
            )
        string += "]"
        return string

    # given a polygonial chain, adds it to map
    # assumes points are given in order of the polygonal chain
    def add(self, chain):
        self.map.append(chain)

    # remove the chain from map, if it is in the map
    def remove(self, chain):
        self.map.remove(chain)

    def points_to_line(self, points):
        # Find the point with the maximum distance
        dmax = 0
        index = 0
        end = len(points)
        if end == 0:
            return []
        for i in range(1, end - 1):
            d = perpendicularDistance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d

        # If max distance is greater than epsilon, recursively simplify
        if dmax > self.epsilon:
            # Recursive call
            recResults1 = self.points_to_line(points[: index + 1])
            recResults2 = self.points_to_line(points[index:])
            # Build the result list
            result = recResults1 + recResults2[1:]
        else:
            result = [points[0], points[-1]]
        # Return the result
        return result


def euler_length(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dist(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def perpendicularDistance(point, start_point, end_point):
    x0, y0 = start_point
    x1, y1 = end_point
    x, y = point
    def_val = min(euler_length(point, start_point), euler_length(point, end_point))
    if x0 == x1:
        if min(y0, y1) <= y <= max(y0, y1):
            return abs(x - x0)
        return def_val
    if y0 == y1:
        if min(x0, x1) <= x <= max(x0, x1):
            return abs(y - y0)
        return def_val
    m = (y0 - y1) / (x0 - x1)
    inter_x = (y - y0 + m * x0 + x / m) / (m + 1 / m)
    inter_y = y0 + m * (inter_x - x0)
    if min(x0, x1) <= inter_x <= max(x0, x1):
        return sqrt((x - inter_x) ** 2 + (y - inter_y) ** 2)
    return def_val


if __name__ == "__main__":
    epsilon = 0.1
    new_map = Map([], 10, epsilon)

    new_map.add_points_to_map(
        [(0, 0), (0, 0.1), (0, 0.3), (0.1, 0.4), (1, 1), (0.9, 1.1), (0.95, 1)]
    )
    new_map.show()
