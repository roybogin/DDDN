from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
from helper import perpendicularDistance
import consts
from PRM import Vertex

SAMPLE_DIST = 0.8
testing = True


class Map:
    # map is a list of segments, the obstacles
    def __init__(self, map=[], size=int(consts.size_map_quarter * 1.2)):
        # map represented as a list of polygonal chains, each chain is a list of consecutive vertices.
        self.map = map
        self.size = size

        ##for testing:
        if testing:
            self.points = []
            self.distances = []
            self.new_segments = []
            self.number_of_segment = []

    def plot(self, ax):
        """
        a function to draw the current map,
        used for debuging and demos
        """
        segments = self.segment_representation()
        # we can find the left,bottom point with two lines of length epsilon,
        # one going from x1y1 in on the line, and one perpendicular to it.
        # should be a bit more accurate


        # add a rectangle for each segment:
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
                        (x1 - consts.epsilon, min(y1, y2) - consts.epsilon),
                        2 * consts.epsilon,
                        2 * consts.epsilon + sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                        color='red'
                    )
                )

            else:
                m = (y1 - y2) / (x1 - x2)
                theta = np.arctan(m)
                x = x1 - consts.epsilon * sqrt(2) * np.cos(theta + np.pi / 4)
                y = y1 - consts.epsilon * sqrt(2) * np.sin(theta + np.pi / 4)
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        2 * consts.epsilon + sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                        2 * consts.epsilon,
                        theta * 180 / np.pi,
                        color='red'
                    )
                )
        # drawing the end_points of each segment:
        for segment in self.new_segments:
            for point in segment:
                ax.add_patch(Circle(point, 0.05, color="r"))

        self.number_of_segment.append(len(self.segment_representation()))
        return

    def check_batch(self, points):
        """
        checks if the batch of points representing the car might colide with some segment in the map.

        :param points: the list of points which is the car.
        :return: True if none of the points are too close to an obstacle.
        """
        segment_representation = self.segment_representation()
        for point in points:
            for segment in segment_representation:
                if perpendicularDistance(point, segment[0], segment[1]) < consts.epsilon:   # TODO: enter on false?
                    return False
        return True

    def check_state(self, vertex: Vertex, num_sample_car=2):
        """
        the code used to check which vertecies we need to remove from the PRM graph.
        the car is split into evenly num_sample_car^2 points,
        and checks if any of them is too close to an obstacle on the map.

        :param vertex: the position vertex of the car.
        :param num_sample_car: the number of vertecies to represent the car's position.
        :return: True if the state is valid with the current map.
        """
        to_check = []
        for i in range(num_sample_car):
            for j in range(num_sample_car):
                x_temp = consts.length * (-1 / 2 + i / (num_sample_car - 1))
                y_temp = consts.width * (-1 / 2 + j / (num_sample_car - 1))
                to_check.append(
                    (vertex.pos[0] + x_temp * np.cos(vertex.theta) - y_temp * np.sin(vertex.theta),
                        vertex.pos[1] + x_temp * np.sin(vertex.theta) + y_temp * np.cos(vertex.theta))
                )
        return self.check_batch(to_check)

    def add_points_to_map(self, points):
        """
        given a list of scan points, add new segments to the map.
        the new segments are an approximation of the true obstacles.

        :param points: the points to add.
        """
        new_points = []
        if testing:
            self.points += points
            self.new_segments = []
        segment_representation = self.segment_representation()

        # filtering the points we don't need to add.
        for point in points:
            should_add = True
            for segment in segment_representation:
                if perpendicularDistance(point, segment[0], segment[1]) < consts.epsilon:
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
            # if the next point is too far away, start a new segment
            if dist(new_points[i], new_points[i + 1]) > SAMPLE_DIST:
                new_segment = self.points_to_line(segment_to_add)
                self.new_segments.append(segment_to_add)
                self.add(new_segment)
                segment_to_add = [new_points[i + 1]]
            else:
                segment_to_add.append(new_points[i + 1])
        new_segment = self.points_to_line(segment_to_add)
        self.new_segments.append(segment_to_add)
        self.add(new_segment)

    def segment_representation(self):
        """
        returns a list of all of the segments in the map.
        """
        segments = []
        for i in range(len(self.map)):
            for j in range(1, len((self.map[i]))):
                segments.append((self.map[i][j - 1], self.map[i][j]))

        return segments

    def map_length(self):
        """
        returns the cumulative length of the map
        #not_used
        """
        total_length = 0
        for segment in self.segment_representation():
            total_length += dist(segment[0], segment[1])
        return total_length

    def segment_representation_as_points(self):
        """
        should probably be deleted, not sure how this gives something better then segment_representation
        #not_used
        """
        segments = ()
        for i in range(len(self.map)):
            for j in range(1, len((self.map[i]))):
                x1, y1 = self.map[i][j - 1]
                x2, y2 = self.map[i][j]
                segments.append((x1, y1, x2, y2))
        return segments

    def __str__(self):
        """
        :return: a string representation of the map
        """
        out = ""
        for i in range(len(self.map)):
            out += self.polygonal_chain_to_str(self.map[i])
            out += "\n"
        return out

    def polygonal_chain_to_str(self, list_of_points):
        """
        a helper function to __str__
        :param list_of_points: list of points to get the string representation of
        :return: string representation of the list 
        """
        string = "["
        for i in range(len(list_of_points)):
            string += (
                "(" + str(list_of_points[i][0]) + "," + str(list_of_points[i][1]) + ")"
            )
        string += "]"
        return string

    #
    def add(self, chain):
        """
        given a polygonal chain, adds it to map
        assumes points are given in order of the polygonal chain
        """
        self.map.append(chain)

    def remove(self, chain):
        """
        remove the chain from map, if it is in the map
        #not_used
        """
        self.map.remove(chain)

    def points_to_line(self, points):
        """
        an implementation of the Ramer–Douglas–Peucker algorithm
        to make a set of points into a polygonal chain.
        """
        # Find the point with the maximum distance from the line from the first and last points.
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
        if dmax > consts.epsilon:
            # Recursive call
            recResults1 = self.points_to_line(points[: index + 1])
            recResults2 = self.points_to_line(points[index:])
            # Build the result list
            result = recResults1 + recResults2[1:]
        else:
            result = [points[0], points[-1]]
        # Return the result
        return result


def euler_dist(point1, point2):
    """
    :return: the l2 distance between point1 and point2
    #not_used
    """
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dist(point1, point2):
    """
    :return: the l2 distance between point1 and point2
    """
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


if __name__ == "__main__":
    new_map = Map([], 10)

    new_map.add_points_to_map(
        [(0, 0), (0, 0.1), (0, 0.3), (0.1, 0.4), (1, 1), (0.9, 1.1), (0.95, 1)]
    )
    new_map.plot()
