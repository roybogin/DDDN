

from math import sqrt


def perpendicularDistance(point, start_point, end_point):
    x0 = start_point[0]
    y0 = start_point[1]
    m = (y0-end_point[1])/(x0-end_point[0])
    n = y0 - m*x0
    d = (m*point[0]-point[1]+n)/sqrt(m ** 2 + 1)
    return abs(d)




def points_to_map(points, epsilon):
    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(points)
    for i in range(1,end):
        d = perpendicularDistance( points[i], start_point = points[0], end_point = points[-1]) 
        if (len(points) == 6):
            print(d)
        if (d > dmax):
            index = i
            dmax = d
    result = set()

    # If max distance is greater than epsilon, recursively simplify
    if (dmax > epsilon):
        # Recursive call
        recResults1 = points_to_map(points[:index], epsilon)
        recResults2 = points_to_map(points[index:], epsilon)
    # Build the result list
        result = recResults1 | recResults2
    else:
        result = {points[0], points[-1]}
    
    # Return the result
    return result


print(points_to_map([(1,0),(2,0),(3,1),(4,0),(5,0),(6,0)],0.9))