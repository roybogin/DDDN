from math import sqrt


def perpendicularDistance(point, start_point, end_point):
    x0, y0 = start_point
    x1, y1 = end_point
    x, y = point
    if x0 == x1:
        if min(y0, y1) <= y <= max(y0,y1):
            return abs(x-x0)
        return None
    if y0 == y1:
        if min(x0, x1) <= x <= max(x0,x1):
            return abs(y-y0)
        return None
    m = (y0-y1)/(x0-x1)
    inter_x = (y-y0 + m*x0 + x / m)/(m+1/m)
    inter_y = y0 + m*(inter_x-x0)
    if min(x0,x1) <= inter_x <= max(x0,x1):
        return sqrt((x-inter_x)**2+(y-inter_y)**2)
    return None



def points_to_map(points, epsilon):
    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(points)
    for i in range(1,end):
        d = perpendicularDistance(points[i], points[0], points[-1]) 
        if (d > dmax):
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if (dmax > epsilon):
        # Recursive call
        recResults1 = points_to_map(points[:index+1], epsilon)
        recResults2 = points_to_map(points[index:], epsilon)
    # Build the result list
        result = recResults1 | recResults2
    else:
        result = {points[0], points[-1]}
    
    # Return the result
    return result


print(points_to_map([(1,0),(2,0),(3,1),(4,0),(5,0),(6,0)],1))