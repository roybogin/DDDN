import PRM

'''
Maze Format: json
- "walls": List of walls
- "positions": List of dictionaries of the form (start, end, rotation) - of course list length is the amount of cars
- "title": title of the maze
'''

random_start_end = [
    ((0, 0), (1, 1)),
    ((0, 0), (-3, 2)),
    ((0, 0), (-2, 2)),
    ((1, 0), (-3, -5)),
    ((0, 2), (3, 1)),
    ((7, 2.1), (1, 7)),
    ((4, 2), (-9, 9)),
    ((0, 0), (1, -1)),
    ((0, -8), (2, 7)),
    ((0, 4), (-7.8, 2)),
    ((-1, 4.1), (3, -5)),
    ((1.2, 2), (3, -2)),
    ((7, -4.1), (1, 2)),
    ((4, 5), (-9, 7.2)),
    ((9, 9), (-1, 7.2)),
    ((9, -9), (-3, 2)),
    ((-9, -9), (0, 4)),
    ((-9, 9), (-6, 1)),
]

empty_set = [
    {
        'walls': [],
        'positions': [{'start': [start[0], start[1], 0], 'end': [end[0], end[1], 0], 'rotation': 0}],
        'title': f'empty maze {index}'
    }
    for index, (start, end) in enumerate(random_start_end)
]

default_data_set = [
    {
        'walls': [],
        'positions': [
                        {'start': [0, 0, 0], 'end': [2, 0, 0], 'rotation': 0},
                        {'start': [0, 5, 0], 'end': [2, 5, 0], 'rotation': 0},
                      ],
        'title': 'empty test maze'
    },
    {
        'walls': [[(2.5, 1), (2.5, -1)]],
        'positions': [{'start': [-2, 0, 0], 'end': [7, 2, 0], 'rotation': 0}],
        'title': 'maze with small wall'
    },
    {
        'walls': [[(2.5, 5), (2.5, -2.5)]],
        'positions': [{'start': [0, 0, 0], 'end': [5, 0, 0], 'rotation': 0}],
        'title': 'maze with big wall'
    },
    {
        'walls': [[(1.5, -1.5), (1.5, 1.5)], [(2, 0), (5, 0)]],
        'positions': [{'start': [0, 0, 0], 'end': [5, 0, 0], 'rotation': 0}],
        'title': 'T shape wall'
    },
    {
        'walls': [[(-4, 0), (0, 4)], [(-3, 5), (3, 5)], [(-7, 4), (-3, 3)]],
        'positions': [{'start': [0, 0, 0], 'end': [-4, 4, 0], 'rotation': 0}],
        'title': 'raish shaped wall'
    },
    {
        'walls': [[(0, 2), (2, -1), (-2, -1), (-4, 0), (-3, -4), (-4, -5)]],
        'positions': [{'start': [0, 0, 0], 'end': [-4, 4, 0], 'rotation': 0}],
        'title': 'walls in the path'
    },
    {
        'walls': [[(2, 1), (6, 2), (9, 1), (9, -1), (6, -1), (4, -0.5)]],
        'positions': [{'start': [0, 0, 0], 'end': [0, 8, 0], 'rotation': 0}],
        'title': 'walls around the end'
    }
]

default_training_set = [

    {
        'walls': [[(4, 0), (0, 2), (4, 4), (7, 4), (7, -4)]],
        'positions': [{'start': [0, 0, 0], 'end': [4, 2, 0], 'rotation': 0}],
        'title': 'walls in the path'
    },

    {
        'walls': [[(6, -6), (6, -2), (2, -6), (0, -4), (0, 4), (-2, -2), (-2, 2), (8, 2), (8, -6)]],
        'positions': [{'start': [0, 0, 0], 'end': [2, -4, 0], 'rotation': 0}],
        'title': 'Ben Hagadol'
    }
]
