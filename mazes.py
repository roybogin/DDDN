import consts

random_start_end = [
    ((0, 0), (1, 1)),
    ((0, 0), (-3, 2)),
    ((0, 0), (-2, 2)),
    ((1, 0), (-3, -5)),
    ((0, 2), (3, 1)),
    ((7, 2.1), (1, 7)),
    ((4, 2), (-9, 9)),
    ((0, 0), (1, 1)),
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
    (
        [consts.map_borders],
        [start[0], start[1], 0],
        [end[0], end[1], 0],
    )
    for start, end in random_start_end
]

default_data_set = [
    (
        [consts.map_borders],
        [0, 0, 0],
        [1, 1, 0],
    ),  # empty
    (
        [
            consts.map_borders,
            [(2.5, 5), (2.5, -2.5)],
        ],
        [0, 0, 0],
        [5, 0, 0],
    ),  # one wall inbetween
    (
        [
            consts.map_borders,
            [(1.5, -1.5), (1.5, 1.5)],
            [(2, 0), (5, 0)],
        ],
        [0, 0, 0],
        [5, 0, 0],
    ),  # T shape with hidden wall
    (
        [
            consts.map_borders,
            [(-4, 0), (0, 4)],
            [(-3, 5), (3, 5)],
            [(-7, 4), (-3, 3)],
        ],
        [0, 0, 0],
        [-4, 4, 0],
    ),  # raish shaped wall
    (
        [
            consts.map_borders,
            [(0, 2), (2, -1), (-2, -1), (-4, 0), (-3, -4), (-4, -5)],
        ],
        [0, 0, 0],
        [-4, -4, 0],
    ),  # some walls in between the path
    (
        [
            consts.map_borders,
            [(2, 1), (6, 2), (9, 1), (9, -1), (6, -1), (4, -0.5)],
        ],
        [0, 0, 0],
        [0, 8, 0],
    ),  # alot of walls around the endpoint that dont really disrupt the path
]

default_training_set = [
    (
        [
            consts.map_borders,
            [(4, 0), (0, 2), (4, 4), (5, 4), (5, -4)],
        ],
        [0, 0, 0],
        [4, 2, 0],
    ),  # some walls around the path, rather easy
    (
        [
            consts.map_borders,
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