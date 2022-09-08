# DDDN - Dual Driving &amp; Discovery Networks

The project is trying to solve the following problem:
given a 2d maze, and n-cars with given starting positions and an end goal - get all the cars into their corresponding goals.
we are trying to do so with minimal time, and with 0 collisions.

### Our Assumptions:
- the cars don't know the enviroment beforehand - they scan the enviroment while trying to get to the goal, and deal with obstacles in an online manner.
- the cars share their knowledge  - when a car discoveres a new obstacle, the other car immidiatly gets aware of it and plans ahead acorrdingly.
- at each step the cars are aware of their axact position in the enviroment.



## the Algorithm at action:
https://user-images.githubusercontent.com/72616264/189162466-37b7d72b-6a28-407d-9da1-ad8d920221a9.mp4
