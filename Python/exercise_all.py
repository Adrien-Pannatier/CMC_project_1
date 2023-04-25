"""[Project1] Script to call all exercises"""

from farms_core import pylog
from exercise_example import exercise_example
from exercise_p1 import exercise_1a_networks
from exercise_p2 import (
    exercise_2a_swim,
    exercise_2b_walk,
)
from exercise_p3 import (
    exercise_3a_coordination,
    exercise_3b_coordination
)
from exercise_p4 import exercise_4a_transition
from exercise_p5 import (
    exercise_5a_swim_turn,
    exercise_5b_swim_back,
    exercise_5c_walk_turn,
    exercise_5d_walk_back,
)


def exercise_all(arguments):
    """Run all exercises"""

    verbose = 'not_verbose' not in arguments

    if not verbose:
        pylog.set_level('warning')

    # Timestep
    timestep = 1e-2
    if 'exercise_example' in arguments:
        exercise_example(timestep)
    if '1a' in arguments:
        exercise_1a_networks(plot=False, timestep=1e-2)  # don't show plot
    if '2a' in arguments:
        exercise_2a_swim(timestep)
    if '2b' in arguments:
        exercise_2b_walk(timestep)
    if '3a' in arguments:
        exercise_3a_coordination(timestep)
    if '3b' in arguments:
        exercise_3b_coordination(timestep)
    if '4a' in arguments:
        exercise_4a_transition(timestep)
    if '5a' in arguments:
        exercise_5a_swim_turn(timestep)
    if '5b' in arguments:
        exercise_5b_swim_back(timestep)
    if '5c' in arguments:
        exercise_5c_walk_turn(timestep)
    if '5d' in arguments:
        exercise_5d_walk_back(timestep)

    if not verbose:
        pylog.set_level('debug')


if __name__ == '__main__':
    exercises = ['1a', '2a', '2b', '3a', '3b', '4a', '5a', '5b', '5c', '5d']
    exercise_all(arguments=exercises)

