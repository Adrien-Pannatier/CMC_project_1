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
from exercise_p4 import (
    exercise_4a_transition_walk2swim, 
    exercise_4b_transition_swim2walk,
)
from exercise_p5 import (
    exercise_5a_swim_turn,
    exercise_5b_swim_back,
    exercise_5c_walk_turn,
    exercise_5d_walk_back,
)
from exercise_p6 import (
    exercise_6a_phase_relation,
    exercise_6b_tegotae_limbs,
    exercise_6c_tegotae_spine,
    exercise_6d_open_vs_closed,
)
from exercise_p7 import (
    exercise_7a_land_to_water,
    exercise_7b_water_to_land,
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
        exercise_4a_transition_walk2swim(timestep)
    if '4b' in arguments:
        exercise_4b_transition_swim2walk(timestep)
    if '5a' in arguments:
        exercise_5a_swim_turn(timestep)
    if '5b' in arguments:
        exercise_5b_swim_back(timestep)
    if '5c' in arguments:
        exercise_5c_walk_turn(timestep)
    if '5d' in arguments:
        exercise_5d_walk_back(timestep)
    if '6a' in arguments:
        exercise_6a_phase_relation(timestep)
    if '6b' in arguments:
        exercise_6b_tegotae_limbs(timestep)
    if '6c' in arguments:
        exercise_6c_tegotae_spine(timestep)
    if '6d' in arguments:
        exercise_6d_open_vs_closed(timestep)
    if '7a' in arguments:
        exercise_7a_land_to_water(timestep)
    if '7b' in arguments:
        exercise_7b_water_to_land(timestep)

    if not verbose:
        pylog.set_level('debug')


if __name__ == '__main__':
    # exercises = ['1a', '2a', '2b', '3a', '3b', '4a', '4b', '5a', '5b', '5c', '5d', '6a', '6b', '6c', '6d', '7a', '7b']
    exercises = ['6a', '6b', '6c', '6d', '7a', '7b']
    exercise_all(arguments=exercises)

