"""[Project1] Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import SalamandraState
import farms_pylog as pylog

WALK = 2.6
WALK_BODY_PHI = -2.51327
WALK_BODY_LIMB_PHI = 3
SWIM = 3.25
SWIM_PHI = -2.51327

def exercise_4a_transition_walk2swim(timestep):
    """[Project 1] 4a Transitions

    In this exerices, we will implement transitions.
    The salamander robot needs to perform swimming to walking
    and walking to swimming transitions.

    Hint:
        - set the  arena to 'amphibious'
        - use the sensor(gps) values to find the point where
        the robot should transition 2.7 approximately
        - simulation can be stopped/played in the middle
        by pressing the space bar
        - printing or debug mode of vscode can be used
        to understand the sensor values

    """
        # Parameters
    parameter_set = [
        SimulationParameters(
            timestep = timestep,
            duration=30,
            drive = WALK,
            downward_body_CPG_phi = WALK_BODY_PHI,
            limb_to_body_CPG_phi = WALK_BODY_LIMB_PHI,
            spawn_position=[0,0,0.1],
            spawn_orientation=[0,0,np.pi/2],
            state = 'ground'
        )]

    # Grid search
    os.makedirs('./logs/ex_4/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_4/simulation_w2s_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=True,  # Record video
            record_path='ex_4/walk2swim',
            # str(simulation_i),  # video saving path
            camera_id=1  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    return

def exercise_4b_transition_swim2walk(timestep):
    """[Project 1] 4a Transitions

    In this exerices, we will implement transitions.
    The salamander robot needs to perform swimming to walking
    and walking to swimming transitions.

    Hint:
        - set the  arena to 'amphibious'
        - use the sensor(gps) values to find the point where
        the robot should transition 2.7 approximately
        - simulation can be stopped/played in the middle
        by pressing the space bar
        - printing or debug mode of vscode can be used
        to understand the sensor values

    """
        # Parameters
    parameter_set = [
        SimulationParameters(
            timestep = timestep,
            duration=30,
            drive = SWIM,
            downward_body_CPG_phi = SWIM_PHI,
            spawn_position=[6,0,0.1],
            spawn_orientation=[0,0,-np.pi/2],
            state = 'water'
        )]

    # Grid search
    os.makedirs('./logs/ex_4/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_4/simulation_s2w_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=True,  # Record video
            record_path='ex_4/swim2walk',
            # str(simulation_i),  # video saving path
            camera_id=1  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    return


if __name__ == '__main__':
    exercise_4a_transition_walk2swim(timestep=1e-2)
    exercise_4b_transition_swim2walk(timestep=1e-2)

