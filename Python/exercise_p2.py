"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog

"""""""""""""""""""""""""""""Questions"""""""""""""""""""""""""""""""""""""""
# not sure about what are "amplitudes"

def exercise_2a_swim(timestep):
    """[Project 1] Exercise 2a Swimming

    In this exercise we need to implement swimming for salamander robot.
    Check exericse_example.py to see how to setup simulations.

    Run the simulations for different swimming drives and phase lag between body
    oscillators.
    """
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=drive,  # An example of parameter part of the grid search
            amplitudes=1,  
            phase_lag_body=phase_lag_body, 
            turn=0
        )
        # drive needed to be in swimming mode
        for drive in np.linspace(3, 5, 5) 

        # phase lag implemented as in Fig.6 of "Salamandra Robotica II: An Amphibious Robot to Study Salamander-Like Swimming and Walking Gaits"
        for phase_lag_body in np.linspace(-2*180/np.pi, 2*180/np.pi, 5) 
    ]

    # Grid search
    os.makedirs('./logs/ex_2a/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_2a/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'land'
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video saving path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    return


def exercise_2b_walk(timestep):
    """[Project 1] Exercise 2a Walking

    In this exercise we need to implement walking for salamander robot.
    Check exercise_example.py to see how to setup simulations.

    Run the simulations for different walking drives and phase lag between body
    oscillators.
    """
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=drive,  # An example of parameter part of the grid search
            amplitudes=1,  
            phase_lag_body=phase_lag_body, 
            turn=0
        )
        # drive needed to be in walking mode
        for drive in np.linspace(1, 3, 5) 

        # phase lag implemented as in Fig.6 of "Salamandra Robotica II: An Amphibious Robot to Study Salamander-Like Swimming and Walking Gaits"
        for phase_lag_body in np.linspace(-2*180/np.pi, 2*180/np.pi, 5) # []
    ]

    # Grid search
    os.makedirs('./logs/ex_2b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_2b/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'land'
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video saving path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    
    return


def exercise_test_walk(timestep):
    "[Project 1] Q2 Swimming"
    # Use exercise_example.py for reference
    pass
    return


def exercise_test_swim(timestep):
    "[Project 1] Q2 Swimming"
    # Use exercise_example.py for reference
    pass
    return


if __name__ == '__main__':
    exercise_2a_swim(timestep=1e-2)
    exercise_2b_walk(timestep=1e-2)