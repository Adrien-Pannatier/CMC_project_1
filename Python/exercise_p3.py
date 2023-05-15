"""[Project1] Exercise 3: Limb and Spine Coordination while walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog


def exercise_3a_coordination(timestep):
    """[Project 1] Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and leg oscillators.

    """
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=drive,
            limb_to_body_CPG_phi = limb_to_body_CPG_phi
        )
        # drive needed to be in walking mode
        for drive in np.linspace(1, 3, 10) 

        for limb_to_body_CPG_phi in np.linspace(-np.pi, np.pi, 10)
    ]

    # Grid search
    os.makedirs('./logs/ex_3a/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_3a/simulation_{}.{}'
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


def exercise_3b_coordination(timestep):
    """[Project 1] Exercise 3b Limb and Spine coordination

    This exercise explores how spine amplitude affects coordination.

    Run the simulations for different walking drives and body amplitude.

    """
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=drive,
            amplitude_factor=amplitude_factor # factor multiplying the affine equation of the nominal amplitude for the body
        )
        # drive needed to be in walking mode
        for drive in np.linspace(1, 3, 10) 

        for amplitude_factor in np.linspace(0, 3, 10)
    ]

    # Grid search
    os.makedirs('./logs/ex_3b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_3b/simulation_{}.{}'
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


if __name__ == '__main__':
    # exercise_3a_coordination(timestep=1e-2)
    exercise_3b_coordination(timestep=1e-2)

