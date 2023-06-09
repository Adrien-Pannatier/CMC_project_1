"""[Project1] Exercise 7: Transition ground to water with force feedback"""

import os
import pickle
import numpy as np
import matplotlib.animation as manimation
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog

def exercise_7a_land_to_water(timestep):
    """Exercise 7a - Transition land to water with force feedback
    """
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0,0,0.1],
            spawn_orientation=[0,0,np.pi/2],
            state = 'ground',
            drive = 2.6,
            amplitude_factor = 1, # enables the spine undulation
            feed_back_fac = 1, # determines open or closed loop
            downward_body_CPG_w = 10,
            upward_body_CPG_w = 10,
            contralateral_body_CPG_w = 10,
            limb_to_body_CPG_w = 30,
            within_limb_CPG_w = 10
        )
    ]
    # Grid search
    os.makedirs('./logs/ex_7a/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_7a/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'land'
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=True,  # Record video
            record_path='ex_7/walk2swim',
            camera_id=1  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data

        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    return

def exercise_7b_water_to_land(timestep):
    """Exercise 7b - Transition water to land with force feedback
    """
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[6,0,0.1],
            spawn_orientation=[0,0,-np.pi/2],
            state = 'water',
            amplitude_factor = 1, # enables the spine undulation
            feed_back_fac = 1, # determines open or closed loop
            downward_body_CPG_w = 10,
            upward_body_CPG_w = 10,
            contralateral_body_CPG_w = 10,
            limb_to_body_CPG_w = 30,
            within_limb_CPG_w = 10
        )
    ]
    # Grid search
    os.makedirs('./logs/ex_7b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_7b/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'land'
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=True,  # Record video
            record_path='ex_7/swim2walk',
            camera_id=1  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data

        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    return

if __name__ == '__main__':
    exercise_7a_land_to_water(timestep=1e-2)
    exercise_7b_water_to_land(timestep=1e-2)   
