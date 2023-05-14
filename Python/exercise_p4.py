"""[Project1] Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import SalamandraState
import farms_pylog as pylog

WALK = 2.9
SWIM = 4.9

def on_ground():

    return 0

def exercise_4a_transition(timestep):
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
            duration=60,
            drive = 4.9,
            spawn_position=[4,0,0.1],
            spawn_orientation=[0,0,-np.pi/2],
            amplitudes=1,  
            phase_lag_body=0, 
            state = 'water'
        )]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path='walk2swim',
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
    exercise_4a_transition(timestep=1e-2)

