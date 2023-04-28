"""[Project1] Exercise 1: Implement & run network without MuJoCo"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork


def run_network(duration, update=True, drive=0, timestep=1e-2):
    """ Run network without MuJoCo and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        True: use the prescribed drive parameter, False: update the drive during the simulation
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    sim_parameters = SimulationParameters(
        drive = drive,
        amplitude_gradient=None,
        phase_lag_body=None,
        turn=None,
    )
    #pylog.warning(
    #    'Modify the scalar drive to be a vector of length n_iterations. By doing so the drive will be modified to be drive[i] at each time step i.')
    drive_array = np.linspace(0, 6, n_iterations) # constructs an array of n_iterations size
    state = SalamandraState.salamandra_robot(n_iterations)
    network = SalamandraNetwork(sim_parameters, n_iterations, state)
    osc_left = np.arange(8)
    osc_right = np.arange(8, 16)
    osc_legs = np.arange(16, 20)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output(iteration=0))
    ])
    outputs_log[0, :] = network.get_motor_position_output(iteration=0)

    # comment below pass to run file
    # pylog.warning('Remove the pass to run your code!!')
    # pass

    pylog.warning(
        'Implement plots here, try to plot the various logged data to check the implementation')
    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(drive = drive_array[i])
                )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
    toc = time.time()

    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))

    # Implement plots of network results
    # pylog.warning('Implement plots')

    #joint output commands qi
    #print( np.shape(outputs_log[:, 1]))
    plt.figure()
    plt.subplot(411)
    for i in range(8):
        plt.plot(times,outputs_log[:, i]+i*np.pi/3)

    #limb output command q
    plt.subplot(412)
    for i in range(8):
        plt.plot(times,outputs_log[:, i+8]+i*np.pi/3)

    #frequencies
    plt.subplot(413)
    for i in range(20):
        plt.plot(drive_array, freqs_log[:, i])

    # drive d
    plt.subplot(414)
    plt.plot(times,drive_array)
    
    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    # duration = 5
    # times = np.arange(0, duration, timestep)
    # n_iterations = len(times)

    run_network(duration=10)
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
    return


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())

