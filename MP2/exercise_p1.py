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
    freqs_osc_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.der_phases)
    ])
    freqs_osc_log[0, :] = network.robot_parameters.der_phases/(2*np.pi)
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
    nom_amp_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.nominal_amplitudes)
    ])
    nom_amp_log[0, :] = network.robot_parameters.nominal_amplitudes

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
        freqs_osc_log[i+1, :] = network.robot_parameters.der_phases/(2*np.pi)
        nom_amp_log[i+1, :] = network.robot_parameters.nominal_amplitudes
    toc = time.time()

    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))

    f, axes = plt.subplots(4)
    for i in range(4):
        axes[0].plot(times,outputs_log[:, i]-i*np.pi/3, 'steelblue')
    for i in range(4, 8):
        axes[0].plot(times,outputs_log[:, i]-i*np.pi/3, 'mediumseagreen')
    axes[0].set_yticklabels([])
    axes[0].text(-1.0, -2.5, 'Trunk', rotation = "vertical", bbox=dict(facecolor='white', edgecolor='black'), fontsize=7)
    axes[0].text(-1.0, -6.3, 'Tail', rotation = "vertical", bbox=dict(facecolor='white', edgecolor='black'), fontsize=7)
    axes[0].annotate('', xy=(40, 0), xycoords='data', xytext=(40, -1), textcoords='data', arrowprops={'arrowstyle': '-'})
    axes[0].text(40.3, -0.5, 'π/3', rotation = "horizontal", fontsize=7)
    axes[0].set_xticklabels([])
    axes[0].set_ylabel("x Body")
    
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels([])
    axes[1].plot(times,outputs_log[:, 13], color='steelblue')
    axes[1].plot(times,outputs_log[:, 15]-np.pi/3, color='mediumseagreen')
    axes[1].annotate('', xy=(40, 0), xycoords='data', xytext=(40, -1), textcoords='data', arrowprops={'arrowstyle': '<|-|>'})
    axes[1].text(-1, 0, 'x18', rotation = "horizontal", fontsize=7)
    axes[1].text(-1, -1, 'x20', rotation = "horizontal", fontsize=7)
    axes[1].text(40.6, -0.5, 'π/3', rotation = "horizontal", fontsize=7)
    axes[1].set_ylabel("x Limb")
    # axes[1].legend(loc='upper left')

    #frequencies
    
    axes[2].plot(times,freqs_osc_log[:, 0], color='olivedrab', label='Spine frequencies')
    for i in range(1, 16):
        axes[2].plot(times,freqs_osc_log[:, i], color='olivedrab')
    axes[2].plot(times,freqs_osc_log[:, 16], color='chocolate', label='Limb frequencies')
    for i in range(17, 20):
        axes[2].plot(times,freqs_osc_log[:, i], color='chocolate')
    
    axes[2].set_xticklabels([])
    axes[2].set_ylabel("Freq [Hz]")
    axes[2].set_ylim([-0.05, 1.5])
    axes[2].legend(loc='upper left')

    # drive d
    # plt.subplot(414)
    walk_switch = np.ones(len(times)) * 1
    swim_switch = np.ones(len(times)) * 3
    end_switch = np.ones(len(times)) * 5
    axes[3].plot(times,drive_array, 'k-')
    axes[3].plot(times,walk_switch,linestyle = 'solid', color='rosybrown')
    axes[3].plot(times,swim_switch,linestyle = 'solid', color='rosybrown')
    axes[3].plot(times,end_switch,linestyle = 'solid', color='rosybrown')
    axes[3].set_ylabel("Drive")
    axes[3].set_xlabel("Time [s]")
    axes[3].text(2.73, 1.93, "Walking", rotation = "horizontal")
    axes[3].text(35.7, 3.85, "Swimming", rotation = "horizontal")
    
    # f.legend()

    f, axes = plt.subplots(2)

    #frequencies
    # plt.subplot(211)
    for i in range(16):
        axes[0].plot(drive_array,freqs_log[:, i], 'k-')
    for i in range(16, 20):
        axes[0].plot(drive_array,freqs_log[:, i], linestyle= 'dotted', color = 'black')
    axes[0].set_ylabel("v [Hz]")
    
    # plt.subplot(212)
    for i in range(16):
        axes[1].plot(drive_array,nom_amp_log[:, i], 'k-')
    for i in range(16, 20):
        axes[1].plot(drive_array,nom_amp_log[:, i], linestyle= 'dotted', color = 'black')
    axes[1].set_ylabel("R")
    axes[1].set_xlabel("Drive")

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    run_network(duration=40)
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
    return


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())

