"""Plot results"""

import pickle
import numpy as np
from requests import head
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from network import motor_output
import matplotlib.colors as colors


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=['x', 'y', 'z'][i])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid(True)


def plot_trajectory(link_data, label=None, color=None):
    """Plot trajectory"""
    plt.plot(link_data[:, 0], link_data[:, 1], label=label, color=color)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)

def plot_2d(results, labels, n_data=300, title='', log=False, cmap='nipy_spectral'):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], 'r.')
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='none',
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])
    plt.title(title)


def max_distance(link_data, nsteps_considered=None):
    """Compute max distance"""
    if not nsteps_considered:
        nsteps_considered = link_data.shape[0]
    com = np.mean(link_data[-nsteps_considered:], axis=1)

    # return link_data[-1, :]-link_data[0, 0]
    return np.sqrt(
        np.max(np.sum((link_data[:, :]-link_data[0, :])**2, axis=1)))


def compute_speed(links_positions, links_vel, nsteps_considered=200):
    '''
    Computes the axial and lateral speed based on the PCA of the links positions
    '''

    links_pos_xy = links_positions[-nsteps_considered:, :, :2]
    joints_vel_xy = links_vel[-nsteps_considered:, :, :2]
    time_idx = links_pos_xy.shape[0]

    speed_forward = []
    speed_lateral = []
    com_pos = []

    for idx in range(time_idx):
        x = links_pos_xy[idx, :9, 0]
        y = links_pos_xy[idx, :9, 1]
        
        pheadtail = np.float64(links_pos_xy[idx][0])-np.float64(links_pos_xy[idx][8])  # head - tail
        pcom_xy = np.mean(links_pos_xy[idx, :9, :], axis=0)
        vcom_xy = np.mean(joints_vel_xy[idx], axis=0)

        covmat = np.cov([x, y])
        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index = np.argmax(eig_values)
        largest_eig_vec = eig_vecs[:, largest_index]

        ht_direction = np.sign(np.dot(pheadtail, largest_eig_vec))
        largest_eig_vec = ht_direction * largest_eig_vec

        v_com_forward_proj = np.dot(vcom_xy, largest_eig_vec)

        left_pointing_vec = np.cross(
            [0, 0, 1],
            [largest_eig_vec[0], largest_eig_vec[1], 0]
        )[:2]

        v_com_lateral_proj = np.dot(vcom_xy, left_pointing_vec)

        com_pos.append(pcom_xy)
        speed_forward.append(v_com_forward_proj)
        speed_lateral.append(v_com_lateral_proj)

    return np.mean(speed_forward), np.mean(speed_lateral)


def sum_torques(joints_data):
    """Compute sum of torques"""
    return np.sum(np.abs(joints_data[:, :]))

def plot_ex_2a(num_it):
    speed_fw = np.zeros(num_it)
    cost_of_transport = np.zeros(num_it) # à calculer
    drive = np.zeros(num_it)
    phase_lag_body = np.zeros(num_it)
    for sim_num in range(num_it):
        filename = './logs/ex_2a/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        with open(filename.format(sim_num, 'pickle'), 'rb') as param_file:
            parameters = pickle.load(param_file)
        drive[sim_num] = parameters.drive 
        phase_lag_body[sim_num] = parameters.downward_body_CPG_phi
        links_positions = data.sensors.links.urdf_positions()
        links_vel = data.sensors.links.com_lin_velocities()

        speed_fw[sim_num], speed_lat = compute_speed(links_positions, links_vel, num_it)

    # Plot data
    print(drive)
    print(phase_lag_body)
    print(speed_fw)
    results = np.array([drive, phase_lag_body, speed_fw]).T
    plot_2d(results, ['drive', 'downward body phase lag [rad]', 'forward speed [m/s]'], num_it)


def plot_ex_2b(num_it):
    speed_fw = np.zeros(num_it)
    cost_of_transport = np.zeros(num_it) # à calculer
    drive = np.zeros(num_it)
    phase_lag_body = np.zeros(num_it)
    for sim_num in range(num_it):
        filename = './logs/ex_2b/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        with open(filename.format(sim_num, 'pickle'), 'rb') as param_file:
            parameters = pickle.load(param_file)
        drive[sim_num] = parameters.drive 
        phase_lag_body[sim_num] = parameters.downward_body_CPG_phi
        links_positions = data.sensors.links.urdf_positions()
        links_vel = data.sensors.links.com_lin_velocities()

        speed_fw[sim_num], speed_lat = compute_speed(links_positions, links_vel, num_it)

    # Plot data
    print(drive)
    print(phase_lag_body)
    print(speed_fw)
    results = np.array([drive, phase_lag_body, speed_fw]).T
    plot_2d(results, ['drive', 'downward body phase lag [rad]', 'forward speed [m/s]'], num_it)


def plot_ex_3a(num_it=25):
    speed_fw = np.zeros(num_it)
    drive = np.zeros(num_it)
    phase_limb_body = np.zeros(num_it)
    for sim_num in range(num_it):
        filename = './logs/ex_3a/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        with open(filename.format(sim_num, 'pickle'), 'rb') as param_file:     
            parameters = pickle.load(param_file)
        drive[sim_num] = parameters.drive 
        phase_limb_body[sim_num] = parameters.limb_to_body_CPG_phi
        links_positions = data.sensors.links.urdf_positions()
        links_vel = data.sensors.links.com_lin_velocities()
        speed_fw[sim_num], speed_lat = compute_speed(links_positions, links_vel)

    # Plot data
    print(drive)
    print(phase_limb_body)
    print(speed_fw)
    results = np.array([drive, phase_limb_body, speed_fw]).T
    plt.figure("Phase_limbtobody_drive_to_speed")
    plot_2d(results, ['drive', 'limb to body phase', 'speed_fw'], num_it)

def plot_ex_3b(num_it=25):
    speed_fw = np.zeros(num_it)
    drive = np.zeros(num_it)
    amplitudes = np.zeros(num_it)
    for sim_num in range(num_it):
        filename = './logs/ex_3b/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        with open(filename.format(sim_num, 'pickle'), 'rb') as param_file:     
            parameters = pickle.load(param_file)
        # timestep = data.timestep
        # n_iterations = np.shape(data.sensors.links.array)[0]
        # times = np.arange(
        #     start=0,
        #     stop=timestep*n_iterations,
        #     step=timestep,
        # )
        # timestep = times[1] - times[0]
        drive[sim_num] = parameters.drive 
        # amplitudes[sim_num] = parameters.amplitude_factor*(parameters.bcR1 * parameters.drive + parameters.bcR0)
        amplitudes[sim_num] = parameters.amplitude_factor
        links_positions = data.sensors.links.urdf_positions()
        links_vel = data.sensors.links.com_lin_velocities()
        speed_fw[sim_num], speed_lat = compute_speed(links_positions, links_vel)

    results = np.array([drive, amplitudes, speed_fw]).T
    plt.figure("Nom_amplitude_drive_to_speed")
    plot_2d(results, ['drive', 'nominal amplitude', 'speed_fw'], title='Effects of drive and phase lag on speed')
    
def main(plot=True):
    """Main"""
    # plot_ex_2a(num_it=100)
    # plot_ex_2b(num_it=100)
    # plot_ex_3a(num_it=100)
    plot_ex_3b(num_it=100)
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=save_plots())

