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
from scipy.signal import lfilter


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

def compute_cost_of_transport(data):
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()
    timestep = data.timestep
    # sum the velocities for each joint, for each timestep
    joints_velocities_summed = np.sum(np.abs(joints_velocities), axis=0)
    joints_torques_summed = np.sum(np.abs(joints_torques), axis=0)
    cost_of_transport = np.abs(np.dot(joints_velocities_summed,joints_torques_summed)*timestep)
    return cost_of_transport


# timestep = data.timestep
# n_iterations = np.shape(data.sensors.links.array)[0]
# times = np.arange(
#     start=0,
#     stop=timestep*n_iterations,
#     step=timestep,
# )
# timestep = times[1] - times[0]
# # amplitudes = parameters.amplitudes
#         amplitudes[sim_num] = parameters.amplitudes
#         # osc_phases = data.state.phases()
#         # osc_amplitudes = data.state.amplitudes()
#         # amplitudes[sim_num] = parameters.amplitude_factor*(parameters.bcR1 * parameters.drive + parameters.bcR0)
#         amplitudes[sim_num] = parameters.amplitude_factor
#         links_positions = data.sensors.links.urdf_positions()
#         links_vel = data.sensors.links.com_lin_velocities()
        # head_positions = links_positions[:, 0, :]
#         tail_positions = links_positions[:, 8, :]
#         # joints_positions = data.sensors.joints.positions_all()
#         joints_velocities = data.sensors.joints.velocities_all()
#         joints_torques = data.sensors.joints.motor_torques_all()
#         speed_fw[sim_num], speed_lat = compute_speed(links_positions, links_vel)
#         sum_of_torques[sim_num] = sum_torques(joints_torques)

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
        cost_of_transport[sim_num] = compute_cost_of_transport(data)

        speed_fw[sim_num], speed_lat = compute_speed(links_positions, links_vel, num_it)

    # Plot data
    print(drive)
    print(phase_lag_body)
    print(speed_fw)
    results = np.array([drive, phase_lag_body, speed_fw]).T
    plt.figure("phase_bias_and_drive_effect_to_speed_swimming")
    plot_2d(results, ['drive', 'downward body phase lag [rad]', 'forward speed [m/s]'], n_data=num_it, title='Effect of body phase lag and drive on swimming speed')
    plt.figure("phase_bias_and_drive_effect_to_transport_swimming")
    plot_2d(np.array([drive, phase_lag_body, cost_of_transport]).T, ['drive', 'downward body phase lag [rad]', 'cost of transport [J]'], n_data=num_it, title='Effect of body phase lag and drive on transport cost')



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
        cost_of_transport[sim_num] = compute_cost_of_transport(data)

        speed_fw[sim_num], speed_lat = compute_speed(links_positions, links_vel, num_it)

    # Plot data
    # print(drive)
    # print(phase_lag_body)
    # print(speed_fw)
    results = np.array([drive, phase_lag_body, speed_fw]).T
    plt.figure("phase_bias_and_drive_effect_to_speed_walking")
    plot_2d(results, ['drive', 'downward body phase lag [rad]', 'forward speed [m/s]'], n_data=num_it, title='Effect of body phase lag and drive on walking speed')

    plt.figure("phase_bias_and_drive_effect_to_transport_walking")
    plot_2d(np.array([drive, phase_lag_body, cost_of_transport]).T, ['drive', 'downward body phase lag [rad]', 'cost of transport [J]'], n_data=num_it, title='Effect of body phase lag and drive on transport cost')

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

        drive[sim_num] = parameters.drive 
        amplitudes[sim_num] = data.state.amplitudes()[500][10]
        links_positions = data.sensors.links.urdf_positions()
        links_vel = data.sensors.links.com_lin_velocities()
        speed_fw[sim_num], speed_lat = compute_speed(links_positions, links_vel)

    results = np.array([drive, amplitudes, speed_fw]).T
    plt.figure("Nom_amplitude_drive_to_speed")
    plot_2d(results, ['drive', 'amplitude factor', 'forward speed'], n_data=num_it, title='Effects of amplitude factor and drive on speed')

def plot_ex_4a():
    filename = './logs/ex_4/simulation_w2s_{}.{}'
    data = SalamandraData.from_file(filename.format('0', 'h5'))   
    phases = data.state.phases()
    limb_phases = phases[:,16:]
    body_phases = phases[:,0:8]
    links_positions = data.sensors.links.urdf_positions()
    body_x_positions = np.asarray(links_positions[:, 0:8, 0])
    head_positions = links_positions[:, 0, :]
    head_positions = np.asarray(head_positions)
    
    plt.figure('Phase_body_w2s')
    for i in range(8):

        # plt.plot(body_x_positions[:, i], body_phases[:, i])
        plt.plot(head_positions[:, 0],body_phases[:, i], label=f"spine {i}")
        plt.legend()
        plt.ylabel('Body phase [deg]')
        plt.xlabel('Distance [m]')
        # plt.xlim(2,4)
        # plt.ylim(25,100)
        plt.grid(True)

    plt.figure('Phase_limb_w2s')
    for i in range(4):

        # plt.plot(body_x_positions[:, i], body_phases[:, i])
        plt.plot(head_positions[:, 0],limb_phases[:, i], label=f"limb {i}")
        plt.legend()
        plt.ylabel('Limb phase [deg]')
        plt.xlabel('Distance [m]')
        # plt.xlim(2,4)
        # plt.ylim(25,100)
        plt.grid(True)

    plt.figure('Traj_w2s')
    plot_trajectory(head_positions)  

def plot_ex_4b():
    filename = './logs/ex_4/simulation_s2w_{}.{}'
    data = SalamandraData.from_file(filename.format('0', 'h5'))
    phases = data.state.phases()
    limb_phases = phases[:,16:]
    body_phases = phases[:,0:8]
    links_positions = data.sensors.links.urdf_positions()
    head_positions = links_positions[:, 0, :]
    head_positions = np.asarray(head_positions)

    plt.figure('Phase_body_s2w')
    for i in range(8):

        # plt.plot(body_x_positions[:, i], body_phases[:, i])
        plt.plot(head_positions[:, 0], body_phases[:, i], label=f"spine {i}")
        plt.legend()
        plt.ylabel('Body phase [deg]')
        plt.xlabel('Distance [m]')
        # plt.xlim(2,4)
        # plt.ylim(25,100)
        plt.grid(True)

    plt.figure('Phase_limb_s2w')
    for i in range(4):

        # plt.plot(body_x_positions[:, i], body_phases[:, i])
        plt.plot(head_positions[:, 0],limb_phases[:, i], label=f"limb {i}")
        plt.legend()
        plt.ylabel('Limb phase [deg]')
        plt.xlabel('Distance [m]')
        plt.grid(True)

    plt.figure('Traj_s2w')
    plot_trajectory(head_positions)  

def plot_ex_5a(num_it):
    for sim_num in range(num_it):
        filename = './logs/ex_5a/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        head_positions = np.asarray(head_positions)
        plt.figure('turning_pos_water_sim_' + str(sim_num))
        plot_positions(times, head_positions)  
        plt.figure('turning_water_sim_' + str(sim_num))
        plot_trajectory(head_positions)  

def plot_ex_5b(num_it):
    for sim_num in range(num_it):
        filename = './logs/ex_5b/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        head_positions = np.asarray(head_positions)
        plt.figure('backward_pos_water_sim_' + str(sim_num))
        plot_positions(times, head_positions)  
        plt.figure('backward_traj_water_sim_' + str(sim_num))
        plot_trajectory(head_positions)  

def plot_ex_5c(num_it):
    for sim_num in range(num_it):
        filename = './logs/ex_5c/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        head_positions = np.asarray(head_positions)
        plt.figure('turning_pos_land_sim_' + str(sim_num))
        plot_positions(times, head_positions)  
        plt.figure('turning_land_sim_' + str(sim_num))
        plot_trajectory(head_positions)  

def plot_ex_5d(num_it):
    for sim_num in range(num_it):
        filename = './logs/ex_5d/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        head_positions = np.asarray(head_positions)
        plt.figure('backward_pos_land_sim_' + str(sim_num))
        plot_positions(times, head_positions)  
        plt.figure('backward_traj_land_sim_' + str(sim_num))
        plot_trajectory(head_positions)  

def plot_ex_6a(num_it):
    filename = './logs/ex_6a/simulation_{}.{}'
    data = SalamandraData.from_file(filename.format('0', 'h5'))
    phases = data.state.phases()
    limb_phases = phases[:,16:]
    ground_forces = np.asarray(data.sensors.contacts.reactions()) # array of type [times, 4(limbs), 3(xyz)]
    high_threshold_grf = 18
    low_threshold_grf = 5
    ground_forces_i = low_threshold_grf < ground_forces
    ground_forces_i =  ground_forces > high_threshold_grf
    ground_forces[ground_forces_i] = 0
    n = 10  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    # print(ground_forces)
    # print(np.shape(ground_forces))
    # print(type(ground_forces))
    # f, axes = plt.subplots(4)
    # f.suptitle('')
    timestep = data.timestep
    n_iterations = np.shape(data.sensors.links.array)[0]
    times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
    )
    timestep = times[1] - times[0]
    amplitudes = data.state.amplitudes()
    
    plt.figure('limb_phases_vs_ground_reaction_forces')
    for i in range(4):
        ground_forces[:, i, 2] = lfilter(b, a, ground_forces[:, i, 2])
        plt.plot(limb_phases[:, i], ground_forces[:, i, 2], label=f"limb {i+1}")
    plt.plot(limb_phases[:, i],-np.cos(limb_phases[:, i]), label=f"cosine")
    plt.legend()
    plt.xlabel('Limb phase [rad]')
    plt.ylabel('Ground Force [N]')
    plt.grid(True)

def plot_ex_6b(num_it):
    filename = './logs/ex_6b/simulation_{}.{}'
    data = SalamandraData.from_file(filename.format('0', 'h5'))
    phases = data.state.phases()
    # limb_phases = phases[:,16:]
    with open(filename.format(0, 'pickle'), 'rb') as param_file:     
                parameters = pickle.load(param_file)
    links_positions = data.sensors.links.urdf_positions()
    head_positions = links_positions[:, 0, :]
    head_positions = np.asarray(head_positions)

    plt.figure("6_b_trajectory")
    plot_trajectory(head_positions)
    plt.legend()
    

def plot_ex_6b_2(num_it):
    # to see the effect of the weight of the sensory feedback and justify its choice
    speed_fw = np.zeros(num_it)
    weights = np.zeros(num_it)
    cost_of_transport = np.zeros(num_it)
    for sim_num in range(num_it):
        filename = './logs/ex_6b/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        with open(filename.format(sim_num, 'pickle'), 'rb') as param_file:     
                    parameters = pickle.load(param_file)
        weights[sim_num] = parameters.weight_sensory_feedback
        links_positions = data.sensors.links.urdf_positions()
        links_vel = data.sensors.links.com_lin_velocities()
        cost_of_transport[sim_num] = compute_cost_of_transport(data)
        speed_fw[sim_num], speed_lat = compute_speed(links_positions, links_vel)

    plt.figure('Visualisation_of_best_weight_speed_large_win_undulation')
    plt.plot(weights, speed_fw)
    # plt.legend()
    plt.xlim([0, 1])
    plt.xlabel('Weight parameter')
    plt.ylabel('Forward speed [m/s]')
    plt.grid(True)

def plot_ex_6c(num_it):
    for sim_num in range(num_it):
        filename = './logs/ex_6c/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        with open(filename.format(sim_num, 'pickle'), 'rb') as param_file:     
                    parameters = pickle.load(param_file)
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        head_positions = np.asarray(head_positions)

        plt.figure("6_c_trajectories")
        plot_trajectory(head_positions, label=f'sim {sim_num+1}')
        plt.legend()
    
def plot_ex_6d(num_it):
    cost_of_transport = np.zeros(num_it)
    for sim_num in range(num_it):
        filename = './logs/ex_6d/simulation_{}.{}'
        data = SalamandraData.from_file(filename.format(sim_num, 'h5'))
        with open(filename.format(sim_num, 'pickle'), 'rb') as param_file:     
                    parameters = pickle.load(param_file)
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        head_positions = np.asarray(head_positions)
        cost_of_transport[sim_num] = compute_cost_of_transport(data)

        # if sim_num < 2:
        plt.figure("6_d_trajectories")
        plot_trajectory(head_positions, label=f'sim {sim_num+1}, ct = {cost_of_transport[sim_num]}')
        plt.legend()


def main(plot=True):
    """Main"""
    # plot_ex_2a(num_it=100)
    # plot_ex_2b(num_it=100)
    # plot_ex_3a(num_it=100)
    # plot_ex_3b(num_it=100)
    # plot_ex_4a()
    # plot_ex_4b()
    # plot_ex_5a(2)
    # plot_ex_5b(1)
    # plot_ex_5c(2)
    # plot_ex_5d(1)
    plot_ex_6a(1)
    plot_ex_6b(1)
    # plot_ex_6b_2(1)
    plot_ex_6c(3)
    plot_ex_6d(4)
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=save_plots())

