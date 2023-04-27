"""Oscillator network ODE"""

import numpy as np
from scipy.integrate import ode
from robot_parameters import RobotParameters


def network_ode(_time, state, robot_parameters, loads, contact_sens):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters
    loads: <np.array>
        The lateral forces applied to the body links

    Returns
    -------
    dstate: <np.array>
        Returns derivative of state (phases and amplitudes)

    """
    n_oscillators = robot_parameters.n_oscillators # 20
    n_oscillators_legs = robot_parameters.n_oscillators_legs # 4
    n_body_joints = robot_parameters.n_body_joints # 8
    coupling_weights = robot_parameters.coupling_weights
    phase_bias = robot_parameters.phase_bias
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]
    der_phases = np.zeros_like(phases)
    der_amplitudes = np.zeros_like(amplitudes)
    # CONV_FAC = 20 
    # Implement equation here
    # robot_parameters.update(parameters) a = 20
    # BETTER TO USE FOR LOOP FOR EACH CASE IMO
    ##########################################

    # array with [freq, coupling_weight, phase_bias, rates, nominal_amp] 
    robot_param = robot_parameters.get_parameters() 

    for i in range(n_oscillators): # includes body (1->16) and limbs (17->20)
        for j in range(n_oscillators): # includes body (1->16) and limbs (17->20)
            if i == j:
                continue
            sum_phase = np.sum(amplitudes[j]*robot_param[1][i][j]*np.sin(phases[i] - phases[j]-robot_param[2][i][j]))
        der_phases[i] = 2*np.pi*robot_param[0][i] + sum_phase 
        der_amplitudes[i] = robot_param[3][i]*(robot_param[4][i]-amplitudes[i])
    return np.concatenate([der_phases, der_amplitudes])

def motor_output(phases, amplitudes, iteration):
    """Motor output

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    motor_outputs: <np.array>
        Motor outputs for joint in the system.

    """
    # Last 4 oscillators define output of each limb.
    # Each limb has 2 degree of freedom
    # Implement equation here
    # 16 + 4 oscillators
    # spine motors: 8 -> Mapped from phases[:8] & phases[8:16] and amplitudes[:8] & amplitudes[8:16]
    # leg motors: 8 -> Mapped from phases[16:20] and amplitudes[16:20] with cos(shoulder) & sin(wrist) for each limb
    # output -> spine output for motor (head to tail) + leg output (Front Left
    # shoulder, Front Left wrist, Front Right, Hind Left, Hind right)
    
    #phases = np.append(
    #    np.zeros_like(phases)[
    #        :16], np.zeros_like(phases)[
    #        16:20])
    #amplitudes = np.append(
    #    np.zeros_like(amplitudes)[
    #        :16], np.zeros_like(amplitudes)[
    #        16:20])
    
    motor_outputs = np.zeros_like(amplitudes)[:16]

    # body output for joints 
    for i in range(8):
        motor_outputs[i] = amplitudes[i]*(1 + np.cos(phases[i]) - amplitudes[i+8]*(1 + np.cos(phases[i+8])))

    # body out put for shoulder and wrist joints
    for i in range(4):
        # shoulder joints
        motor_outputs[8+2*i] = amplitudes[16+i]*np.cos(phases[16+i])
        # wrist joints
        motor_outputs[9+2*i] = amplitudes[16+i]*np.sin(phases[16+i])
    
    return motor_outputs

class SalamandraNetwork:
    """Salamandra oscillator network"""

    def __init__(self, sim_parameters, n_iterations, state):
        super().__init__()
        self.n_iterations = n_iterations
        # States
        self.state = state
        # Parameters
        self.robot_parameters = RobotParameters(sim_parameters)
        # Set initial state
        # Replace your oscillator phases here
        self.state.set_phases(
            iteration=0,
            value=1e-4*np.random.rand(self.robot_parameters.n_oscillators),
        )
        # Set solver
        self.solver = ode(f=network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state.array[0], t=0.0)

    def step(self, iteration, time, timestep, loads=None, contact_sens=None):
        """Step"""
        if loads is None:
            loads = np.zeros(self.robot_parameters.n_joints)
        if iteration + 1 >= self.n_iterations:
            return
        self.solver.set_f_params(self.robot_parameters, loads, contact_sens)
        self.state.array[iteration+1, :] = self.solver.integrate(time+timestep)

    def outputs(self, iteration=None):
        """Oscillator outputs"""
        # Implement equation here
        return np.zeros(12)

    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        oscillator_output = motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )
        return oscillator_output

