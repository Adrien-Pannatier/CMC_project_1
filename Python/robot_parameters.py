"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.conv_fac = parameters.conv_fac
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.feedback_gains_swim = np.zeros(self.n_oscillators)
        self.feedback_gains_walk = np.zeros(self.n_oscillators)
        
        # gains for final motor output
        self.position_body_gain = parameters.position_body_gain
        self.position_limb_gain = parameters.position_limb_gain

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights_and_phase_bias(parameters)  # w_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def step(self, iteration, salamandra_data):
        """Step function called at each iteration

        Parameters
        ----------

        salamanra_data: salamandra_simulation/data.py::SalamandraData
            Contains the robot data, including network and sensors.

        gps (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 representing the GPS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        gps = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )
        # print("GPGS: {}".format(gps[4, 0]))
        # print("drive: {}".format(self.sim_parameters.drive))


    def get_parameters(self):
        return self.freqs, self.coupling_weights, self.phase_bias, self.rates, self.nominal_amplitudes
    
    def set_frequencies(self, parameters):
        # need two set of frequencies : for the body and for the limb
        if (parameters.ldlow <= parameters.drive <= parameters.ldhigh):
            freq_limb = parameters.lcv1 * parameters.drive + parameters.lcv0
            self.freqs[16:20] = freq_limb
        else:
            self.freqs[16:20] = parameters.lvsat
        if (parameters.bdlow <= parameters.drive <= parameters.bdhigh):
            freq_body = parameters.bcv1 * parameters.drive + parameters.bcv0
            self.freqs[:16] = freq_body
        else:
            self.freqs[:16] = parameters.bvsat

    def set_coupling_weights_and_phase_bias(self, parameters):
        # upwards and downwards in body CPG
        for i in range(self.n_oscillators - self.n_oscillators_legs):
            for j in range(self.n_oscillators - self.n_oscillators_legs):
                if (i==j):
                    continue
                elif (((i-j)==1) & ((i+j)!=2*self.n_body_joints+1)): # downwards, breaks if case i=8 and j=9
                    self.coupling_weights[i][j] = parameters.downward_body_CPG_w
                    self.phase_bias[i][j] = parameters.downward_body_CPG_phi
                elif ((i-j)==-1 & (i+j)!=17): # upwards
                    self.coupling_weights[i][j] = parameters.upward_body_CPG_w
                    self.phase_bias[i][j] = parameters.upward_body_CPG_phi
                elif (i==j+self.n_body_joints) | (j==i+self.n_body_joints): #contralateral
                    self.coupling_weights[i][j] = parameters.contralateral_body_CPG_w
                    self.phase_bias[i][j] = parameters.contralateral_body_CPG_phi

        # from limb to body CPG
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if (i==j):
                    continue
                if ((j==16)&(i<4)):
                    self.coupling_weights[i][j] = parameters.limb_to_body_CPG_w
                    self.phase_bias[i][j] = parameters.limb_to_body_CPG_phi
                elif ((j==17)&(4<=i<=7)):
                    self.coupling_weights[i][j] = parameters.limb_to_body_CPG_w
                    self.phase_bias[i][j] = parameters.limb_to_body_CPG_phi
                elif ((j==18)&(8<=i<=11)):
                    self.coupling_weights[i][j] = parameters.limb_to_body_CPG_w
                    self.phase_bias[i][j] = parameters.limb_to_body_CPG_phi
                elif ((j==19)&(12<=i<=15)):
                    self.coupling_weights[i][j] = parameters.limb_to_body_CPG_w
                    self.phase_bias[i][j] = parameters.limb_to_body_CPG_phi

        # within the limb CPG
        for i in range(self.n_oscillators - self.n_oscillators_legs + 1, self.n_oscillators):
            for j in range(self.n_oscillators - self.n_oscillators_legs + 1, self.n_oscillators):
                    if (i==j):
                        continue  
                    self.coupling_weights[i][j] = parameters.within_limb_CPG_w
                    self.phase_bias[i][j] = parameters.within_limb_CPG_phi

    def set_amplitudes_rate(self, parameters):
        self.rates[:] = parameters.conv_fac

    def set_nominal_amplitudes(self, parameters):
        # need two set of amplitude rates : for the body and for the limb
        if (parameters.ldlow <= parameters.drive <= parameters.ldhigh):
            nom_amp_limb = parameters.lcR1 * parameters.drive + parameters.lcR0
            self.nominal_amplitudes[16:20] = nom_amp_limb
        else:
            self.nominal_amplitudes[16:20] = parameters.lRsat
        if (parameters.bdlow <= parameters.drive <= parameters.bdhigh):
            nom_amp_body = parameters.bcR1 * parameters.drive + parameters.bcR0
            self.nominal_amplitudes[:16] = nom_amp_body
        else:
            self.nominal_amplitudes[:16] = parameters.bRsat
