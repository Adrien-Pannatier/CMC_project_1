"""Robot parameters"""

import numpy as np
from farms_core import pylog
from simulation_parameters import SimulationParameters

WALK = 2.6
WALK_BODY_PHI = -2.51327
WALK_BODY_LIMB_PHI = 3
SWIM = 3.25
SWIM_PHI = -2.51327

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
        self.der_phases = np.zeros(self.n_oscillators)

        # gains for final motor output
        self.position_body_gain = parameters.position_body_gain
        self.position_limb_gain = parameters.position_limb_gain

        # for part 6
        self.feedback_fac = parameters.feedback_fac
        self.weight_sensory_feedback = parameters.weight_sensory_feedback

        # for part 7
        self.is_touching_ground = False
        self.was_just_touching_ground = False
        self.spawning = True

        

        self.set_amplitudes_rate(parameters)

        self.update(parameters)

        self.state = parameters.state

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_nominal_amplitudes(parameters)  # R_i
        self.set_coupling_weights_and_phase_bias(parameters)

    def step(self, iteration, salamandra_data):
        """Step function called at each iteration

        Parameters
        ----------

        salamandra_data: salamandra_simulation/data.py::SalamandraData
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


        grf_z = np.array(salamandra_data.sensors.contacts.array[:,:,2])[iteration] 
        self.is_touching_ground = np.any(grf_z)

        if not self.is_touching_ground and not self.was_just_touching_ground:
            if self.state == 'ground' and not self.spawning:
                # change the simulation parameters
                self.update(SimulationParameters(drive = SWIM, downward_body_CPG_phi = SWIM_PHI,))
                self.state = 'water'
            if self.state == 'water' and self.spawning:
                self.update(SimulationParameters(drive = SWIM, downward_body_CPG_phi = SWIM_PHI,))
                self.spawning = False
        elif self.is_touching_ground and self.was_just_touching_ground:
            if self.state == 'water':
                self.update(SimulationParameters(drive = WALK, downward_body_CPG_phi = WALK_BODY_PHI, limb_to_body_CPG_phi = WALK_BODY_LIMB_PHI,))
                self.state = 'ground'
            if self.state == 'ground' and self.spawning:
                self.spawning = False
        assert iteration >= 0

        # UNCOMMENT AND COMMENT ABOVE TO RUN EXERCISE 4
        # x_pos = gps[4, 0]
        # if x_pos > 2.7 and self.state == 'ground':
        #     # change the simulation parameters
        #     self.update(SimulationParameters(drive = SWIM, downward_body_CPG_phi = SWIM_PHI,))
        #     # print(grf_z)
        #     self.state == 'water'
        # elif x_pos < 2.7 and self.state == 'water':
        #     self.update(SimulationParameters(drive = WALK, downward_body_CPG_phi = WALK_BODY_PHI, limb_to_body_CPG_phi = WALK_BODY_LIMB_PHI,))
        #     self.state == 'ground'
        # assert iteration >= 0

        self.was_just_touching_ground = self.is_touching_ground

    def get_parameters(self):
        return self.freqs, self.coupling_weights, self.phase_bias, self.rates, self.nominal_amplitudes
    
    def set_der_phases(self, der_phases):
        self.der_phases = der_phases

    def set_frequencies(self, parameters):
        if (parameters.ldlow <= parameters.drive <= parameters.ldhigh):
            freq_limb_l = parameters.lcv1 * parameters.drive_factor_left * parameters.drive + parameters.lcv0
            self.freqs[16] = freq_limb_l
            self.freqs[18] = freq_limb_l
            freq_limb_r = parameters.lcv1 * parameters.drive_factor_right * parameters.drive + parameters.lcv0
            self.freqs[17] = freq_limb_r
            self.freqs[19] = freq_limb_r
        else:
            self.freqs[16:20] = parameters.lvsat
            
        if (parameters.bdlow <= parameters.drive <= parameters.bdhigh):
            freq_body_l = parameters.bcv1 * parameters.drive_factor_left * parameters.drive + parameters.bcv0
            self.freqs[:8] = freq_body_l
            freq_body_r = parameters.bcv1 * parameters.drive_factor_right * parameters.drive + parameters.bcv0
            self.freqs[8:16] = freq_body_r
        else:
            self.freqs[:16] = parameters.bvsat

    def set_coupling_weights_and_phase_bias(self, parameters):
        # upwards, downwards and contralateral in body CPG
        for i in range(self.n_oscillators - self.n_oscillators_legs): # range(16) 0->15
            for j in range(self.n_oscillators - self.n_oscillators_legs): # range(16) 0->15
                if (i==j):
                    continue
                elif (((i-j)==1) and (i+j)!=2*self.n_body_joints-1): # downwards, breaks if case i=7 and j=8
                    self.coupling_weights[i, j] = parameters.downward_body_CPG_w
                    self.phase_bias[i, j] = parameters.downward_body_CPG_phi
                elif ((i-j)==-1 and (i+j)!=2*self.n_body_joints-1): # upwards
                    self.coupling_weights[i, j] = parameters.upward_body_CPG_w
                    self.phase_bias[i, j] = parameters.upward_body_CPG_phi
                elif ((i==j+self.n_body_joints) or (j==i+self.n_body_joints)): #contralateral
                    self.coupling_weights[i, j] = parameters.contralateral_body_CPG_w
                    self.phase_bias[i, j] = parameters.contralateral_body_CPG_phi

        # from limb to body CPG
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if (i==j):
                    continue
                if ((j==16)and(i<4)):
                    self.coupling_weights[j, i] = parameters.limb_to_body_CPG_w
                    self.phase_bias[j, i] = parameters.limb_to_body_CPG_phi
                elif ((j==18)and(4<=i<=7)):
                    self.coupling_weights[j, i] = parameters.limb_to_body_CPG_w
                    self.phase_bias[j, i] = parameters.limb_to_body_CPG_phi
                elif ((j==17)and(8<=i<=11)):
                    self.coupling_weights[j, i] = parameters.limb_to_body_CPG_w
                    self.phase_bias[j, i] = parameters.limb_to_body_CPG_phi
                elif ((j==19)and(12<=i<=15)):
                    self.coupling_weights[j, i] = parameters.limb_to_body_CPG_w
                    self.phase_bias[j, i] = parameters.limb_to_body_CPG_phi
        
        # within the limb CPG
        for i in range(self.n_oscillators - self.n_oscillators_legs, self.n_oscillators):
            for j in range(self.n_oscillators - self.n_oscillators_legs, self.n_oscillators):
                if (i==j) or (i+j==35): # for no diagonal weight
                    continue  
                if ((i + j) == 33) or ((i + j) == 37):
                    self.coupling_weights[i, j] = parameters.within_limb_CPG_w
                    self.phase_bias[i, j] = parameters.within_limb_CPG_phi
                if ((i + j) == 34) or ((i + j) == 36):
                    self.coupling_weights[i, j] = parameters.within_limb_CPG_w
                    if (parameters.walk_backwards == True):
                        self.phase_bias[i, j] = -parameters.within_limb_CPG_phi/2
                    else:
                        self.phase_bias[i, j] = parameters.within_limb_CPG_phi

    def set_amplitudes_rate(self, parameters):
        self.rates[:] = parameters.conv_fac

    def set_nominal_amplitudes(self, parameters):
        # need two set of amplitude rates : for the body and for the limb
        if (parameters.ldlow <= parameters.drive <= parameters.ldhigh):
            nom_amp_limb_l = parameters.lcR1 * parameters.drive_factor_left * parameters.drive + parameters.lcR0
            self.nominal_amplitudes[16] = nom_amp_limb_l
            self.nominal_amplitudes[18] = nom_amp_limb_l
            nom_amp_limb_r = parameters.lcR1 * parameters.drive_factor_right * parameters.drive + parameters.lcR0
            self.nominal_amplitudes[17] = nom_amp_limb_r
            self.nominal_amplitudes[19] = nom_amp_limb_r
        else:
            self.nominal_amplitudes[16:20] = parameters.lRsat

        if (parameters.bdlow <= parameters.drive <= parameters.bdhigh):
            nom_amp_body_l = parameters.amplitude_factor*(parameters.bcR1 * parameters.drive_factor_left * parameters.drive + parameters.bcR0)
            self.nominal_amplitudes[:8] = nom_amp_body_l
            nom_amp_body_r = parameters.amplitude_factor*(parameters.bcR1 * parameters.drive_factor_right * parameters.drive + parameters.bcR0)
            self.nominal_amplitudes[8:16] = nom_amp_body_r
        else:
            self.nominal_amplitudes[:16] = parameters.bRsat
