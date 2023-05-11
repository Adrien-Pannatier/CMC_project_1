"""Simulation parameters"""

import numpy as np

class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 4
        self.duration = 30
        self.initial_phases = None
        self.position_body_gain = 0.6  # default do not change
        self.position_limb_gain = 1  # default do not change
        self.phase_lag_body = None
        self.amplitude_gradient = None
        self.downward_body_CPG_w = 10
        self.upward_body_CPG_w = 10
        self.contralateral_body_CPG_w = 10
        self.limb_to_body_CPG_w = 30
        self.within_limb_CPG_w = 10
        self.downward_body_CPG_phi = -2*np.pi/self.n_body_joints
        self.upward_body_CPG_phi = 2*np.pi/self.n_body_joints
        self.contralateral_body_CPG_phi = np.pi
        self.limb_to_body_CPG_phi = np.pi
        self.within_limb_CPG_phi = np.pi
        self.conv_fac = 20
        self.bcv1   = 0.2
        self.bcv0   = 0.3
        self.bvsat  = 0
        self.bcR1   = 0.065
        self.bcR0   = 0.196
        self.bRsat  = 0
        self.bdlow  = 1
        self.bdhigh = 5
        self.lcv1   = 0.2
        self.lcv0   = 0
        self.lvsat  = 0
        self.lcR1   = 0.131
        self.lcR0   = 0.131
        self.lRsat  = 0
        self.ldlow  = 1
        self.ldhigh = 3
        self.drive = 0
        self.state = None
        # Feel free to add more parameters 

        # Disruptions
        self.set_seed = False
        self.randseed = 0
        self.n_disruption_couplings = 0
        self.n_disruption_oscillators = 0
        self.n_disruption_sensors = 0

        # Tegotae
        self.weights_contact_body = 0.0
        self.weights_contact_limb_i = 0.0
        self.weights_contact_limb_c = 0.0

        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)