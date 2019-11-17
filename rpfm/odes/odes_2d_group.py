"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from openmdao.api import Group

from dymos import declare_time, declare_state, declare_parameter
from rpfm.odes.odes_2d import ODE2dConstThrust, ODE2dVarThrust, SafeAlt, Polar2RApo, Polar2AH

@declare_time(units='s')
@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('theta', rate_source='thetadot', targets=['theta'], units='rad')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')
@declare_parameter('alpha', targets=['alpha'], units='rad')
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')
class ODE2dVToff(Group):

    def initialize(self):

        self.options.declare('num_nodes', types=int)

        self.options.declare('GM', types=float)
        self.options.declare('w', types=float)
        self.options.declare('R', types=float)
        self.options.declare('alt_safe', types=float)
        self.options.declare('slope', types=float)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_subsystem(name='odes', subsys=ODE2dVarThrust(num_nodes=nn, GM=self.options['GM'], w=self.options['w']),
                           promotes_inputs=['r', 'u', 'v', 'm', 'alpha', 'thrust', 'w'],
                           promotes_outputs=['rdot', 'thetadot', 'udot', 'vdot', 'mdot'])

        self.add_subsystem(name='safe_alt',
                           subsys=SafeAlt(num_nodes=nn, R=self.options['R'], alt_safe=self.options['alt_safe'],
                                          slope=self.options['slope']),
                           promotes_inputs=['r', 'theta'],
                           promotes_outputs=['r_safe', 'dist_safe'])


@declare_time(units='s')
@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('theta', rate_source='thetadot', units='rad')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')
@declare_parameter('alpha', targets=['alpha'], units='rad')
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')
class ODE2dLLO2Apo(Group):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('T', types=float)
        self.options.declare('w', types=float)
        self.options.declare('ra', types=float)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_subsystem(name='odes', subsys=ODE2dConstThrust(num_nodes=nn, GM=self.options['GM'],
                                                                T=self.options['T'], w=self.options['w']),
                           promotes_inputs=['r', 'u', 'v', 'm', 'alpha', 'thrust', 'w'],
                           promotes_outputs=['rdot', 'thetadot', 'udot', 'vdot', 'mdot'])

        self.add_subsystem(name='injection2apo',
                           subsys=Polar2RApo(num_nodes=nn, GM=self.options['GM'], ra=self.options['ra']),
                           promotes_inputs=['r', 'u', 'v'], promotes_outputs=['c'])


@declare_time(units='s')
@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('theta', rate_source='thetadot', units='rad')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')
@declare_parameter('alpha', targets=['alpha'], units='rad')
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')
class ODE2dLLO2HEO(Group):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('T', types=float)
        self.options.declare('w', types=float)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_subsystem(name='odes', subsys=ODE2dConstThrust(num_nodes=nn, GM=self.options['GM'],
                                                                T=self.options['T'], w=self.options['w']),
                           promotes_inputs=['r', 'u', 'v', 'm', 'alpha', 'thrust', 'w'],
                           promotes_outputs=['rdot', 'thetadot', 'udot', 'vdot', 'mdot'])

        self.add_subsystem(name='polar2coe',
                           subsys=Polar2AH(num_nodes=nn, GM=self.options['GM']),
                           promotes_inputs=['r', 'u', 'v'], promotes_outputs=['a', 'h'])
