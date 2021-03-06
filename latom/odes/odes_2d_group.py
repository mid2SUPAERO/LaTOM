"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from openmdao.api import Group

from dymos import declare_time, declare_state, declare_parameter
from latom.odes.odes_2d import ODE2dConstThrust, ODE2dVarThrust, SafeAlt, Polar2RApo, Polar2AEH


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
    """`ODE2dVToff` class defines the equations of motion for a two dim. powered trajectory with vertical take-off.

    Other Parameters
    ----------------
    time : ndarray
        Represents the time variable of the system [-]
    r : ndarray
        Represents a position along the trajectory. The distance is measured from the center of the central body [-]
    theta : ndarray
        Angle spawn from the starting point of the orbit to the final one [-]
    u : ndarray
        Radial velocity of a point along the trajectory [-]
    v : ndarray
        Tangential velocity of a point along the trajectory [-]
    m : ndarray
        Mass of the space vehicle that performs the trajectory [-]
    alpha : ndarray
        Angle defining the thrust direction [-]
    thrust : ndarray
        Value of the applied thrust force [-]
    w : float
        Value of the exhaust velocity [-]
    num_nodes : int
        Number of nodes where to compute the equations
    GM : float
        Standard gravitational parameter for the central attracting body [-]
    R : float
        Moon radius [-]
    alt_safe : float
        Altitude of the curve representing the geographical constraint [-]
    slope : float
        Slope of the curves defining a geographical constraint [-]

    """

    def initialize(self):
        """Initializes the `ODE2dVToff` class variables. """

        self.options.declare('num_nodes', types=int)

        self.options.declare('GM', types=float)
        self.options.declare('w', types=float)
        self.options.declare('R', types=float)
        self.options.declare('alt_safe', types=float)
        self.options.declare('slope', types=float)

    def setup(self):
        """Setup of `ODE2dVToff` parameters. Declaration of subsystem, input and output. """

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
    """`ODE2dLLO2Apo` class defines the equations of motion for a two dim. powered trajectory to leave the intial LLO
    and enter a ballistic arc with apoapsis radius equal to `ra`.

    Other Parameters
    ----------------
    time : ndarray
        Represents the time variable of the system [-]
    r : ndarray
        Represents a position along the trajectory. The distance is measured from the center of the central body [-]
    theta : ndarray
        Angle spawn from the starting point of the orbit to the final one [-]
    u : ndarray
        Radial velocity of a point along the trajectory [-]
    v : ndarray
        Tangential velocity of a point along the trajectory [-]
    m : ndarray
        Mass of the space vehicle that performs the trajectory [-]
    alpha : ndarray
        Angle defining the thrust direction [-]
    thrust : ndarray
        Value of the applied thrust force [-]
    w : float
        Value of the exhaust velocity [-]
    num_nodes : int
        Number of nodes where to compute the equations
    GM : float
        Standard gravitational parameter for the central attracting body [-]
    T : float
       Value of the constant thrust force [-]
    ra : float
        Value of the apoapsis radius [-]

    """

    def initialize(self):
        """Initializes the `ODE2dLLO2Apo` class variables. """

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('T', types=float)
        self.options.declare('w', types=float)
        self.options.declare('ra', types=float)

    def setup(self):
        """Setup of `ODE2dLLO2Apo` parameters. Declaration of subsystem, input and output. """

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
    """ODE2dLLO2HEO class defines the equations of motion for a two dim. powered trajectory from LLO to HEO.

    Other Parameters
    ----------------
    time : ndarray
        Represents the time variable of the system [-]
    r : ndarray
        Represents a position along the trajectory. The distance is measured from the center of the central body [-]
    theta : ndarray
        Angle spawn from the starting point of the orbit to the final one [-]
    u : ndarray
        Radial velocity of a point along the trajectory [-]
    v : ndarray
        Tangential velocity of a point along the trajectory [-]
    m : ndarray
        Mass of the space vehicle that performs the trajectory [-]
    alpha : ndarray
        Angle defining the thrust direction [-]
    thrust : ndarray
        Value of the applied thrust force [-]
    w : float
        Value of the exhaust velocity [-]
    num_nodes : int
        Number of nodes where to compute the equations
    GM : float
        Standard gravitational parameter for the central attracting body [-]
    T : float
       Value of the constant thrust force [-]

    """

    def initialize(self):
        """Initializes the `ODE2dLLO2HEO` class variables. """

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('T', types=float)
        self.options.declare('w', types=float)

    def setup(self):
        """Setup of `ODE2dLLO2HEO` parameters. Declaration of subsystem, input and output. """

        nn = self.options['num_nodes']

        self.add_subsystem(name='odes', subsys=ODE2dConstThrust(num_nodes=nn, GM=self.options['GM'],
                                                                T=self.options['T'], w=self.options['w']),
                           promotes_inputs=['r', 'u', 'v', 'm', 'alpha', 'thrust', 'w'],
                           promotes_outputs=['rdot', 'thetadot', 'udot', 'vdot', 'mdot'])

        self.add_subsystem(name='polar2aeh',
                           subsys=Polar2AEH(num_nodes=nn, GM=self.options['GM']),
                           promotes_inputs=['r', 'u', 'v'], promotes_outputs=['a', 'eps', 'h'])
