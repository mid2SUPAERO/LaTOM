"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from openmdao.api import ExplicitComponent
from dymos import declare_time, declare_state, declare_parameter


@declare_time(units='s')
@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('theta', rate_source='thetadot', units='rad')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')
@declare_parameter('alpha', targets=['alpha'], units='rad')
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')
class ODE2dThrust(ExplicitComponent):
    """ODE2dThrust class defines the equations of motion for a two dim. powered trajectory.

    Other Parameters
    ----------------
    time : float
        Represents the time variable of the system [-]
    r : float
        Represents a position along the trajectory. The distance is measured from the center of the central body [-]
    theta : float
        Angle spawn from the starting point of the orbit to the final one [-]
    u : float
        Radial velocity of a point along the trajectory [-]
    v : float
        Tangential velocity of a point along the trajectory [-]
    m : float
        Mass of the space vehicle that performs the trajectory [-]
    alpha : float
        Angle defining the thrust direction [-]
    thrust : float
        Value of the applied thrust force [-]
    w : float
        Value of the exhaust velocity [-]
    num_nodes : int
        Number of nodes where to compute the equations
    GM : float
        Gravitational constant [-]

    """

    def initialize(self):
        """Initializes the `ODE2dThrust` class variables. """

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('w', types=float)

    def setup(self):
        """Setup of ODE2dThrust parameters. Declaration of input, output and partials variables. """

        nn = self.options['num_nodes']
        w = self.options['w']

        # inputs (ODE right-hand side)
        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity', units='m/s')
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')

        self.add_input('alpha', val=np.zeros(nn), desc='thrust direction', units='rad')
        self.add_input('w', val=w*np.ones(nn), desc='exhaust velocity', units='m/s')

        # outputs (ODE left-hand side)
        self.add_output('rdot', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_output('thetadot', val=np.zeros(nn), desc='angular rate', units='rad/s')
        self.add_output('udot', val=np.zeros(nn), desc='radial acceleration', units='m/s**2')
        self.add_output('vdot', val=np.zeros(nn), desc='tangential acceleration', units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), desc='mass rate', units='kg/s')

        # partial derivatives of the ODE outputs respect to the ODE inputs
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='rdot', wrt='u', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='thetadot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='thetadot', wrt='v', rows=ar, cols=ar)

        self.declare_partials(of='udot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='thrust', rows=ar, cols=ar)

        self.declare_partials(of='vdot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='vdot', wrt='thrust', rows=ar, cols=ar)

        self.declare_partials(of='mdot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='mdot', wrt='w', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        """ Compute the output variables """

        gm = self.options['GM']

        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        thrust = inputs['thrust']
        w = inputs['w']

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        outputs['rdot'] = u
        outputs['thetadot'] = v / r
        outputs['udot'] = -gm / r ** 2 + v ** 2 / r + (thrust / m) * sin_alpha
        outputs['vdot'] = -u * v / r + (thrust / m) * cos_alpha
        outputs['mdot'] = -thrust / w

    def compute_partials(self, inputs, jacobian):
        """ Compute the partial derivatives variables """

        gm = self.options['GM']

        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        thrust = inputs['thrust']
        w = inputs['w']

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        jacobian['thetadot', 'r'] = -v / r ** 2
        jacobian['thetadot', 'v'] = 1 / r

        jacobian['udot', 'r'] = 2 * gm / r ** 3 - (v / r) ** 2
        jacobian['udot', 'v'] = 2 * v / r
        jacobian['udot', 'm'] = -(thrust / m ** 2) * sin_alpha
        jacobian['udot', 'alpha'] = (thrust / m) * cos_alpha
        jacobian['udot', 'thrust'] = sin_alpha / m

        jacobian['vdot', 'r'] = u * v / r ** 2
        jacobian['vdot', 'u'] = -v / r
        jacobian['vdot', 'v'] = -u / r
        jacobian['vdot', 'm'] = -(thrust / m ** 2) * cos_alpha
        jacobian['vdot', 'alpha'] = -(thrust / m) * sin_alpha
        jacobian['vdot', 'thrust'] = cos_alpha / m

        jacobian['mdot', 'thrust'] = -1 / w
        jacobian['mdot', 'w'] = thrust / w ** 2


class ODE2dConstThrust(ODE2dThrust):
    """ODE2dThrust class defines the equations of motion for a two dim. powered trajectory with constant thrust.

    Other Parameters
    ----------------
    time : float
        Represents the time variable of the system [-]
    r : float
        Represents a position along the trajectory. The distance is measured from the center of the central body [-]
    theta : float
        Angle spawn from the starting point of the orbit to the final one [-]
    u : float
        Radial velocity of a point along the trajectory [-]
    v : float
        Tangential velocity of a point along the trajectory [-]
    m : float
        Mass of the space vehicle that performs the trajectory [-]
    alpha : float
        Angle defining the thrust direction [-]
    thrust : float
        Value of the applied thrust force [-]
    w : float
        Value of the exhaust velocity [-]
    num_nodes : int
        Number of nodes where to compute the equations
    GM : float
        Gravitational constant [-]
    T : float
       Value of the constant thrust force [-]

    """

    def initialize(self):
        """Initializes the `ODE2dConstThrust` class variables. """

        ODE2dThrust.initialize(self)
        self.options.declare('T', types=float)

    def setup(self):
        """ Setup of ODE2dConstThrust parameters. Declaration of input, output and partials variables."""

        ODE2dThrust.setup(self)
        self.add_input('thrust', val=self.options['T']*np.ones(self.options['num_nodes']), desc='thrust', units='N')


class ODE2dVarThrust(ODE2dThrust):
    """ODE2dThrust class defines the equations of motion for a two dim. powered trajectory with variable thrust.

    Other Parameters
    ----------------
    time : float
        Represents the time variable of the system [-]
    r : float
        Represents a position along the trajectory. The distance is measured from the center of the central body [-]
    theta : float
        Angle spawn from the starting point of the orbit to the final one [-]
    u : float
        Radial velocity of a point along the trajectory [-]
    v : float
        Tangential velocity of a point along the trajectory [-]
    m : float
        Mass of the space vehicle that performs the trajectory [-]
    alpha : float
        Angle defining the thrust direction [-]
    thrust : float
        Value of the applied thrust force [-]
    w : float
        Value of the exhaust velocity [-]
    num_nodes : int
        Number of nodes where to compute the equations
    GM : float
        Gravitational constant [-]

    """

    def setup(self):
        """ Setup of `ODE2dVarThrust` parameters. Declaration of input, output and partials variables."""

        ODE2dThrust.setup(self)
        self.add_input('thrust', val=np.zeros(self.options['num_nodes']), desc='thrust', units='N')

@declare_time(units='s')
@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')
class ODE2dVertical(ExplicitComponent):
    """ODE2dThrust class defines the equations of motion for a two dim. powered trajectory with variable thrust.

    Other Parameters
    ----------------
    time : float
        Represents the time variable of the system [-]
    r : float
        Represents a position along the trajectory. The distance is measured from the center of the central body [-]
    u : float
        Radial velocity of a point along the trajectory [-]
    m : float
        Mass of the space vehicle that performs the trajectory [-]
    thrust : float
        Value of the applied thrust force [-]
    w : float
        Value of the exhaust velocity [-]
    num_nodes : int
        Number of nodes where to compute the equations
    GM : float
        Gravitational constant [-]
    T : float
       Value of the constant thrust force [-]

    """

    def initialize(self):
        """Initializes the `ODE2dVertical` class variables. """

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('T', types=float)
        self.options.declare('w', types=float)

    def setup(self):
        """ Setup of `ODE2dVertical` parameters. Declaration of input, output and partials variables."""

        nn = self.options['num_nodes']

        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')

        self.add_input('thrust', val=self.options['T']*np.ones(nn), desc='thrust', units='N')
        self.add_input('w', val=self.options['w']*np.ones(nn), desc='exhaust velocity', units='m/s')

        self.add_output('rdot', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_output('udot', val=np.zeros(nn), desc='radial acceleration', units='m/s**2')
        self.add_output('mdot', val=np.zeros(nn), desc='mass flow rate', units='kg/s')

        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='rdot', wrt='u', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='udot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='udot', wrt='thrust', rows=ar, cols=ar)

        self.declare_partials(of='mdot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='mdot', wrt='w', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        """ Compute the output variables"""
        gm = self.options['GM']

        r = inputs['r']
        u = inputs['u']
        m = inputs['m']
        thrust = inputs['thrust']
        w = inputs['w']

        outputs['rdot'] = u
        outputs['udot'] = -gm / r ** 2 + thrust / m
        outputs['mdot'] = -thrust / w

    def compute_partials(self, inputs, jacobian):
        """ Compute the partial derivative variables"""

        gm = self.options['GM']

        r = inputs['r']
        m = inputs['m']
        thrust = inputs['thrust']
        w = inputs['w']

        jacobian['udot', 'r'] = 2 * gm / r ** 3
        jacobian['udot', 'm'] = -thrust / m ** 2
        jacobian['udot', 'thrust'] = 1 / m

        jacobian['mdot', 'thrust'] = -1 / w
        jacobian['mdot', 'w'] = thrust / w ** 2


class SafeAlt(ExplicitComponent):
    """SafeAlt class defines the curve representing geographical constraints on the central body surface

    Other Parameters
    ----------------
    num_nodes : int
        Number of nodes where to compute the equations
    R : float
        Moon radius [m]
    alt_safe : float
        Altitude of the curve representing the geographical constraint [-]
    slope : float
        Slope of the curves defining a geographical constraint

    """

    def initialize(self):
        """ Initializes the `SafeAlt` class variables """
        self.options.declare('num_nodes', types=int)
        self.options.declare('R', types=float)
        self.options.declare('alt_safe', types=float)
        self.options.declare('slope', types=float)

    def setup(self):
        """ Setup of `SafeAlt` parameters. Declaration of input, output and partials variables."""

        nn = self.options['num_nodes']

        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('theta', val=np.zeros(nn), desc='true anomaly', units='rad')

        self.add_output('r_safe', val=np.zeros(nn), desc='minimum safe radius')
        self.add_output('dist_safe', val=np.zeros(nn), desc='distance from minimum safe radius')

        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='r_safe', wrt='theta', rows=ar, cols=ar)
        self.declare_partials(of='dist_safe', wrt='r', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='dist_safe', wrt='theta', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        """ Compute the output variables"""
        r_moon = self.options['R']
        alt_safe = self.options['alt_safe']
        slope = self.options['slope']

        r = inputs['r']
        theta = inputs['theta']

        r_safe = r_moon + alt_safe*r_moon*theta/(r_moon*theta + alt_safe/slope)

        outputs['r_safe'] = r_safe
        outputs['dist_safe'] = r - r_safe

    def compute_partials(self, inputs, jacobian):
        """ Compute the partial derivative variables"""
        r_moon = self.options['R']
        alt_safe = self.options['alt_safe']
        slope = self.options['slope']

        theta = inputs['theta']

        drsafe_dtheta = alt_safe**2*r_moon*slope/(alt_safe + slope*r_moon*theta)**2

        jacobian['r_safe', 'theta'] = drsafe_dtheta
        jacobian['dist_safe', 'theta'] = -drsafe_dtheta


class Polar2COE(ExplicitComponent):
    """Polar2COE class defines the set of equations to derive the Classical Orbital Elements from Polar coordinates.

    Other Parameters
    ----------------
    num_nodes : int
            Number of nodes where to compute the equations
    GM : float
        Gravitational constant [-]

    """

    def initialize(self):
        """Initializes the `Polar2COE` class variables. """

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)

    def setup(self):
        """Setup of `Polar2COE` parameters. Declaration of input, output and partials variables. """

        nn = self.options['num_nodes']

        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity', units='m/s')


class Polar2RApo(Polar2COE):
    """Polar2RApo class defines the set of equations to derive the apoapsis radius from Polar coordinates.

    Other Parameters
    ----------------
    ra : float
        Value of the apoapsis radius [-]

    """

    def initialize(self):
        """Initializes the `Polar2RApo` class variables. """

        Polar2COE.initialize(self)
        self.options.declare('ra', types=float)

    def setup(self):
        """Setup of `Polar2RApo` parameters. Declaration of input, output and partials variables. """

        Polar2COE.setup(self)

        self.add_output('c', val=np.zeros(self.options['num_nodes']), desc='condition to reach the apolune')

        ar = np.arange(self.options['num_nodes'])

        """
        self.declare_partials(of='c', wrt='r', method='cs')
        self.declare_partials(of='c', wrt='u', method='cs')
        self.declare_partials(of='c', wrt='v', method='cs')
        """

        self.declare_partials(of='c', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='c', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='c', wrt='v', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        """Compute the output variables. """

        gm = self.options['GM']
        ra = self.options['ra']
        r = inputs['r']
        u = inputs['u']
        v = inputs['v']

        a = 2*gm - r*(u*u + v*v)
        outputs['c'] = a*(a*ra*ra + r*(r*r*v*v - 2*gm*ra))

        # outputs['c'] = ra - (r*(gm + ((r*v*v-gm)**2 + (r*u*v)**2)**0.5))/(2*gm - r*(u**2 + v**2))

    def compute_partials(self, inputs, jacobian):
        """Compute the partial derivative variables. """

        gm = self.options['GM']
        ra = self.options['ra']
        r = inputs['r']
        u = inputs['u']
        v = inputs['v']

        a = 2 * gm - r * (u * u + v * v)
        b = 2*ra*(a*ra - gm*r) + r**3*v**2

        jacobian['c', 'r'] = - (u*u + v*v)*b + a*(3*r**2*v**2 - 2*gm*ra)
        jacobian['c', 'u'] = - 2*r*u*b
        jacobian['c', 'v'] = - 2*r*v*b + 2*a*v*r**3


class Polar2AEH(Polar2COE):
    """Polar2AEH class defines the set of equations to derive the a and h from Polar coordinates.

    The method computes the semi-major axis a, the specific energy and the angular momentum h starting from the polar
    coordinates r, u, v.

    Other Parameters
    ----------------
    num_nodes : int
            Number of nodes where to compute the equations
    GM : float
        Gravitational constant [-]

    """

    def setup(self):
        """Setup of `Polar2AEH` parameters. Declaration of input, output and partials variables. """

        Polar2COE.setup(self)

        nn = self.options['num_nodes']

        self.add_output('a', val=np.zeros(nn), desc='semimajor axis')
        self.add_output('eps', val=np.zeros(nn), desc='specific energy')
        self.add_output('h', val=np.zeros(nn), desc='angular momentum')

        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='a', wrt='r', cols=ar, rows=ar)
        self.declare_partials(of='a', wrt='u', cols=ar, rows=ar)
        self.declare_partials(of='a', wrt='v', cols=ar, rows=ar)

        self.declare_partials(of='eps', wrt='r', cols=ar, rows=ar)
        self.declare_partials(of='eps', wrt='u', cols=ar, rows=ar)
        self.declare_partials(of='eps', wrt='v', cols=ar, rows=ar)

        self.declare_partials(of='h', wrt='r', cols=ar, rows=ar)
        self.declare_partials(of='h', wrt='v', cols=ar, rows=ar)

    def compute(self, inputs, outputs):
        """Compute the output variables. """

        gm = self.options['GM']

        r = inputs['r']
        u = inputs['u']
        v = inputs['v']

        outputs['a'] = gm*r/(2*gm - r*(u*u + v*v))
        outputs['eps'] = (u*u + v*v)*0.5 - gm/r
        outputs['h'] = r*v

    def compute_partials(self, inputs, jacobian):
        """Compute the partial derivative variables. """

        gm = self.options['GM']

        r = inputs['r']
        u = inputs['u']
        v = inputs['v']

        d = 2*gm - r*(u*u + v*v)

        jacobian['a', 'r'] = 2*gm*gm/d/d
        jacobian['a', 'u'] = 2*gm*r*r*u/d/d
        jacobian['a', 'v'] = 2*gm*r*r*v/d/d

        jacobian['eps', 'r'] = gm/r/r
        jacobian['eps', 'u'] = u
        jacobian['eps', 'v'] = v

        jacobian['h', 'r'] = v
        jacobian['h', 'v'] = r
