"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from openmdao.api import ExplicitComponent, Group
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
class ODE2dConstThrust(ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('T', types=float)
        self.options.declare('w', types=float)

    def setup(self):

        nn = self.options['num_nodes']
        thrust = self.options['T']
        w = self.options['w']

        # inputs (ODE right-hand side)
        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity', units='m/s')
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')

        self.add_input('alpha', val=np.zeros(nn), desc='thrust direction', units='rad')
        self.add_input('thrust', val=thrust*np.ones(nn), desc='thrust', units='N')
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


@declare_time(units='s')
@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('theta', rate_source='thetadot', units='rad')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')
@declare_parameter('alpha', targets=['alpha'], units='rad')
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')
class ODE2dVarThrust(ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('w', types=float)

    def setup(self):

        nn = self.options['num_nodes']
        w = self.options['w']

        # inputs (ODE right-hand side)
        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity', units='m/s')
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')

        self.add_input('alpha', val=np.zeros(nn), desc='thrust direction', units='rad')
        self.add_input('thrust', val=np.zeros(nn), desc='thrust', units='N')
        self.add_input('w', val=w * np.ones(nn), desc='exhaust velocity', units='m/s')

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


@declare_time(units='s')
@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')
class ODE2dVertical(ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('T', types=float)
        self.options.declare('w', types=float)

    def setup(self):

        nn = self.options['num_nodes']
        thrust = self.options['T']
        w = self.options['w']

        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')

        self.add_input('thrust', val=thrust*np.ones(nn), desc='thrust', units='N')
        self.add_input('w', val=w*np.ones(nn), desc='exhaust velocity', units='m/s')

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

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('R', types=float)
        self.options.declare('alt_safe', types=float)
        self.options.declare('slope', types=float)

    def setup(self):

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

        r_moon = self.options['R']
        alt_safe = self.options['alt_safe']
        slope = self.options['slope']

        r = inputs['r']
        theta = inputs['theta']

        r_safe = r_moon + alt_safe*r_moon*theta/(r_moon*theta + alt_safe/slope)

        outputs['r_safe'] = r_safe
        outputs['dist_safe'] = r - r_safe

    def compute_partials(self, inputs, jacobian):

        r_moon = self.options['R']
        alt_safe = self.options['alt_safe']
        slope = self.options['slope']

        theta = inputs['theta']

        drsafe_dtheta = alt_safe**2*r_moon*slope/(alt_safe + slope*r_moon*theta)**2

        jacobian['r_safe', 'theta'] = drsafe_dtheta
        jacobian['dist_safe', 'theta'] = -drsafe_dtheta


class Injection2Apolune(ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)
        self.options.declare('ra', types=float)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity', units='m/s')

        self.add_output('c', val=np.zeros(nn), desc='condition to reach the apolune')

        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='c', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='c', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='c', wrt='v', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        gm = self.options['GM']
        ra = self.options['ra']
        r = inputs['r']
        u = inputs['u']
        v = inputs['v']

        a = 2*gm - r*(u*u + v*v)

        outputs['c'] = a*(a*ra*ra + r*(r*r*v*v - 2*gm*ra))

    def compute_partials(self, inputs, jacobian):

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


class Polar2COE(ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('GM', types=float)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity', units='m/s')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity', units='m/s')

        self.add_output('a', val=np.zeros(nn), desc='semimajor axis')
        self.add_output('e2', val=np.zeros(nn), desc='eccentricity squared')
        self.add_output('h', val=np.zeros(nn), desc='angular momentum')

        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='a', wrt='r', cols=ar, rows=ar)
        self.declare_partials(of='a', wrt='u', cols=ar, rows=ar)
        self.declare_partials(of='a', wrt='v', cols=ar, rows=ar)

        self.declare_partials(of='h', wrt='r', cols=ar, rows=ar)
        self.declare_partials(of='h', wrt='v', cols=ar, rows=ar)

        self.declare_partials(of='e2', wrt='r', cols=ar, rows=ar)
        self.declare_partials(of='e2', wrt='u', cols=ar, rows=ar)
        self.declare_partials(of='e2', wrt='v', cols=ar, rows=ar)

    def compute(self, inputs, outputs):

        gm = self.options['GM']

        r = inputs['r']
        u = inputs['u']
        v = inputs['v']

        n = r*u*v
        d = r*v*v - gm

        outputs['a'] = gm*r/(2*gm - r*(u*u + v*v))
        outputs['e2'] = (d * d + n * n) / gm / gm
        outputs['h'] = r*v

    def compute_partials(self, inputs, jacobian):

        gm = self.options['GM']

        r = inputs['r']
        u = inputs['u']
        v = inputs['v']

        d = 2*gm - r*(u*u + v*v)

        jacobian['a', 'r'] = 2*gm*gm/d/d
        jacobian['a', 'u'] = 2*gm*r*r*u/d/d
        jacobian['a', 'v'] = 2*gm*r*r*v/d/d

        jacobian['h', 'r'] = v
        jacobian['h', 'v'] = r

        jacobian['e2', 'r'] = 2*v*v/gm/gm*(r*(u*u + v*v) - gm)
        jacobian['e2', 'u'] = 2*r*r*v*v*u/gm/gm
        jacobian['e2', 'v'] = 2*r*v/gm/gm*(2*(r*v*v - gm) + r*u*u*v)
