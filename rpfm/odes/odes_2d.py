"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from openmdao.api import ExplicitComponent, Group


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


class SafeAlt(ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('R', types=float)
        self.options.declare('alt_min', types=float)
        self.options.declare('slope', types=float)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('theta', val=np.zeros(nn), desc='true anomaly', units='rad')

        self.add_output('r_safe', val=np.zeros(nn), desc='minimum safe radius', units='m')
        self.add_output('dist_safe', val=np.zeros(nn), desc='distance from minimum safe radius', units='m')

        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='r_safe', wrt='theta', rows=ar, cols=ar)
        self.declare_partials(of='dist_safe', wrt='r', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='dist_safe', wrt='theta', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        r_moon = self.options['R']
        alt_min = self.options['alt_min']
        slope = self.options['slope']

        r = inputs['r']
        theta = inputs['theta']

        r_safe = r_moon + alt_min*r_moon*theta/(r_moon*theta + alt_min/slope)

        outputs['r_safe'] = r_safe
        outputs['dist_safe'] = r - r_safe

    def compute_partials(self, inputs, jacobian):

        r_moon = self.options['R']
        alt_min = self.options['alt_min']
        slope = self.options['slope']

        theta = inputs['theta']

        drsafe_dtheta = alt_min**2*r_moon*slope/(alt_min + slope*r_moon*theta)**2

        jacobian['r_safe', 'theta'] = drsafe_dtheta
        jacobian['dist_safe', 'theta'] = -drsafe_dtheta


class ODE2dVToff(Group):

    def initialize(self):

        self.options.declare('num_nodes', types=int)

        self.options.declare('GM', types=float)
        self.options.declare('w', types=float)
        self.options.declare('R', types=float)
        self.options.declare('alt_min', types=float)
        self.options.declare('slope', types=float)

    def setup(self):

        nn = self.options['num_nodes']

        self.add_subsystem(name='odes', subsys=ODE2dVarThrust(num_nodes=nn, GM=self.options['GM'], w=self.options['w']),
                           promotes_inputs=['r', 'u', 'v', 'm', 'alpha', 'thrust', 'w'],
                           promotes_outputs=['rdot', 'thetadot', 'udot', 'vdot', 'mdot'])

        self.add_subsystem(name='safe_alt',
                           subsys=SafeAlt(num_nodes=nn, R=self.options['R'], alt_min=self.options['alt_min'],
                                          slope=self.options['slope']),
                           promotes_inputs=['r', 'theta'],
                           promotes_outputs=['r_safe', 'dist_safe'])
