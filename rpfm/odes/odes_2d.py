"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from openmdao.api import ExplicitComponent


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

        GM = self.options['GM']

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
        outputs['udot'] = -GM / r ** 2 + v ** 2 / r + (thrust / m) * sin_alpha
        outputs['vdot'] = -u * v / r + (thrust / m) * cos_alpha
        outputs['mdot'] = -thrust / w

    def compute_partials(self, inputs, jacobian):

        GM = self.options['GM']

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

        jacobian['udot', 'r'] = 2 * GM / r ** 3 - (v / r) ** 2
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

        GM = self.options['GM']

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
        outputs['udot'] = -GM / r ** 2 + v ** 2 / r + (thrust / m) * sin_alpha
        outputs['vdot'] = -u * v / r + (thrust / m) * cos_alpha
        outputs['mdot'] = -thrust / w

    def compute_partials(self, inputs, jacobian):

        GM = self.options['GM']

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

        jacobian['udot', 'r'] = 2 * GM / r ** 3 - (v / r) ** 2
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
