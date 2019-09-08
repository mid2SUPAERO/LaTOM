"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent
from dymos import declare_time, declare_state, declare_parameter


class ODE2dConstThrust(ExplicitComponent):

    pass


class ODE2dVarThrust(ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)
        self.options.declare('Isp', types=float)
        self.options.declare('g0', types=float)

    def setup(self):

        nn = self.options['num_nodes']
        isp = self.options['Isp']

        self.add_input('r', val=np.zeros(nn), desc='orbit radius')
        self.add_input('u', val=np.zeros(nn), desc='radial velocity')
        self.add_input('v', val=np.zeros(nn), desc='tangential velocity')
        self.add_input('m', val=np.zeros(nn), desc='mass')

        self.add_input('alpha', val=np.zeros(nn), desc='thrust direction')
        self.add_input('thrust', val=np.zeros(nn), desc='thrust')
        self.add_input('Isp', val=isp*np.ones(nn), desc='specific impulse')

        self.add_output('rdot', val=np.zeros(nn), desc='radial velocity')
        self.add_output('thetadot', val=np.zeros(nn), desc='angular rate')
        self.add_output('udot', val=np.zeros(nn), desc='radial acceleration')
        self.add_output('vdot', val=np.zeros(nn), desc='tangential acceleration')
        self.add_output('mdot', val=np.zeros(nn), desc='mass rate')

        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='rdot', wrt='u', rows=ar, cols=ar, val=1)

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
        self.declare_partials(of='mdot', wrt='Isp', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        mu = self.options['mu']
        g0 = self.options['g0']

        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        thrust = inputs['thrust']
        isp = inputs['Isp']

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        outputs['rdot'] = u
        outputs['thetadot'] = v/r
        outputs['udot'] = -mu/r**2 + v**2/r + (thrust/m)*sin_alpha
        outputs['vdot'] = -u*v/r + (thrust/m)*cos_alpha
        outputs['mdot'] = -thrust/isp/g0

    def compute_partials(self, inputs, jacobian):

        mu = self.options['mu']
        g0 = self.options['g0']

        r = inputs['r']
        u = inputs['u']
        v = inputs['v']
        m = inputs['m']
        alpha = inputs['alpha']
        thrust = inputs['thrust']
        isp = inputs['Isp']

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        jacobian['thetadot', 'r'] = -v/r**2
        jacobian['thetadot', 'v'] = 1/r

        jacobian['udot', 'r'] = 2*mu/r**3 - v**2/r**2
        jacobian['udot', 'v'] = 2*v/r
        jacobian['udot', 'm'] = -thrust*sin_alpha/m**2
        jacobian['udot', 'alpha'] = thrust*cos_alpha/m
        jacobian['udot', 'thrust'] = sin_alpha/m

        jacobian['vdot', 'r'] = u*v/r**2
        jacobian['vdot', 'u'] = -v/r
        jacobian['vdot', 'v'] = -u/r
        jacobian['vdot', 'm'] = -thrust*cos_alpha/m**2
        jacobian['vdot', 'alpha'] = -thrust*sin_alpha/m
        jacobian['vdot', 'thrust'] = cos_alpha/m

        jacobian['mdot', 'thrust'] = -1/isp/g0
        jacobian['mdot', 'Isp'] = thrust/isp**2/g0
