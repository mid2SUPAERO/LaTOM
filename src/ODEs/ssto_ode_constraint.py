# -*- coding: utf-8 -*-
"""
The script defines the equations of motion (ODEs) and their partial derivatives required
to describe a two-dimensional ascent trajectory with constrained minimum altitude

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""
    
from __future__ import print_function, division, absolute_import

from openmdao.api import Group
from dymos import declare_time, declare_state, declare_parameter

from ODEs.ssto_ode import sstoODEthrottle
from ODEs.constraint_eqn import ConstraintComponent

@declare_time(units='s')

@declare_state('r', rate_source='rdot', targets=['r'], units='m')
@declare_state('theta', rate_source='thetadot', targets=['theta'], units='rad')
@declare_state('u', rate_source='udot', targets=['u'], units='m/s')
@declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
@declare_state('m', rate_source='mdot', targets=['m'], units='kg')

@declare_parameter('alpha', targets=['alpha'], units='rad')
@declare_parameter('k', targets=['k'])
@declare_parameter('thrust', targets=['thrust'], units='N')
@declare_parameter('w', targets=['w'], units='m/s')

class sstoODEconstraint(Group):

    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            mu:         lunar gravitational parameter (m^3/s^2)
            F:          thrust (N)
            w:          exaust velocity (m/s)
            R:          lunar radius (m)
            hmin:       minimum safe altitude (m)
            mc:         constraint shape (-)
        """
        
        self.options.declare('num_nodes', types=int)
        
        self.options.declare('mu', types=float)
        self.options.declare('F', types=float)
        self.options.declare('w', types=float)
        
        self.options.declare('R', types=float)
        self.options.declare('hmin', types=float)
        self.options.declare('mc', types=float)

    def setup(self):
        
        """
        add the subsystems that describe the ODEs for the varying thrust ascent trajectory and the equation for the constraint
        """
        
        nn = self.options['num_nodes']
        mu = self.options['mu']
        F = self.options['F']
        w = self.options['w']
        R = self.options['R']
        hmin = self.options['hmin']
        mc = self.options['mc']

        self.add_subsystem(name='ssto_ode',
                           subsys=sstoODEthrottle(num_nodes=nn, mu=mu, F=F, w=w),
                           promotes_inputs=['r', 'u', 'v', 'm', 'alpha', 'k', 'w', 'thrust'],
                           promotes_outputs=['rdot', 'thetadot', 'udot', 'vdot', 'mdot'])

        self.add_subsystem(name='constraint',
                           subsys=ConstraintComponent(num_nodes=nn, R=R, hmin=hmin, mc=mc),
                           promotes_inputs=['r', 'theta'],
                           promotes_outputs=['dist_safe'])