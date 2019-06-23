# -*- coding: utf-8 -*-
"""
The script defines the equation to describe the path constraint imposed on the ascent trajectory
to avoid collisions with the lunar geographical shapes

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent

class ConstraintComponent(ExplicitComponent):
    
    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            R:          lunar radius (m)
            hmin:       minimum safe altitude (m)
            mc:         constraint shape (-)
        """
        
        self.options.declare('num_nodes', types=int)
        self.options.declare('R', types=float)
        self.options.declare('hmin', types=float)
        self.options.declare('mc', types=float)

    def setup(self):
        
        """
        define the equation inputs, outputs and jacobian components
        """
        
        nn = self.options['num_nodes']

        #inputs
        self.add_input('r', val=np.zeros(nn), desc='orbit radius', units='m')
        self.add_input('theta', val=np.zeros(nn), desc='true anomaly', units='rad')
        
        #output
        self.add_output('dist_safe', val=np.zeros(nn), desc='safe distance')

        #jacobian
        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='dist_safe', wrt='r', rows=ar, cols=ar, val=1)
        self.declare_partials(of='dist_safe', wrt='theta', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        
        """
        compute the equation outputs
        """
        
        R = self.options['R']
        hmin = self.options['hmin']
        mc = self.options['mc']
        r = inputs['r']
        theta = inputs['theta']

        outputs['dist_safe'] = r - (R + hmin*R*theta/(R*theta + hmin/mc))

    def compute_partials(self, inputs, jacobian):
        
        """
        compute the jacobian components
        """
        
        R = self.options['R']
        hmin = self.options['hmin']
        mc = self.options['mc']
        theta = inputs['theta']
        
        jacobian['dist_safe', 'theta'] = -hmin**2*R*mc/(hmin + mc*R*theta)**2