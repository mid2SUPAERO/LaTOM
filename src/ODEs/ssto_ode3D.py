# -*- coding: utf-8 -*-
"""
The script defines the equations of motion (EOMs) and their partial derivatives required to describe a three-dimensional ascent trajectory
in cartesian coordinates as well as the equations to compute the spacecraft radial position, specific angular momentum vector, eccentricity
vector and thrust direction unit vector magnitude

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent, Group
from dymos import declare_time, declare_state, declare_parameter

#equations of motion in cartesian coordinates

@declare_time(units='s')

@declare_state('x', rate_source='xdot', targets=['x'], units='m')
@declare_state('y', rate_source='ydot', targets=['y'], units='m')
@declare_state('z', rate_source='zdot', targets=['z'], units='m')

@declare_state('vx', rate_source='vxdot', targets=['vx'], units='m/s')
@declare_state('vy', rate_source='vydot', targets=['vy'], units='m/s')
@declare_state('vz', rate_source='vzdot', targets=['vz'], units='m/s')

@declare_state('m', rate_source='mdot', targets=['m'], units='kg')

@declare_parameter('ux', targets=['ux'])
@declare_parameter('uy', targets=['uy'])
@declare_parameter('uz', targets=['uz'])

@declare_parameter('T', targets=['T'], units='N')
@declare_parameter('Isp', targets=['Isp'], units='s')

class sstoODE_3D(ExplicitComponent):

    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            mu:         lunar gravitational parameter (m^3/s^2)
            g0:         standard gravitational acceleration (m/s^2)
            Isp:        specific impulse (s)
        """
        
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)
        self.options.declare('g0', types=float)
        self.options.declare('Isp', types=float)

    def setup(self):
        
        """
        define the equation inputs, outputs and jacobian components
        """
        
        nn = self.options['num_nodes']
        Isp = self.options['Isp']
    
        #inputs (ODE right-hand side)
        self.add_input('x', val=np.zeros(nn), desc='position along x', units='m')
        self.add_input('y', val=np.zeros(nn), desc='position along y', units='m')
        self.add_input('z', val=np.zeros(nn), desc='position along z', units='m')
                
        self.add_input('vx', val=np.zeros(nn), desc='velocity along x', units='m/s')
        self.add_input('vy', val=np.zeros(nn), desc='velocity along y', units='m/s')
        self.add_input('vz', val=np.zeros(nn), desc='velocity along z', units='m/s')
        
        self.add_input('m', val=np.zeros(nn), desc='mass', units='kg')
        
        self.add_input('ux', val=np.zeros(nn), desc='thrust direction unit vector along x')
        self.add_input('uy', val=np.zeros(nn), desc='thrust direction unit vector along y')
        self.add_input('uz', val=np.zeros(nn), desc='thrust direction unit vector along z')
        
        self.add_input('T', val=np.zeros(nn), desc='thrust magnitude', units='N')
        self.add_input('Isp', val=Isp*np.ones(nn), desc='specific impulse', units='s')

        #outputs (ODE left-hand side)
        self.add_output('xdot', val=np.zeros(nn), desc='velocity along x', units='m/s')
        self.add_output('ydot', val=np.zeros(nn), desc='velocity along y', units='m/s')
        self.add_output('zdot', val=np.zeros(nn), desc='velocity along z', units='m/s')
        
        self.add_output('vxdot', val=np.zeros(nn), desc='acceleration along x', units='m/s**2')
        self.add_output('vydot', val=np.zeros(nn), desc='acceleration along y', units='m/s**2')
        self.add_output('vzdot', val=np.zeros(nn), desc='acceleration along z', units='m/s**2')
        
        self.add_output('mdot', val=np.zeros(nn), desc='mass rate', units='kg/s')
        
        #partial derivatives of the ODE outputs respect to the ODE inputs
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='xdot', wrt='vx', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='ydot', wrt='vy', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='zdot', wrt='vz', rows=ar, cols=ar, val=1.0)
        
        self.declare_partials(of='vxdot', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='vxdot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vxdot', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='vxdot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='vxdot', wrt='ux', rows=ar, cols=ar)
        self.declare_partials(of='vxdot', wrt='T', rows=ar, cols=ar)
        
        self.declare_partials(of='vydot', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='vydot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vydot', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='vydot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='vydot', wrt='uy', rows=ar, cols=ar)
        self.declare_partials(of='vydot', wrt='T', rows=ar, cols=ar)
        
        self.declare_partials(of='vzdot', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='vzdot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vzdot', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='vzdot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='vzdot', wrt='uz', rows=ar, cols=ar)
        self.declare_partials(of='vzdot', wrt='T', rows=ar, cols=ar)

        self.declare_partials(of='mdot', wrt='T', rows=ar, cols=ar)
        self.declare_partials(of='mdot', wrt='Isp', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        """
        compute the equation outputs
        """
        
        mu = self.options['mu']
        g0 = self.options['g0']
        
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        vx = inputs['vx']
        vy = inputs['vy']
        vz = inputs['vz']
        m = inputs['m']
        ux = inputs['ux']
        uy = inputs['uy']
        uz = inputs['uz']
        T = inputs['T']
        Isp = inputs['Isp']
        
        r = (x*x + y*y + z*z)**0.5
        
        outputs['xdot'] = vx
        outputs['ydot'] = vy
        outputs['zdot'] = vz
        
        outputs['vxdot'] = T*ux/m - mu*x/r**3
        outputs['vydot'] = T*uy/m - mu*y/r**3
        outputs['vzdot'] = T*uz/m - mu*z/r**3
        
        outputs['mdot'] = - T/(Isp*g0)
        
    def compute_partials(self, inputs, jacobian):
        
        """
        compute the jacobian components
        """

        mu = self.options['mu']
        g0 = self.options['g0']
        
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        m = inputs['m']
        ux = inputs['ux']
        uy = inputs['uy']
        uz = inputs['uz']
        T = inputs['T']
        Isp = inputs['Isp']
        
        r = (x*x + y*y + z*z)**0.5
        
        #partials of vxdot
        jacobian['vxdot', 'x'] = (mu/r**3)*(3*x*x/r**2 - 1.0)
        jacobian['vxdot', 'y'] = 3*mu*x*y/r**5
        jacobian['vxdot', 'z'] = 3*mu*x*z/r**5
        jacobian['vxdot', 'm'] = - T*ux/m**2
        jacobian['vxdot', 'ux'] = T/m
        jacobian['vxdot', 'T'] = ux/m
        
        #partials of vydot
        jacobian['vydot', 'x'] = 3*mu*x*y/r**5
        jacobian['vydot', 'y'] = (mu/r**3)*(3*y*y/r**2 - 1.0)
        jacobian['vydot', 'z'] = 3*mu*y*z/r**5
        jacobian['vydot', 'm'] = - T*uy/m**2
        jacobian['vydot', 'uy'] = T/m
        jacobian['vydot', 'T'] = uy/m

        #partials of vzdot
        jacobian['vzdot', 'x'] = 3*mu*x*z/r**5
        jacobian['vzdot', 'y'] = 3*mu*y*z/r**5
        jacobian['vzdot', 'z'] = (mu/r**3)*(3*z*z/r**2 - 1.0)
        jacobian['vzdot', 'm'] = - T*uz/m**2
        jacobian['vzdot', 'uz'] = T/m
        jacobian['vzdot', 'T'] = uz/m

        #partials of mdot
        jacobian['mdot', 'T'] = - 1.0/(Isp*g0)
        jacobian['mdot', 'Isp'] = T/(Isp*Isp*g0)
        
        
#equations to compute the radius, specific angular momentum vector, eccentricity vector and thrust direction unit vector magnitude from the EOMs inputs
        
class sstoHE_3D(ExplicitComponent):
    
    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            mu:         lunar gravitational parameter (m^3/s^2)
        """
        
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)

    def setup(self):
        
        """
        define the equation inputs, outputs and jacobian components
        """
        
        nn = self.options['num_nodes']

        #inputs
        self.add_input('x', val=np.zeros(nn), desc='position along x', units='m')
        self.add_input('y', val=np.zeros(nn), desc='position along y', units='m')
        self.add_input('z', val=np.zeros(nn), desc='position along z', units='m')
        
        self.add_input('vx', val=np.zeros(nn), desc='velocity along x', units='m/s')
        self.add_input('vy', val=np.zeros(nn), desc='velocity along y', units='m/s')
        self.add_input('vz', val=np.zeros(nn), desc='velocity along z', units='m/s')
        
        self.add_input('ux', val=np.zeros(nn), desc='thrust direction unit vector along x')
        self.add_input('uy', val=np.zeros(nn), desc='thrust direction unit vector along y')
        self.add_input('uz', val=np.zeros(nn), desc='thrust direction unit vector along z')
        
        #outputs
        self.add_output('r', val=np.zeros(nn), desc='position vector norm') 
        self.add_output('hx', val=np.zeros(nn), desc='specific angular momentum along x')
        self.add_output('hy', val=np.zeros(nn), desc='specific angular momentum along y')
        self.add_output('hz', val=np.zeros(nn), desc='specific angular momentum along z')
        self.add_output('ex', val=np.zeros(nn), desc='eccentricity vector x component')
        self.add_output('ey', val=np.zeros(nn), desc='eccentricity vector y component')
        self.add_output('ez', val=np.zeros(nn), desc='eccentricity vector z component')
        self.add_output('u2', val=np.zeros(nn), desc='thrust direction unit vector squared norm')
        
        #partial derivatives of the ODE outputs respect to the ODE inputs
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of='r', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='r', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='r', wrt='z', rows=ar, cols=ar)
        
        self.declare_partials(of='hx', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='hx', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='hx', wrt='vy', rows=ar, cols=ar)
        self.declare_partials(of='hx', wrt='vz', rows=ar, cols=ar)
        
        self.declare_partials(of='hy', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='hy', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='hy', wrt='vx', rows=ar, cols=ar)
        self.declare_partials(of='hy', wrt='vz', rows=ar, cols=ar)
        
        self.declare_partials(of='hz', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='hz', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='hz', wrt='vx', rows=ar, cols=ar)
        self.declare_partials(of='hz', wrt='vy', rows=ar, cols=ar)
        
        self.declare_partials(of='ex', wrt='x', method='cs')
        self.declare_partials(of='ex', wrt='y', method='cs')
        self.declare_partials(of='ex', wrt='z', method='cs')
        self.declare_partials(of='ex', wrt='vx', method='cs')
        self.declare_partials(of='ex', wrt='vy', method='cs')
        self.declare_partials(of='ex', wrt='vz', method='cs')
        
        self.declare_partials(of='ey', wrt='x', method='cs')
        self.declare_partials(of='ey', wrt='y', method='cs')
        self.declare_partials(of='ey', wrt='z', method='cs')
        self.declare_partials(of='ey', wrt='vx', method='cs')
        self.declare_partials(of='ey', wrt='vy', method='cs')
        self.declare_partials(of='ey', wrt='vz', method='cs')
        
        self.declare_partials(of='ez', wrt='x', method='cs')
        self.declare_partials(of='ez', wrt='y', method='cs')
        self.declare_partials(of='ez', wrt='z', method='cs')
        self.declare_partials(of='ez', wrt='vx', method='cs')
        self.declare_partials(of='ez', wrt='vy', method='cs')
        self.declare_partials(of='ez', wrt='vz', method='cs')
        
        self.declare_partials(of='u2', wrt='ux', rows=ar, cols=ar)
        self.declare_partials(of='u2', wrt='uy', rows=ar, cols=ar)
        self.declare_partials(of='u2', wrt='uz', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        """
        compute the equation outputs
        """
        
        mu = self.options['mu']
        
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        
        vx = inputs['vx']
        vy = inputs['vy']
        vz = inputs['vz']
        
        ux = inputs['ux']
        uy = inputs['uy']
        uz = inputs['uz']
        
        #specific angular momentum vector components
        outputs['hx'] = y*vz - z*vy
        outputs['hy'] = z*vx - x*vz
        outputs['hz'] = x*vy - y*vx
        
        #velocity norm squared
        v2 = vx*vx + vy*vy + vz*vz
        
        #dot product between R and V
        rvr = x*vx + y*vy + z*vz
        
        #position vector norm
        r = (x*x + y*y + z*z)**0.5
        outputs['r'] = r
        
        #eccentricity vector components
        b = v2 - mu/r
        outputs['ex'] = (b*x - rvr*vx)/mu
        outputs['ey'] = (b*y - rvr*vy)/mu
        outputs['ez'] = (b*z - rvr*vz)/mu
        
        #thrust direction unit vector squared norm
        outputs['u2'] = ux*ux + uy*uy + uz*uz

    def compute_partials(self, inputs, jacobian):
        
        """
        compute the jacobian components
        """
        
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        
        vx = inputs['vx']
        vy = inputs['vy']
        vz = inputs['vz']
                
        ux = inputs['ux']
        uy = inputs['uy']
        uz = inputs['uz']
        
        r = (x*x + y*y + z*z)**0.5
        
        jacobian['r', 'x'] = x/r
        jacobian['r', 'y'] = y/r
        jacobian['r', 'z'] = z/r
        
        jacobian['hx', 'y'] = vz
        jacobian['hx', 'z'] = -vy
        jacobian['hx', 'vy'] = -z
        jacobian['hx', 'vz'] = y
        
        jacobian['hy', 'x'] = -vz
        jacobian['hy', 'z'] = vx
        jacobian['hy', 'vx'] = z
        jacobian['hy', 'vz'] = -x
        
        jacobian['hz', 'x'] = vy
        jacobian['hz', 'y'] = -vx
        jacobian['hz', 'vx'] = -y
        jacobian['hz', 'vy'] = x
                
        jacobian['u2', 'ux'] = 2*ux
        jacobian['u2', 'uy'] = 2*uy
        jacobian['u2', 'uz'] = 2*uz
        
#link together the EOMs and the constraints defined in the classes sstoODE_3D and sstoODE_HE
        
@declare_time(units='s')

@declare_state('x', rate_source='xdot', targets=['x'], units='m')
@declare_state('y', rate_source='ydot', targets=['y'], units='m')
@declare_state('z', rate_source='zdot', targets=['z'], units='m')

@declare_state('vx', rate_source='vxdot', targets=['vx'], units='m/s')
@declare_state('vy', rate_source='vydot', targets=['vy'], units='m/s')
@declare_state('vz', rate_source='vzdot', targets=['vz'], units='m/s')

@declare_state('m', rate_source='mdot', targets=['m'], units='kg')

@declare_parameter('ux', targets=['ux'])
@declare_parameter('uy', targets=['uy'])
@declare_parameter('uz', targets=['uz'])

@declare_parameter('T', targets=['T'], units='N')
@declare_parameter('Isp', targets=['Isp'], units='s')
        
class sstoODE_3D_group(Group):

    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            mu:         lunar gravitational parameter (m^3/s^2)
            g0:         standard gravitational acceleration (m/s^2)
            Isp:        specific impulse (s)
        """
        
        self.options.declare('num_nodes', types=int)
        self.options.declare('mu', types=float)
        self.options.declare('g0', types=float)
        self.options.declare('Isp', types=float)
        
    def setup(self):
        
        """
        add the subsystems that describe the ODEs for the varying thrust ascent trajectory and the relative
        constraints
        """
        
        nn = self.options['num_nodes']
        mu = self.options['mu']
        g0 = self.options['g0']
        Isp = self.options['Isp']
        
        self.add_subsystem(name='ode_3D',
                           subsys=sstoODE_3D(num_nodes=nn, mu=mu, g0=g0, Isp=Isp),
                           promotes_inputs=['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'ux', 'uy', 'uz', 'T', 'Isp'],
                           promotes_outputs=['xdot', 'ydot', 'zdot', 'vxdot', 'vydot', 'vzdot', 'mdot'])
        
        self.add_subsystem(name='he_3D',
                           subsys=sstoHE_3D(num_nodes=nn, mu=mu),
                           promotes_inputs=['x', 'y', 'z', 'vx', 'vy', 'vz', 'ux', 'uy', 'uz'],
                           promotes_outputs=['hx', 'hy', 'hz', 'ex', 'ey', 'ez', 'r', 'u2'])