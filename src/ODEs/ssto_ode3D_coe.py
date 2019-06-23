# -*- coding: utf-8 -*-
"""
The script defines the equations to compute the spacecraft COEs from the EOMs inputs and add them together in an OpenMDAO Group instance

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

from __future__ import print_function, division, absolute_import

#adjust path
import sys

if '..' not in sys.path:
    sys.path.append('..')

import numpy as np

from openmdao.api import ExplicitComponent, Group
from dymos import declare_time, declare_state, declare_parameter

from ODEs.ssto_ode3D import sstoODE_3D

#link the spacecraft EOMs and COEs

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
        
class sstoODE_3D_group_coe(Group):

    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            eps:        smallest number such that 1.0 + eps != 1.0
            mu:         lunar gravitational parameter (m^3/s^2)
            g0:         standard gravitational acceleration (m/s^2)
            Isp:        specific impulse (s)
        """
        
        self.options.declare('num_nodes', types=int)
        self.options.declare('eps', types=float)
        self.options.declare('mu', types=float)
        self.options.declare('g0', types=float)
        self.options.declare('Isp', types=float)
        
    def setup(self):
        
        """
        add the subsystems that describe the ODEs for the varying thrust ascent trajectory and the relative
        constraints
        """
        
        nn = self.options['num_nodes']
        eps = self.options['eps']
        mu = self.options['mu']
        g0 = self.options['g0']
        Isp = self.options['Isp']
        
        self.add_subsystem(name='ode_3D',
                           subsys=sstoODE_3D(num_nodes=nn, mu=mu, g0=g0, Isp=Isp),
                           promotes_inputs=['x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'ux', 'uy', 'uz', 'T', 'Isp'],
                           promotes_outputs=['xdot', 'ydot', 'zdot', 'vxdot', 'vydot', 'vzdot', 'mdot'])
        
        self.add_subsystem(name='coe_3D',
                           subsys=sstoCOE_3D(num_nodes=nn, eps=eps, mu=mu),
                           promotes_inputs=['x', 'y', 'z', 'vx', 'vy', 'vz', 'ux', 'uy', 'uz'],
                           promotes_outputs=['r', 'a', 'e', 'i', 'raan', 'w', 'ta', 'u2'])

#spacecraft COEs

class sstoCOE_3D(ExplicitComponent):
    
    def initialize(self):
        
        """
        initialize an instance specifying the followings:
            num_nodes:  number of discrete points in which the constraint has to be evaluated
            eps:        smallest number such that 1.0 + eps != 1.0
            mu:         lunar gravitational parameter (m^3/s^2)
        """
        
        self.options.declare('num_nodes', types=int)
        self.options.declare('eps', types=float)
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
        self.add_output('a', val=np.zeros(nn), desc='semimajor axis')
        self.add_output('e', val=np.zeros(nn), desc='eccentricity')
        self.add_output('i', val=np.zeros(nn), desc='inclination')
        self.add_output('raan', val=np.zeros(nn), desc='right ascension of the ascending node')
        self.add_output('w', val=np.zeros(nn), desc='argument of perigee')
        self.add_output('ta', val=np.zeros(nn), desc='true anomaly')
        self.add_output('u2', val=np.zeros(nn), desc='thrust direction unit vector squared norm')
        
        #partial derivatives of the ODE outputs respect to the ODE inputs
        ar = np.arange(self.options['num_nodes'])
        
        self.declare_partials(of='r', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='r', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='r', wrt='z', rows=ar, cols=ar)
        
        self.declare_partials(of='a', wrt='x', method='cs')
        self.declare_partials(of='a', wrt='y', method='cs')
        self.declare_partials(of='a', wrt='z', method='cs')
        self.declare_partials(of='a', wrt='vx', method='cs')
        self.declare_partials(of='a', wrt='vy', method='cs')
        self.declare_partials(of='a', wrt='vz', method='cs')
        
        self.declare_partials(of='e', wrt='x', method='cs')
        self.declare_partials(of='e', wrt='y', method='cs')
        self.declare_partials(of='e', wrt='z', method='cs')
        self.declare_partials(of='e', wrt='vx', method='cs')
        self.declare_partials(of='e', wrt='vy', method='cs')
        self.declare_partials(of='e', wrt='vz', method='cs')
        
        self.declare_partials(of='i', wrt='x', method='cs')
        self.declare_partials(of='i', wrt='y', method='cs')
        self.declare_partials(of='i', wrt='z', method='cs')
        self.declare_partials(of='i', wrt='vx', method='cs')
        self.declare_partials(of='i', wrt='vy', method='cs')
        self.declare_partials(of='i', wrt='vz', method='cs')
        
        self.declare_partials(of='raan', wrt='x', method='cs')
        self.declare_partials(of='raan', wrt='y', method='cs')
        self.declare_partials(of='raan', wrt='z', method='cs')
        self.declare_partials(of='raan', wrt='vx', method='cs')
        self.declare_partials(of='raan', wrt='vy', method='cs')
        self.declare_partials(of='raan', wrt='vz', method='cs')
        
        self.declare_partials(of='w', wrt='x', method='cs')
        self.declare_partials(of='w', wrt='y', method='cs')
        self.declare_partials(of='w', wrt='z', method='cs')
        self.declare_partials(of='w', wrt='vx', method='cs')
        self.declare_partials(of='w', wrt='vy', method='cs')
        self.declare_partials(of='w', wrt='vz', method='cs')
        
        self.declare_partials(of='ta', wrt='x', method='cs')
        self.declare_partials(of='ta', wrt='y', method='cs')
        self.declare_partials(of='ta', wrt='z', method='cs')
        self.declare_partials(of='ta', wrt='vx', method='cs')
        self.declare_partials(of='ta', wrt='vy', method='cs')
        self.declare_partials(of='ta', wrt='vz', method='cs')
        
        self.declare_partials(of='u2', wrt='ux', rows=ar, cols=ar)
        self.declare_partials(of='u2', wrt='uy', rows=ar, cols=ar)
        self.declare_partials(of='u2', wrt='uz', rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        
        """
        compute the equation outputs
        """
        
        nn = self.options['num_nodes']
        mu = self.options['mu']
        eps = self.options['eps']
        
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        
        vx = inputs['vx']
        vy = inputs['vy']
        vz = inputs['vz']
        
        ux = inputs['ux']
        uy = inputs['uy']
        uz = inputs['uz']
        
        #specific angular momentum vector and norm
        hx = y*vz - z*vy
        hy = z*vx - x*vz
        hz = x*vy - y*vx
        h = (hx*hx + hy*hy + hz*hz)**0.5
        
        #node line vector and norm
        nx = -hy
        ny = hx
        n = (nx*nx + ny*ny)**0.5
        
        #velocity norm squared
        v2 = vx*vx + vy*vy + vz*vz
        
        #position vector norm
        r = (x*x + y*y + z*z)**0.5
        
        #dot product between R and V
        rvr = x*vx + y*vy + z*vz
        
        #eccentricity vector and norm
        b = v2 - mu/r
        ex = (b*x - rvr*vx)/mu
        ey = (b*y - rvr*vy)/mu
        ez = (b*z - rvr*vz)/mu
        e = (ex*ex + ey*ey + ez*ez)**0.5
        
        #semimajor axis
        a = h*h/mu/(1-e*e)
        
        #inclination
        i = np.arccos(hz/h)
        
        raan = np.zeros(nn)
        w = np.zeros(nn)
        ta = np.zeros(nn)
        
        for j in range(nn):    
            #right ascension of the ascending node
            if i[j]>=eps: #inclined orbit
                raan[j] = np.arccos(nx[j]/n[j])
                if ny[j]<0.0:
                    raan[j] = 2*np.pi - raan[j]
            else: #equatorial orbit
                raan[j] = 0.0
                
            #argument of perigee (rad)
            if e[j]>=eps: #elliptical orbit
                if i[j]>=eps: #inclined orbit
                    w[j] = np.arccos((nx[j]*ex[j] + ny[j]*ey[j])/n[j]/e[j])
                    if ez[j]<0.0:
                        w[j] = 2*np.pi - w[j]
                else: #equatorial orbit
                    w[j] = np.arccos(ex[j]/e[j])
                    if ey[j]<0.0:
                        w[j] = 2*np.pi - w[j]
            else: #circular orbit
                w[j] = 0.0
                
            #true anomaly (rad)
            if e[j]>=eps: #elliptical orbit
                ta[j] = np.arccos((ex[j]*x[j] + ey[j]*y[j] + ez[j]*z[j])/e[j]/r[j])
                if rvr[j]<0.0:
                    ta[j] = 2*np.pi - ta[j]
            else: #circular orbit
                if i[j]>=eps: #inclined orbit
                    ta[j] = np.arccos((nx[j]*x[j] + ny[j]*y[j])/n[j]/r[j])
                    if z[j]<0.0:
                        ta[j] = 2*np.pi - ta[j]
                else: #equatorial orbit
                    ta[j] = np.arccos(x[j]/r[j])
                    if y[j]<0.0:
                        ta[j] = 2*np.pi - ta[j]
        
        #classical orbital elements
        outputs['r'] = r
        outputs['a'] = a
        outputs['e'] = e
        outputs['i'] = i
        outputs['raan'] = raan
        outputs['w'] = w
        outputs['ta'] = ta

        #thrust direction unit vector squared norm
        outputs['u2'] = ux*ux + uy*uy + uz*uz

    def compute_partials(self, inputs, jacobian):
        
        """
        compute the jacobian components
        """
        
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
                
        ux = inputs['ux']
        uy = inputs['uy']
        uz = inputs['uz']
        
        r = (x*x + y*y + z*z)**0.5
        
        jacobian['r', 'x'] = x/r
        jacobian['r', 'y'] = y/r
        jacobian['r', 'z'] = z/r
                
        jacobian['u2', 'ux'] = 2*ux
        jacobian['u2', 'uy'] = 2*uy
        jacobian['u2', 'uz'] = 2*uz