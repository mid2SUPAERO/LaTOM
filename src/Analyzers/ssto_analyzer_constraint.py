# -*- coding: utf-8 -*-
"""
The script defines the class required to set up, solve and display the results for the optimal control problem of finding the most
fuel-efficient ascent trajectory from the Moon surface to a specified LLO with variable thrust and constrained minimum safe altitude

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

#adjust path
import sys

if '..' not in sys.path:
    sys.path.append('..')

#import required modules

import numpy as np
from time import time

import matplotlib.pyplot as plt

from Optimizers.ssto_optimizer_constraint import sstoOptConstraint
from Analyzers.ssto_analyzer import sstoAnalyzerThrottle

class sstoAnalyzerConstraint(sstoAnalyzerThrottle): #single phase with constant thrust
    
    def __init__(self, const, settings):
        
        """
        creates a new class instance given the following inputs:
            
            const, a dictionary with the following keys:
            
                g0:             standard gravity acceleration (m/s^2)
                mu:             lunar standard gravitational parameter (m^3/s^2)
                R:              lunar radius (m)
                H:              target LLO altitude (lunar_radii or m)
                m0:             initial spacecraft mass (kg)
                lunar_radii:    orbit altitude H expressed in lunar radii (True) or in meters (False)
                hmin:           minimum safe altitude (m)
                mc:             path constraint shape
                
            settings, a dictionary with the following keys:
                
                solver:                 NLP solver
                tol:                    NLP stopping tolerance
                maxiter:                maximum number of iterations
                transcription:          transcription method
                num_seg:                number of segments in time in which the phase is divided
                transcription_order:    order of the states interpolating polynomials whitin each segment
                scalers:                optimizer scalers for (time, r, theta, u, v, m)
                defect_scalers:         optimizer defect scalers for (time, r, theta, u, v, m)
                top_level_jacobian:     jacobian format used by openMDAO
                dynamic_simul_derivs:   accounts for the sparsity while computing the jacobian
                compressed:             compressed transcription to reduce the number of variables
                debug:                  check partial derivatives defined in the ODEs
                acc_guess:              use or not an accurate initial guess
                alpha_rate2_cont:       impose or not the continuity of the second derivative of alpha
                duration_bounds:        impose or not the phase duration bounds specified in t0
                exp_sim:                explicit integration using the optimal control profile
            
        and defining the following quantities:
            
            H:      target LLO altitude (m)
            rf:     target LLO radius (m)
            vf:     target LLO tangential velocity (m/s)
            W0:     initial spacecraft weight on the Moon surface (N)
            bcs:    array with all the required boundary conditions
        """
                
        sstoAnalyzerThrottle.__init__(self, const, settings)
        self.hmin = const['hmin'] #minimim safe altitude (m)
        self.mc = const['mc'] #path constraint slope
        
    def hohmann_tof(self):
        
        """
        computes the time of flight for an Hohmann transfer from the Moon surface to a specified LLO
        """
        
        a_hoh = self.R + self.H/2 #semimajor axis
        tof_hoh = np.pi/(self.mu**0.5)*(a_hoh**1.5)
        
        return tof_hoh
        
    def set_params(self, Isp, twr, klim, t0=None):
        
        """
        defines the Isp, twr, w and F values for which solve the optimization problem,
        the throttle limits and an initial guess for the required time of flight where:
            
            Isp:        specific impulse (s)
            twr:        thrust/initial weight ratio
            w:          exaust velocity (m/s)
            F:          thrust (N)
            klim:       throttle limits (kmin, kmax)
            t0:         time of flight initial guess and bounds as (lb, tof, ub) (s)
        """
        
        if t0 is None:
            t_guess = self.hohmann_tof()
            t0 = np.array([t_guess-5e2, t_guess, t_guess+5e2])
        
        sstoAnalyzerThrottle.set_params(self, Isp, twr, klim, t0)

    def set_results(self, d):
        
        """
        saves as class attributes the results for an already computed trajectory stored in a given dictionary d
        """
        sstoAnalyzerThrottle.set_results(self, d)
        self.hmin = d['hmin']
        self.mc = d['mc']
        
    def set_pbm(self, p):
        
        """
        returns an OpenMDAO problem class instance for the current optimal control problem
        """
        
        self.p = p
        
    def get_results(self):
        
        """
        returns a dictionary with the parameters and the profiles for the optimal ascent trajectory
        """
                
        d = sstoAnalyzerThrottle.get_results(self)
        d1 = {'hmin':self.hmin, 'mc':self.mc}
        d.update(d1)
        
        return d
    
    def get_history(self, p):
        
        """
        retreives the states and control histories from a given preoblem p and returns them
        in a unique array hist defined as [t, r, theta, u, v, m, alpha, k, dist_safe]
        """
        
        hist0 = sstoAnalyzerThrottle.get_history(self, p)
        ds = p.get_val('phase0.timeseries.dist_safe').T
        hist = np.concatenate((hist0, ds))
        
        return hist
    
    def alt_plot(self, h, h_exp, colors=('r', 'b')):
        
        """
        plots the altitude profile vs angle
        """
        
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        if h_exp is not None:
            alt_exp = (h_exp[1]-self.R)/1e3 #altitude (km)
            ta_exp = h_exp[2]*180/np.pi #angle (deg)
            ax.plot(ta_exp, alt_exp, color='g')
        
        alt = (h[1]-self.R)/1e3 #altitude (km)
        ds = alt-h[8]/1e3 #constraint (km)
        ta = h[2]*180/np.pi #angle (deg)
        k = h[7] #throttle values
        
        ax.plot(ta[k!=0], alt[k!=0], 'o', color=colors[0], label='powered', zorder=2)
        ax.plot(ta[k==0], alt[k==0], 'o', color=colors[1], label='coasting', zorder=1)
        ax.plot(ta, ds, color='k', label='constraint', zorder=3)
        ax.set_xlabel('theta (deg)')
        ax.set_ylabel('h (km)')
        ax.set_title('Altitude profile')
        ax.grid()
        ax.legend(loc='best')
        
    def trajectory_plot(self, h, rf):
        
        """
        plots the ascent trajectory in the xy plane
        """
    
        #Moon and final orbit radius (km)
        r_moon = self.R/1e3
        r_orbit = rf/1e3
        
        #ascent trajectory points (km)
        x_ascent = h[1]*np.cos(h[2])/1e3
        y_ascent = h[1]*np.sin(h[2])/1e3
        
        xc = (h[1]-h[8])*np.cos(h[2])/1e3
        yc = (h[1]-h[8])*np.sin(h[2])/1e3
            
        #axis limits and ticks
        limit = np.ceil(r_orbit/1e3)*1e3
        ticks = np.linspace(-limit, limit, 9)
        
        #angle vector to plot the surface of the Moon and the final orbit (rad)
        theta = np.linspace(0, 2*np.pi, 200)
    
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(r_moon*np.cos(theta), r_moon*np.sin(theta), label='Moon surface')
        ax.plot(r_orbit*np.cos(theta), r_orbit*np.sin(theta), label='Target orbit')
        ax.plot(x_ascent, y_ascent, label='Ascent trajectory')
        ax.plot(xc, yc, label='Constraint')
        ax.set_aspect('equal')
        ax.grid()
        ax.legend(bbox_to_anchor=(1, 1), loc=2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title('Optimal ascent trajectory')
        ax.tick_params(axis='x', rotation=60)
        
    def set_optimizer(self):
        
        """
        returns an instance of the class sstoOptimizer
        """
        
        return sstoOptConstraint(self.const, self.settings)