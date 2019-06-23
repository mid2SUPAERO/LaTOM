#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:38:36 2019

@authors:
    Alberto Fossa'
    Giuliana Miceli

This script defines a class to compute an accurate initial guess for the optimal control problem
of finding the most fuel-efficient lunar ascent trajectory from the lunar surface to a specified
Low Lunar Orbit (LLO) with variable thrust
"""

#adjust path
import sys

if '..' not in sys.path:
    sys.path.append('..')

#import required modules
import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt

class hohmannTransfer():
    
    def __init__(self, mu, R, H):
        
        """
        defines the parmenters of an Hohmann transfer between the Moon surface and the specified LLO
        """
        
        self.mu = mu #lunar standard gravitational parameter (m^3/s^2)
        self.R = R #lunar radius (m)
        self.H = H #target orbit altitude (m)
        
        self.a = self.R + self.H/2 #semimajor axis (m)
        self.e = self.H/(2*self.R + self.H) #eccentricity
        self.h = (self.mu*self.a*(1 - self.e**2))**0.5 #angular momentum (m^2/s)
        self.tof = np.pi/self.mu**0.5*self.a**1.5 #time of flight (s)

        self.vcp=(self.mu/self.R)**0.5 #initial circular velocity (m/s)
        self.vca=(self.mu/(self.R+self.H))**0.5 #final circular velocity (m/s)
        
        self.vp=(2*self.mu*(self.R + self.H)/(self.R*(2*self.R+self.H)))**0.5 #periselene velocity (m/s)
        self.va=(2*self.mu*self.R/((self.R+self.H)*(2*self.R+self.H)))**0.5 #aposelene velocity (m/s)
        
        self.dvp=self.vp-self.vcp #periselene dV (m/s)
        self.dva=self.vca-self.va #aposelene dV (m/s)
        
    def get_vp(self):
        
        """
        return the required periselene velocity to perform the Hohmann transfer
        """
        
        return self.vp
    
    def get_tof(self):
        
        """
        returns the Hohmann transfer time of flight
        """
        
        return self.tof
    
    def compute_states(self, t=None):
        
        """
        computes the timeseries of theta, r, u, v
        """
        
        if t is None:
            t = np.linspace(0,self.tof)
            
        nb_nodes = len(t)
        n = (self.mu/self.a**3)**0.5 #mean motion (rad/s)
        E0 = np.linspace(0,np.pi,nb_nodes) #eccentric anomaly initial guess
        
        print("\nSolving Kepler's equation using Scipy root function")
        
        sol = spo.root(self.kepler, E0, args=(self.e, n, t), tol=1e-12) #solve the Kepler's equation for E
        
        print("output:", sol['message'])
        
        self.E = sol.x #eccentric anomaly
        self.theta = 2*np.arctan(((1 + self.e)/(1 - self.e))**0.5*np.tan(self.E/2)) #true anomaly (rad)
        self.r = self.a*(1 - self.e**2)/(1 + self.e*np.cos(self.theta)) #orbit radius (m)
        self.u = self.mu/self.h*self.e*np.sin(self.theta) #radial velocity (m/s)
        self.v = self.mu/self.h*(1 + self.e*np.cos(self.theta)) #tangential velocity (m/s)
        
        return sol
        
    def kepler(self, E, e, n, t):
        
        """
        Kepler's equation
        """
        
        return E - e*np.sin(E) - n*t
    
    
class surfaceGrazing():
    
    def __init__(self, mu, R, vp, m0, F, w):
        
        """
        defines the paramenters for the initial surface grazing to acquire the required periselene tangential
        velocity to perform the Hohmann transfer
        """
        
        self.mu = mu #lunar standard gravitational parameter (m^3/s^2)
        self.R = R #lunar radius (m)
        self.vp = vp #periselene velocity (m/s)
        self.m0 = m0 #initial spacecraft mass (kg)
        self.F = F #thrust (N)
        self.w = w #exaust velocity (m/s)
        
    def get_tof(self):
        
        """
        returns the surface grazing time of flight
        """
        
        return self.tof
            
    def compute_tof(self):
        
        """
        computes the required time of flight to achieve vp
        """
        
        print('\nComputing time of flight for initial powered trajectory at constant R using Scipy solve_ivp function')
        
        sol = spi.solve_ivp(fun = lambda v, t: self.dt_dv(v, t, self.mu, self.R, self.m0, self.F, self.w),
                            t_span=(0, self.vp), y0=[0], rtol=1e-9, atol=1e-12)
        
        self.tof = sol.y[-1,-1] #sufrace grazing time of flight (s)
         
        #y, sol = spi.odeint(self.dt_dv, y0=[0], t=[0, self.vp], args=(self.mu, self.R, self.m0, self.F, self.w),
        #                    full_output=1, rtol=1e-9, atol=1e-12, tfirst=True)
        #self.tof = y[-1,-1] #sufrace grazing time of flight (s)
        
        print('output:', sol['message'])
        
        return sol
        
    def compute_states(self, t_eval):
        
        """
        computes the timeseries of theta, v, m, alpha
        """
        
        print('\nIntegrating ODEs for initial powered trajectory at constant R...')
        
        try:
            sol = spi.solve_ivp(fun = lambda t, x: self.dx_dt(t, x, self.mu, self.R, self.m0, self.F, self.w),
                                t_span=(0, self.tof), y0=[0, 0], t_eval=t_eval, rtol=1e-9, atol=1e-12)
        
            print('using Scipy solve_ivp function')
            
            self.t = sol.t #time (s)
            self.theta = sol.y[0] #true anomaly timeseries (rad)
            self.v = sol.y[1] #tangential velocity timeseries (m/s)
            
        except:
            print('time vector not strictly monotonically increasing, using Scipy odeint function')
            
            y, sol = spi.odeint(self.dx_dt, y0=[0, 0], t=t_eval, args=(self.mu, self.R, self.m0, self.F, self.w),
                                full_output=1, rtol=1e-9, atol=1e-12, tfirst=True)
            self.t = t_eval #time (s)
            self.theta = y[:,0] #true anomaly timeseries (rad)
            self.v = y[:,1] #tangential velocity timeseries (m/s)
        
        print('output:', sol['message'])
        
        self.m = self.m0 - (self.F/self.w)*self.t #spacecraft mass timeseries (kg)
        
        v_dot = self.dv_dt(self.t, self.v, self.mu, self.R, self.m0, self.F, self.w)
        num = self.mu/self.R**2 - self.v**2/self.R
        self.alpha = np.arctan2(num,v_dot) #thrust direction timeseries (rad)
        
        return sol
    
    def dt_dv(self, v, t, mu, R, m0, F, w):
        
        """
        ODE dt/dv = f(v,t,params)
        """
        
        dt_dv = 1/self.dv_dt(t, v, mu, R, m0, F, w)
        
        return dt_dv
    
    def dv_dt(self, t, v, mu, R, m0, F, w):
        
        """
        ODE dv/dt = f(t,v,params)
        """
        
        dv_dt = ((F/(m0-(F/w)*t))**2 - (mu/R**2-v**2/R)**2)**0.5
        
        return dv_dt
    
    def dx_dt(self, t, x, mu, R, m0, F, w):
        
        """
        ODE dx/dt = f(t,x,params) where x=[theta, v]
        """
        
        x0_dot = x[1]/R
        x1_dot = self.dv_dt(t, x[1], mu, R, m0, F, w)
        
        return [x0_dot, x1_dot]

class sstoGuess():
    
    def __init__(self, const, F, w):
        
        """
        defines the paramenters for a transfer trajectory between the Moon surface and a specified LLO
        approximated as a two-phases trajectory with an initial surface grazing to acquire the required
        periselene tangential velocity and an Hohmann transfer
        """
        
        self.g0 = const['g0'] #standard gravity acceleration (m/s^2)
        self.mu = const['mu'] #lunar standard gravitational parameter (m^3/s^2)
        self.R = const['R'] #lunar radius (m)
        self.H = const['H'] #target orbit altitude (lunar radii or m)
        
        if const['lunar_radii']:
            self.H = self.H*self.R #target orbit altitude (m)
            
        self.m0 = const['m0'] #initial spacecraft mass (kg)
        
        self.F = F #thrust (N)
        self.w = w #exaust velocity (m/s)
        
        self.ht = hohmannTransfer(self.mu, self.R, self.H) #hohmannTransfer class instance
        self.vp = self.ht.get_vp() #periselene tangential velocity (m/s)
        
        self.sg = surfaceGrazing(self.mu, self.R, self.vp, self.m0, self.F, self.w) #surfaceGrazing class instance
        
    def compute_tof(self):
        
        sol_t = self.sg.compute_tof()
        
        self.tof_sg = self.sg.get_tof() #surface grazing time of flight (s)
        self.tof_hoh = self.ht.get_tof() #Hohmann transfer time of flight (s)
        self.tof = self.tof_sg + self.tof_hoh #total time of flight (s)
        
        return self.tof, sol_t
        
    def compute_time_vectors(self, t=None):
        
        if t is None:
            self.t = np.linspace(0, self.tof)
        else:
            self.t = t
            
        self.t_sg = self.t[self.t<=self.tof_sg] #surface grazing time vector (s)
        self.t_hoh = self.t[self.t>self.tof_sg] #Hohmann transfer time vector (s)
        
        self.nb_sg = len(self.t_sg)
        self.nb_hoh = len(self.t_hoh)
        
    def compute_trajectory(self, t=None):
        
        """
        computes the timeseries for the whole trajectory
        """
        
        self.compute_time_vectors(t)
        
        sol_sg = self.sg.compute_states(self.t_sg)
        sol_ht = self.ht.compute_states(self.t_hoh-self.tof_sg)
        
        self.r = np.concatenate((self.R*np.ones(self.nb_sg), self.ht.r))
        self.theta = np.concatenate((self.sg.theta, self.ht.theta+self.sg.theta[-1]))
        self.u = np.concatenate((np.zeros(self.nb_sg), self.ht.u))
        self.v = np.concatenate((self.sg.v, self.ht.v))
        
        self.m_hoh = self.sg.m[-1] #spacecraft mass during Hohmann transfer (kg)
        self.m_final = self.m_hoh*np.exp(-self.ht.dva/self.w) #spacecraft mass after insertion burn (kg)
        self.m = np.concatenate((self.sg.m, self.m_hoh*np.ones(self.nb_hoh-1), [self.m_final]))
        self.alpha = np.concatenate((self.sg.alpha, np.zeros(self.nb_hoh)))
        self.k = np.concatenate((np.ones(self.nb_sg), np.zeros(self.nb_hoh-1), [1]))
        
        self.hist = np.array([self.t, self.r, self.theta, self.u, self.v, self.m, self.alpha, self.k])
        
        return sol_sg, sol_ht
    
    def plot_all(self):
        
        from Analyzers.ssto_analyzer import sstoAnalyzerThrottle
        
        rf = self.R+self.H #final orbit radius (m)
        
        sstoAnalyzerThrottle.states_plot(self, self.hist)
        sstoAnalyzerThrottle.alt_plot(self, self.hist)
        sstoAnalyzerThrottle.control_plot(self, self.hist)
        sstoAnalyzerThrottle.trajectory_plot(self, self.hist, rf)
        
        plt.show()
        