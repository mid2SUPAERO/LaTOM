# -*- coding: utf-8 -*-
"""
the script defines the classes required to set up, solve and display the results for the optimal control problem of finding the most
fuel-efficent, two-phases powered descent trajectory with a final constrained vertical path

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

#adjust path
import sys

if '..' not in sys.path:
    sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from time import time

from Optimizers.descent_optimizer import descentOptimizer2p, descentOptimizerThrottle2p
from Analyzers.descent_analyzer import descentAnalyzer, descentAnalyzerThrust, descentAnalyzerThrottle
from Analyzers.ssto_analyzer import sstoAnalyzer2p, sstoAnalyzerThrottle

class descentAnalyzer2p(descentAnalyzer): #two phases, single parameters descent trajectory
    
    def set_params(self, Isp, twr, hp, hs, t0):
        
        """
        defines the Isp, twr, exaust velocity and thrust as well as the intermediate orbit periapsis altitude,
        the switch altitude and the time of flight initial guess before computing the BCs for the TPBVP
        calling the method deorbit_burn()
        
        inputs:
            
            Isp:    specific impulse (s)
            twr:    thrust/initial weight ratio
            hp:     intermediate orbit periapsis altitude (m)
            hs:     switch altitude (m)
            t0:     time of flight guess (s)
            
        variables stored as class attributes:
            
            rp:     intermediate orbit periapsis radius (m)
            rs:     switch radius between horizontal and vertical braking phase (m)
            v0:     circular parking orbit tangential velocity (m/s)
            vp:     intermediate orbit tangential velocity at periapsis (m/s)
            w:      exaust velocity (m/s)
            F:      thrust (N)
        """
        
        self.hs = hs #switch altitude between horizontal and vertical braking phase (m)
        self.rs = self.R + self.hs #switch radius between horizontal and vertical braking phase (m)
        
        descentAnalyzer.set_params(self, Isp, twr, hp, t0)
        
    def set_bcs(self):
        
        """
        computes the BCs for the TPBVP storing them in the class attribute bcs defined as [r, theta, u, v, m, alpha]
        """
        
        self.bcs = np.array([[self.rp, self.rs, self.rf], [0.0, np.pi/3, np.pi/2], [0.0, self.uf*20, self.uf],
                             [self.vp, 0.0, 0.0], [self.m0, self.m0/3, self.m0/2], [np.pi/2, 3/2*np.pi, 3/2*np.pi]])
        
    def get_history(self, p):
                
        """
        retreives the states and control histories from a given problem p and returns them
        in a unique array hist defined as [t, r, theta, u, v, m, alpha]
        """
        
        #state and control variables in the free-attitude ascent phase
        th, rh, thetah, uh, vh, mh, alphah, nbh = sstoAnalyzer2p.get_history_horiz(self, p)
        histh = np.array([th, rh, thetah, uh, vh, mh, alphah])
        
        #switch time, theta and radial velocity
        ts = th[-1]
        thetas = thetah[-1]
        us = uh[-1]
        
        #state and control variables in the vertical-rise phase
        tv, rv, uv, mv, nbv = sstoAnalyzer2p.get_history_vert(self, p)
        thetav = np.ones(nbv)*thetas
        vv = np.zeros(nbv)
        alphav = np.ones(nbv)*np.pi/2
        histv = np.array([tv, rv, thetav, uv, vv, mv, alphav])
        
        #state and control variables throughout the whole trajectory        
        hist = np.concatenate((histh, histv), 1)
        
        return hist, ts, us, nbh, nbv
    
    def get_results(self):
        
        """
        returns a dictionary with the following keys:
            
            H:              initial circular parking orbit altitude (m)
            hp:             intermediate orbit periapsis altitude (m)
            hs:             switch altitude (m)
            hf:             final altitude (m)
            r0:             initial circular parking orbit radius (m)
            rp:             intermediate orbit periapsis radius (m)
            rs:             switch radius (m)
            rf:             final radius (m)
            a0:             intermediate orbit semimajor axis (m)
            Isp:            specific impulse (s)
            twr:            thrust/initial weight ratio
            F:              thrust (N)
            ts:             switch time (s)
            tof:            time of flight (s)
            m0:             initial spacecraft mass (kg)
            m_final:        final spacecraft mass (kg)
            m_prop:         propellant mass (kg)
            prop_frac:      propellant fraction
            alpha0:         initial thrust direction (rad)
            alphaf:         final thrust direction (rad)
            hist:           optimal trajectory states and controls histories
            ut:             switch radial velocity (m/s)
            nbh:            number of collocation points in the attitude-free phase
            nbv:            number of collocation points in the vertical phase
            
        and optionally:
            
            m_park:         initial spacecraft mass in the parking orbit (kg)
            deorbit_prop:   propellant mass required for de-orbit burn (kg)
            deorbit_dv:     impulsive delta-V required for de-orbit burn (m/s)
        """
        
        d = descentAnalyzer.get_results(self)
        d_transition = {'hs':self.hs, 'rs':self.rs, 'ts':self.ts, 'us':self.us, 'nbh':self.nbh, 'nbv':self.nbv}
        d.update(d_transition)
        
        return d
    
    def set_results(self, d):
        
        """
        saves as class attributes the results for an already computed trajectory stored in a given dictionary d
        """

        self.hs = d['hs']
        self.rs = d['rs']
        self.ts = d['ts']
        self.us = d['us']
        self.nbh = d['nbh']
        self.nbv = d['nbv']
        
        descentAnalyzer.set_results(self, d)

    def display_summary(self):
        
        """
        prints out the following optimal trajectory's parameters:
            
            h0:                 initial circular parking orbit altitude (m)
            hp:                 intermediate orbit periapsis altitude (m)
            Isp:                specific impulse (s)
            twr:                thrust/initial weight ratio
            F:                  thrust (N)
            tof:                time of flight (s)
            m0:                 initial spacecraft mass (kg)
            m_final:            final spacecraft mass (kg)
            m_prop:             propellant mass (kg)
            m_final/m0:         final/initial mass ratio (%)
            prop_frac:          propellant fraction (%)
            alpha0:             initial thrust direction (deg)
            alphaf:             final thrust direction (deg)
            hs:                 switch altitude (km)
            us:                 switch radial velocity (m/s)
            ts:                 switch time (s)
            
        and optionally:
            
            m_park:             initial spacecraft mass in the parking orbit (kg)
            deorbit_prop:       propellant mass required for de-orbit burn (kg)
            deorbit_dv:         impulsive delta-V required for de-orbit burn (m/s)
        """
        
        descentAnalyzer.display_summary(self)
        
        print('{:<50s}{:>24.2f}'.format('transition altitude (km)', self.hs/1e3))
        print('{:<50s}{:>24.2f}'.format('radial velocity at transition (m/s)', self.us))
        print('{:<50s}{:>24.2f}'.format('time at transition (s)', self.ts))
        print('')
        
    def plot_all(self, h, rp, n):
        
        """
        calls the following class methods to display the different plots:
            
            states_plot(h, n, colors, labels):  altitude, angle, radial and tangential velocities vs time
            alt_plot(h, n, colors, labels):     altitude profile vs true anomaly
            control_plot(h):                    thrust direction vs time
            trajectory_plot(h, rp):             parking orbit, intermediate orbit and descent trajectory in the xy plane
        """
        
        colors = ['b', 'r']
        labels = ['attitude-free', 'vertical']
        
        sstoAnalyzer2p.states_plot(self, h, n, colors, labels) #state variables vs time
        sstoAnalyzer2p.alt_plot(self, h, n, colors, labels) #altitude profile vs true anomaly
        self.control_plot(h) #control vs time
        self.trajectory_plot(h, rp) #trajectories in the xy plane
        
        plt.show()
                
    def run_optimizer(self):
        
        """
        creates an instance of the class descentOptimizer2p and solves the optimal control problem
        storing the following results as class attributes:
            
            p:              problem class instance for the optimal transfer trajectory
            hist:           array with the states and control variables history [t, r, theta, u, v, m, alpha]
        """
        
        optimizer = descentOptimizer2p(self.const, self.settings) #create an optimizer instance
        
        execution_start = time() #measure the execution time
        
        failed = optimizer.run_optimizer(self.F, self.w, self.bcs, self.t0) #solve the optimal control problem
                
        execution_end = time()
        elapsed_time = execution_end-execution_start
        
        if not failed:
            
            print("\nOptimization done in " + str(elapsed_time) + " s\n") 
            
            self.p = optimizer.get_pbm()
            self.hist, self.ts, self.us, self.nbh, self.nbv = self.get_history(self.p)        
            self.get_scalars(self.hist)
            
            self.display_summary()
            self.plot_all(self.hist, self.rp, self.nbh)
            
            return self.p
                                    
        else:
            print("\nOptimization failed!\n")
            
            
class descentAnalyzerThrust2p(descentAnalyzer2p): #two phases, multiple thrust values descent trajectory
    
    def init_arrays(self, n):
        
        """
        initializes the following arrays with n elements:
            
            tof:            time of flights array (s)
            ts:             switch time array (s)
            m_final:        final mass array (kg)
            m_prop:         propellant masses array (kg)
            prop_frac:      propellant fractions array
            alpha0:         initial thrust direction array (rad)
            alphaf:         final thrust direction array (rad)
            hist:           time histories for all the computed trajectories
            failures:       array to keep track of the optimizer failures
            us:             switch radial velocity (m/s)
        """
        
        descentAnalyzerThrust.init_arrays(self, n)
        self.us = np.zeros(n)
        self.ts = np.zeros(n)
        
    def fill_arrays(self, h, tsi, usi, i):
        
        """
        fills the arrays defined in init_arrays() with the values given by the states and control histories
        array h and returns the final mass
        """
        
        mf = descentAnalyzerThrust.fill_arrays(self, h, i)
        
        self.ts[i] = tsi #switch time (m/s)
        self.us[i] = usi #switch radial velocity (m/s)
        
        return mf
            
    def get_results(self):
        
        """
        returns a dictionary with the following keys:
            
            h0:                 initial circular parking orbit altitude (m)
            hp:                 intermediate orbit periapsis altitude (m)
            hs:                 switch altitude (m)
            hf:                 final altitude (m)
            r0:                 initial circular parking orbit radius (m)
            rp:                 intermediate orbit periapsis radius (m)
            rs:                 switch radius (m)
            rf:                 final radius (m)
            a0:                 intermediate orbit semimajor axis (m)
            Isp:                specific impulse (s)
            twr:                thrust/initial weight ratio
            F:                  thrust (N)
            tof:                time of flight (s)
            ts:                 switch time (s)
            m0:                 initial spacecraft mass (kg)
            m_final:            final spacecraft mass (kg)
            m_prop:             propellant mass (kg)
            prop_frac:          propellant fraction
            alpha0:             initial thrust direction (rad)
            alphaf:             final thrust direction (rad)
            hist:               optimal trajectory states and controls histories
            us:                 switch radial velocity (m/s)
            nbh:                number of collocation points in the attitude-free phase
            nbv:                number of collocation points in the vertical phase
            idx_opt:            array index corresponding to the optimal trajectory
            failures:           recorded failures matrix
            fail_summary:       tuple with number of runs, number of failures and failures rate
            
        and optionally:
            
            arr_fail:           array with values for which the optimization failed
            m_park:             initial spacecraft mass in the parking orbit (kg)
            deorbit_prop:       propellant mass required for de-orbit burn (kg)
            deorbit_dv:         impulsive delta-V required for de-orbit burn (m/s)
        """
        
        d = descentAnalyzer2p.get_results(self)
        d2 = descentAnalyzerThrust.get_idx_fail(self)
        d.update(d2)
        
        return d
    
    def set_results(self, d):
        
        """
        saves as class attributes the results for an already computed trajectory stored in a given dictionary d
        """
        
        descentAnalyzer2p.set_results(self, d)
        descentAnalyzerThrust.set_idx_fail(self, d)
        
    def display_summary(self):
        
        """
        prints out the following optimal trajectory's parameters:
            
            h0:                 initial circular parking orbit altitude (m)
            hp:                 intermediate orbit periapsis altitude (m)
            Isp:                specific impulse (s)
            twr:                thrust/initial weight ratio
            F:                  thrust (N)
            tof:                time of flight (s)
            m0:                 initial spacecraft mass (kg)
            m_final:            final spacecraft mass (kg)
            m_prop:             propellant mass (kg)
            m_final/m0:         final/initial mass ratio (%)
            prop_frac:          propellant fraction (%)
            alpha0:             initial thrust direction (deg)
            alphaf:             final thrust direction (deg)
            hs:                 switch altitude (km)
            us:                 switch radial velocity (m/s)
            ts:                 switch time (s)
            
        and optionally:
            
            m_park:             initial spacecraft mass in the parking orbit (kg)
            deorbit_prop:       propellant mass required for de-orbit burn (kg)
            deorbit_dv:         impulsive delta-V required for de-orbit burn (m/s)
        """
        
        descentAnalyzerThrust.display_summary(self)
        
        print('{:<50s}{:>24.2f}'.format('transition altitude (km)', self.hs/1e3))
        print('\nradial velocity at transition (m/s)')
        print(self.us)
        print('\ntime at transition (s)')
        print(self.ts)
        print('')
        
    def plot_all(self, hist, idx_opt, F, rp, n):
        
        """
        calls all the required functions to plot the following quantities:
            
            states_plot(h, n, colors, labels):  state variables histories for the optimal trajectory
            alt_plot(h, n, colors, labels):     altitude profile vs true anomaly
            conrtol_plot(h):                    control history for the optimal trajectory
            trajectory_plot(h, rp):             optimal trajectory in the xy plane
            final_mass_thrust():                final spacecraft mass vs thrust
            param_plot(hist, param, d):         thrust direction and altitude profile for varying thrust
        """
        
        colors = ['b', 'r']
        labels = ['attitude-free', 'vertical']
        
        h = hist[idx_opt] #states and control histories for the optimal trajectory
        
        print("\n\nOptimal soft landing powered descent trajectory with " + str(self.F[self.idx_opt]/1e3) + " kN of thrust")
        
        sstoAnalyzer2p.states_plot(self, h, n, colors, labels) #state variables vs time
        sstoAnalyzer2p.alt_plot(self, h, n, colors, labels) #altitude profile vs true anomaly
        self.control_plot(h) #control vs time
        self.trajectory_plot(h, rp) #trajectory in the xy plane
        descentAnalyzerThrust.final_mass_thrust(self) #final mass vs thrust
        
        d = {'title_alpha':'Thrust direction for different thrust values',
             'title_alt':'Altitude profile for different thrust values', 'label':' kN'}
        
        descentAnalyzerThrust.param_plot(self, hist, F/1e3, d) #thrust direction and altitude profile for varying thrust
                
        plt.show()
    
    def run_optimizer(self):
        
        """
        creates an instance of the class descentOptimizer2p and solves the optimal control problem for every
        thrust value storing the optimal trajectory and all the parameters as class attributes
        """
                
        self.init_arrays(np.size(self.F)) #initialize the matrices to store the different results
        m_opt=-1 #initialize a fake final mass
        
        optimizer = descentOptimizer2p(self.const, self.settings) #create an optimizer instance
        
        execution_start = time() #measure the execution time
        
        for i in range(np.size(self.F)):
                
            print("\nthrust value: " + str(self.F[i]) + " N\n") #current thrust value (N)
                            
            failed = optimizer.run_optimizer(self.F[i], self.w, self.bcs, self.t0[i]) #run the optimizer
                
            if not failed: #current optimization successful
                    
                #retrieve the phase, the trajectory and the final mass
                p = optimizer.get_pbm()
                h, tsi, usi, nbh, nbv = self.get_history(p)
                mf = self.fill_arrays(h, tsi, usi, i)
                    
                #compare the current final mass with its maximum value to determine if the solution is better
                if mf>m_opt:
                    m_opt = mf
                    idx_opt = i
                    
            else:
                self.failures[i] = 1.0 #current optimization failed, update corresponding matrix
                    
        #print out the execution time
        execution_end = time()
        elapsed_time = execution_end-execution_start
        print("\nOptimization done in " + str(elapsed_time) + " s\n")
        
        #print out the failure rate
        descentAnalyzerThrust.get_failures(self, self.F)
        print(str(self.fail_summary[1]) + " failures in " + str(self.fail_summary[0]) + " runs")
        print("Failure rate: " + str(self.fail_summary[2]*100) + " %\n")
        
        #save the number of collocation points in the two phases and the optimal trajectory index
        self.nbh = nbh
        self.nbv = nbv
        self.idx_opt = idx_opt
        
        #display the results
        self.display_summary()
        self.plot_all(self.hist, self.idx_opt, self.F, self.rp, self.nbh)
                
        
class descentAnalyzerThrottle2p(descentAnalyzer2p):
    
    def set_params(self, Isp, twr, hp, hs, klim, t0):
        
        """
        defines the Isp, twr, exaust velocity and thrust as well as the intermediate orbit periapsis altitude,
        the switch altitude and the time of flight initial guess before computing the BCs for the TPBVP
        calling the method deorbit_burn()
        
        inputs:
            
            Isp:    specific impulse (s)
            twr:    thrust/initial weight ratio
            hp:     intermediate orbit periapsis altitude (m)
            hs:     switch altitude (m)
            klim:   throttle limits (kmin, kmax)
            t0:     time of flight guess (s)
            
        variables stored as class attributes:
            
            rp:     intermediate orbit periapsis radius (m)
            rs:     switch radius between horizontal and vertical braking phase (m)
            v0:     circular parking orbit tangential velocity (m/s)
            vp:     intermediate orbit tangential velocity at periapsis (m/s)
            w:      exaust velocity (m/s)
            F:      thrust (N)
        """
        
        self.hs = hs #switch altitude between horizontal and vertical braking phase (m)
        self.rs = self.R + self.hs #switch radius between horizontal and vertical braking phase (m)
        
        descentAnalyzerThrottle.set_params(self, Isp, twr, hp, klim, t0)
        
    def set_bcs(self):
        
        """
        computes the BCs for the TPBVP storing them in the class attribute bcs defined as [r, theta, u, v, m, alpha, k]
        """
            
        descentAnalyzer2p.set_bcs(self)
        
        k = np.array([self.klim[0], self.klim[1], self.klim[1]]) #throttle limits
        self.bcs = np.concatenate((self.bcs, [k]))
                
    def get_history(self, p):
                
        """
        retreives the states and control histories from a given trajectory trj and returns them
        in a unique array hist defined as [t, r, theta, u, v, m, k]
        """
        
        hist0, ts, us, nbh, nbv = descentAnalyzer2p.get_history(self, p)
        
        kh = p.get_val('trj.horiz.timeseries.controls:k').flatten() #throttle in the free-attitude phase
        kh[kh<1e-6]=0. #set to zero all throttle values below a specified threshold
        k = np.concatenate((kh, np.ones(nbv))) #throttle for the whole trajectory
        
        hist0[6][k==0.]=np.nan #remove thrust direction in the nodes where the throttle is zero
        
        hist = np.concatenate((hist0, [k]))

        return hist, ts, us, nbh, nbv
    
    def plot_all(self, h, rp, n):
        
        """
        calls all the required functions to plot the following quantities:
            
            states_plot(h, n, colors, labels):  state variables histories for the optimal trajectory
            alt_plot(h, n, colors, labels):     altitude profile vs true anomaly
            conrtol_plot(h):                    control history for the optimal trajectory
            trajectory_plot(h):                 optimal trajectory in the xy plane
        """
        
        colors = ['b', 'r']
        labels = ['attitude-free', 'vertical']
        
        sstoAnalyzer2p.states_plot(self, h, n, colors, labels) #state variables vs time
        sstoAnalyzer2p.alt_plot(self, h, n, colors, labels) #altitude profile vs true anomaly
        sstoAnalyzerThrottle.control_plot(self, h) #control vs time
        self.trajectory_plot(h, rp) #trajectory in the xy plane
        
        plt.show()

    def run_optimizer(self):
        
        """
        creates an instance of the class descentOptimizerThrottle2p and solves the optimal control problem
        storing the following results as class attributes:
            
            p:              problem class instance for the optimal transfer trajectory
            hist:           array with the states and control variables history [t, r, theta, u, v, m, alpha, k]
        """
        
        optimizer = descentOptimizerThrottle2p(self.const, self.settings) #create an optimizer instance
        
        execution_start = time() #measure the execution time
        
        failed = optimizer.run_optimizer(self.F, self.w, self.bcs, self.t0) #solve the optimal control problem
                
        execution_end = time()
        elapsed_time = execution_end-execution_start
        
        if not failed:
            
            print("\nOptimization done in " + str(elapsed_time) + " s\n") 
            
            self.p = optimizer.get_pbm() 
            self.hist, self.ts, self.us, self.nbh, self.nbv = self.get_history(self.p)        
            self.get_scalars(self.hist)
            
            self.display_summary()
            self.plot_all(self.hist, self.rp, self.nbh)
            
            return self.p
                                    
        else:
            print("\nOptimization failed!\n")
