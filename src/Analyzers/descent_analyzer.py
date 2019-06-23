# -*- coding: utf-8 -*-
"""
the script defines the classes required to set up, solve and display the results for the optimal control problem of finding the most
fuel-efficent single-phase powered descent trajectory from a specified LLO to the Moon surface

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

#adjust path
import sys

if '..' not in sys.path:
    sys.path.append('..')

import numpy as np
from time import time

import matplotlib.pyplot as plt

from Optimizers.descent_optimizer import descentOptimizer, descentOptimizerThrottle
from Analyzers.ssto_analyzer import sstoAnalyzer, sstoAnalyzerThrottle

class descentAnalyzer(sstoAnalyzer): #single phase, single parameters descent trajectory
    
    def __init__(self, const, settings):
        
        """
        creates a new class instance given the following inputs:
            
            const, a dictionary with the following keys:
            
                g0:     standard gravity acceleration (m/s^2)
                mu:     lunar standard gravitational parameter (m^3/s^2)
                R:      lunar radius (m)
                m0:     initial spacecraft mass (kg)
                H:      initial circular parking orbit altitude (m)
                hf:     final altitude (m)
                uf:     final radial velocity (m/s)
                vf:     final tangential velocity (m/s)
                
            settings, a dictionary with the following keys:
                
                solver:                 NLP solver
                tol:                    NLP solver tolerance
                maxiter:                maximum number of iterations
                transcription:          transcription method
                scalers:                optimizer scalers for (time, r, theta, u, v, m)
                defect_scalers:         optimizer defect scalers for (time, r, theta, u, v, m)
                top_level_jacobian:     jacobian format used by openMDAO
                dynamic_simul_derivs:   accounts for the sparsity while computing the jacobian
                compressed:             compressed transcription to reduce the number of variables
                debug:                  check provided partial derivatives
                acc_guess:              use accurate guess (True) or linear interpolation of BCs (False)
                alpha_rate2_cont:       enforce continuity of the second time derivative of the control variable alpha
                deorbit:                consider or not the impulsive de-orbit burn
                
            and (one phase trajectory):
                    
                num_seg:                number of segments in time in which the phase is divided
                transcription_order:    order of the states interpolating polynomials within each segment
                
            or (two phases trajectory, see child classes):
                
                num_seg_horiz:                  number of segments for horizontal phase
                transcription_order_horiz:      order of interpolating polynomials for horizontal phase
                num_seg_vert:                   number of segments for vertical phase
                transcription_order_vert:       order of interpolating polynomials for vertical phase

        and defining the following quantities:
            
            r0:     initial circular parking orbit radius (m)
            rf:     final radius (m)
            W0:     initial spacecraft weight on the Moon surface (N)
        """
        
        self.g0 = const['g0'] #standard gravity acceleration (m/s^2)
        self.mu = const['mu'] #lunar standard gravitational parameter (m^3/s^2)
        self.R = const['R']   #lunar radius (m)
        self.m0 = const['m0'] #initial spacecraft mass (kg)
        self.H = const['H'] #initial circular parking orbit altitude (m)
        self.hf = const['hf'] #final altitude (m)
        self.uf = const['uf'] #final radial velocity (m/s)
        self.vf = const['vf'] #final tangential velocity (m/s)
        
        self.r0 = self.R + self.H #initial circular parking orbit radius (m)
        self.rf = self.R + self.hf #final radius (m)
        self.W0 = self.m0*(self.mu/self.R**2) #initial spacecraft weight on the Moon surface (N)
        
        self.const = const
        self.settings = settings
        
    def set_params(self, Isp, twr, hp, t0):
        
        """
        defines the Isp, twr, exaust velocity and thrust as well as the intermediate orbit periapsis radius
        and the time of flight initial guess before computing the BCs for the TPBVP calling the method
        deorbit_burn()
        
        inputs:
            
            Isp:    specific impulse (s)
            twr:    thrust/initial weight ratio
            hp:     intermediate orbit periapsis altitude (m)
            t0:     time of flight guess (s)
            
        variables stored as class attributes:
            
            rp:     intermediate orbit periapsis radius (m)
            v0:     circular parking orbit tangential velocity (m/s)
            vp:     intermediate orbit tangential velocity at periapsis (m/s)
            w:      exaust velocity (m/s)
            F:      thrust (N)
        """
        
        self.hp = hp #intermediate orbit periapsis altitude (m)
        self.rp = self.R + self.hp #intermediate orbit periapsis radius (m)
        self.v0 = np.sqrt(self.mu/self.r0) #circular parking orbit tangential velocity (m/s)
        
        sstoAnalyzer.set_params(self, Isp, twr, t0)
        
        self.deorbit_burn()
        self.set_bcs() #set TPBVP BCs
        
    def set_bcs(self):
        
        """
        computes the BCs for the TPBVP storing them in the class attribute bcs defined as [r, theta, u, v, m, alpha]
        """
        
        self.bcs = np.array([[self.rp, self.rf], [0.0, np.pi/2], [0.0, self.uf], [self.vp, self.vf],
                             [self.m0, self.m0/100], [np.pi/2, 3/2*np.pi]])
        
    def deorbit_burn(self):
        
        """
        computes the required delta-V and propellant mass to perform the impulsive de-orbit burn
        from the initial circular parking orbit to the intermediate orbit
        
        variables stored as class attributes:
            
            a0:             intermedite orbit semimajor axis (m)
            vp:             intermediate orbit periapsis tangential velocity (m/s)
            va:             intermediate orbit apoapsis tangential velocity (m/s)
            dv:             de-orbit burn impulsive delta-V (m/s)
            
        and optionally:
            
            m_park:         initial spacecraft mass while on the parking orbit (kg)
            m0:             spacecraft mass while on the intermediate orbit after impulsive de-orbit burn (kg)
            deorbit_prop:   propellant mass required for the de-orbit burn (kg)
        """
        
        self.a0 = (self.r0+self.rp)/2.0 #intermediate orbit semimajor axis (m)
        self.vp = np.sqrt(self.mu*(2.0/self.rp-1.0/self.a0)) #intermediate orbit periapsis tangential velocity (m/s)
        self.va = np.sqrt(self.mu*(2.0/self.r0-1.0/self.a0)) #intermediate orbit apoapsis tangential velocity (m/s)
        self.dv = np.abs(self.v0-self.va) #de-orbit burn impulsive delta-V (m/s)
        
        if self.settings['deorbit']:
            self.m_park = self.m0 #initial spacecraft mass prior to the de-orbit burn (kg)
            self.m0 = self.m_park*np.exp(-self.dv/self.g0/self.Isp) #spacecraft mass after de-orbit burn (kg)
            self.deorbit_prop = self.m_park-self.m0 #de-orbit burn propellant mass (kg)
                
    def get_scalars(self, hist):
        
        """
        retrieves from a given histories array hist the following quantities:
            
            m_final:        final spaecraft mass (kg)
            m_prop:         propellant mass (kg)
            prop_frac:      propellant fraction
            tof:            time of flight (s)
            alpha0:         initial thrust direction (rad)
            alphaf:         final thrust direction (rad)
        """
        
        sstoAnalyzer.get_scalars(self, hist)
        self.alphaf = hist[6,-1] #final thrust direction (rad)
        
    def get_results(self):
        
        """
        returns a dictionary with the following keys:
            
            H:              initial circular parking orbit altitude (m)
            hp:             intermediate orbit periapsis altitude (m)
            hf:             final altitude (m)
            r0:             initial circular parking orbit radius (m)
            rp:             intermediate orbit periapsis radius (m)
            rf:             final radius (m)
            a0:             intermediate orbit semimajor axis (m)
            Isp:            specific impulse (s)
            twr:            thrust/initial weight ratio
            F:              thrust (N)
            tof:            time of flight (s)
            m0:             initial spacecraft mass (kg)
            m_final:        final spacecraft mass (kg)
            m_prop:         propellant mass (kg)
            prop_frac:      propellant fraction
            alpha0:         initial thrust direction (rad)
            alphaf:         final thrust direction (rad)
            hist:           optimal trajectory states and controls histories
            
        and optionally:
            
            m_park:         initial spacecraft mass in the parking orbit (kg)
            deorbit_prop:   propellant mass required for de-orbit burn (kg)
            deorbit_dv:     impulsive delta-V required for de-orbit burn (m/s)
        """
        
        d = sstoAnalyzer.get_results(self)
        d1 = {'hp':self.hp, 'hf':self.hf, 'r0':self.r0, 'rp':self.rp, 'a0':self.a0, 'F':self.F, 'alphaf':self.alphaf}
        d.update(d1)
        
        if self.settings['deorbit']:
            d_deorbit = self.get_deorbit_results()
            d.update(d_deorbit)
        
        return d
    
    def get_deorbit_results(self):
        
        """
        returns a dictionary with the following keys:
            
            m_park:         initial spacecraft mass in the parking orbit (kg)
            deorbit_prop:   propellant mass required for de-orbit burn (kg)
            deorbit_dv:     impulsive delta-V required for de-orbit burn (m/s)
        """
        
        d = {'m_park':self.m_park, 'deorbit_prop':self.deorbit_prop, 'deorbit_dv':self.dv}
        
        return d
    
    def set_results(self, d):
        
        """
        saves as class attributes the results for an already computed trajectory stored in a given dictionary d
        """
        
        sstoAnalyzer.set_results(self, d)
        
        self.hp = d['hp']
        self.r0 = d['r0']
        self.rp = d['rp']
        self.a0 = d['a0']
        self.F = d['F']
        self.alphaf = d['alphaf']
        
        if self.settings['deorbit']:
            self.set_deorbit_results(d)
        
    def set_deorbit_results(self, d):
        
        """
        saves as class attributes the results for an already computed deorbit burn stored in a given dictionary d
        """
        
        self.m_park = d['m_park']
        self.deorbit_prop = d['deorbit_prop']
        self.dv = d['deorbit_dv']
        
    def display_deorbit(self):
        
        """
        prints out the following optimal trajectory's parameters:
            m_park:             initial spacecraft mass in the parking orbit (kg)
            m0:                 spacecraft mass after deorbit burn (kg)
            deorbit_prop:       propellant mass required for de-orbit burn (kg)
            deorbit_dv:         impulsive delta-V required for de-orbit burn (m/s)
        """
        
        print('')
        print('{:^54s}'.format('De-orbit burn from initial parking orbit'))
        print('{:<30s}{:>24.2f}'.format('initial mass (kg)', self.m_park))
        print('{:<30s}{:>24.2f}'.format('final mass (kg)', self.m0))
        print('{:<30s}{:>24.2f}'.format('propellant mass (kg)', self.deorbit_prop))
        print('{:<30s}{:>24.2f}'.format('impulsive dV (km/s)', self.dv/1000))
        print('')
        
    def display_summary(self):
        
        """
        prints out the following optimal trajectory's parameters:
            
            H:                  initial circular parking orbit altitude (km)
            hp:                 intermediate orbit periapsis altitude (km)
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
            
        and optionally:
            
            m_park:             initial spacecraft mass in the parking orbit (kg)
            m0:                 spacecraft mass after deorbit burn (kg)
            deorbit_prop:       propellant mass required for de-orbit burn (kg)
            deorbit_dv:         impulsive delta-V required for de-orbit burn (m/s)
        """
                
        #intermediate orbit different from initial parking orbit
        if self.settings['deorbit']:
            self.display_deorbit()
            
        print('')
        print('{:^74s}'.format('Optimal soft landing trajectory'))
        print('{:<50s}{:>24.2f}'.format('circular parking orbit altitude (km)', self.H/1e3))
        print('{:<50s}{:>24.2f}'.format('intermediate orbit periapsis altitude (km)', self.hp/1e3))
        print('{:<50s}{:>24.2f}'.format('specific impulse (s)', self.Isp))
        print('{:<50s}{:>24.2f}'.format('thrust/initial weight ratio', self.twr))
        print('{:<50s}{:>24.2f}'.format('thrust magnitude (N)', self.F))
        print('{:<50s}{:>24.2f}'.format('time of flight (s)', self.tof))
        print('{:<50s}{:>24.2f}'.format('initial spacecraft mass (kg)', self.m0))
        print('{:<50s}{:>24.2f}'.format('final spacecraft mass (kg)', self.m_final))
        print('{:<50s}{:>24.2f}'.format('propellant mass (kg)', self.m_prop))
        print('{:<50s}{:>24.2f}'.format('final/initial mass ratio (%)', self.m_final/self.m0*100))
        print('{:<50s}{:>24.2f}'.format('propellant fraction (%)', self.prop_frac*100))
        print('{:<50s}{:>24.2f}'.format('initial thrust direction (deg)', self.alpha0*180/np.pi))
        print('{:<50s}{:>24.2f}'.format('final thrust direction (deg)', self.alphaf*180/np.pi))
        print('')
        
    def plot_all(self, h, rp):
        
        """
        calls the following class methods to display the different plots:
            
            states_plot(h):             altitude, angle, radial and tangential velocities profiles vs time
            alt_plot(h):                altitude profile vs true anomaly
            control_plot(h):            thrust direction vs time
            trajectory_plot(h, rp):     parking orbit, intermediate orbit and descent trajectory in the xy plane
        """
        
        self.states_plot(h) #state variables vs time
        self.alt_plot(h) #altitude profile vs true anomaly
        self.control_plot(h) #optimal control vs time
        self.trajectory_plot(h, rp) #trajectories in the xy plane
        
        plt.show()
                
    def trajectory_plot(self, h, rp):
        
        """
        plots the optimal soft landing trajectory in the xy plane
        """
    
        #Moon and initial circular parking orbit radius (km)
        r_moon = self.R/1e3
        r_orbit = self.r0/1e3
        
        #soft landing trajectory points (km)
        x_ascent = h[1]*np.cos(h[2])/1e3
        y_ascent = h[1]*np.sin(h[2])/1e3
            
        #axis limits and ticks
        limit = np.ceil(r_orbit/1e3)*1e3
        ticks = np.linspace(-limit, limit, 9)
        
        #true anomaly vector to plot the surface of the Moon and the parking orbit (rad)
        theta = np.linspace(0, 2*np.pi, 200)
    
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(r_moon*np.cos(theta), r_moon*np.sin(theta), label='Moon surface')
        ax.plot(r_orbit*np.cos(theta), r_orbit*np.sin(theta), label='Parking orbit')
        
        if self.r0!=rp: #intermediate transfer orbit with lower preriapsis
            
            a0 = (self.r0+rp)/2.0
            theta = np.linspace(np.pi, 2*np.pi, 100)
            e = (self.r0-rp)/(self.r0+rp)
            r = a0*(1-e**2)/(1+e*np.cos(theta))
            x = r*np.cos(theta)/1e3
            y = r*np.sin(theta)/1e3
            ax.plot(x, y, label='Intermediate orbit')
        
        ax.plot(x_ascent, y_ascent, label='Powered descent')
        ax.set_aspect('equal')
        ax.grid()
        ax.legend(bbox_to_anchor=(1, 1), loc=2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title('Optimal soft landing trajectory')
        ax.tick_params(axis='x', rotation=60)
                
    def run_optimizer(self):
        
        """
        creates an instance of the class descentOptimizer and solves the optimal control problem
        storing the following results as class attributes:
            
            trj:        implicitly obtained trajectory
            hist:       array with the states and control variables history [t, r, theta, u, v, m, alpha]
        """
        
        optimizer = descentOptimizer(self.const, self.settings) #create an optimizer instance
        
        execution_start = time() #measure the execution time
        
        failed = optimizer.run_optimizer(self.F, self.w, self.bcs, self.t0) #solve the optimal control problem
                
        execution_end = time()
        elapsed_time = execution_end-execution_start
        
        if not failed:
            
            print("\nOptimization done in " + str(elapsed_time) + " s\n") 
            
            self.p = optimizer.get_pbm()
            self.hist = self.get_history(self.p) #retrieve the optimal states and control histories          
            self.get_scalars(self.hist) #retrieve the optimal trajectory final values
            
            self.display_summary()
            self.plot_all(self.hist, self.rp)
            
            return self.p #return Problem class instance
                                    
        else:
            print("\nOptimization failed!\n")
            
            
class descentAnalyzerPeriapsis(descentAnalyzer): #single phase, multiple periapsis altitudes trajectories
        
    def init_arrays(self, n):
        
        """
        initializes the following arrays with n elements:
            
            tof:        time of flights array (s)
            m_final:    final mass array (kg)
            m_prop:     propellant masses array (kg)
            prop_frac:  propellant fractions array
            alpha0:     initial thrust direction array (rad)
            alphaf:     final thrust direction array (rad)
            hist:       time histories for all the computed trajectories
            failures:   array to keep track of the optimizer failures
        """
        
        self.tof = np.zeros(n)
        self.m_final = np.zeros(n)
        self.m_prop = np.zeros(n)
        self.prop_frac = np.zeros(n)
        self.alpha0 = np.zeros(n)
        self.alphaf = np.zeros(n)
        self.hist = [] #empty list
        self.failures = np.zeros(n)
        
    def fill_arrays(self, h, i):
        
        """
        fills the arrays defined in init_arrays() with the values given by the states and control histories
        array h and returns the final mass
        """
        
        mf = h[5,-1] #final spacecraft mass (kg)
        m0 = h[5,0] #initial spacecraft mass (kg)
        
        self.tof[i] = h[0,-1] #time of flight (s)
        self.m_final[i] = mf #final spacecraft mass (kg)
        self.m_prop[i] = m0-mf #propellant mass (kg)
        self.prop_frac[i] = self.m_prop[i]/m0 #propellant fraction
        self.alpha0[i] = h[6,0] #initial thrust direction array (rad)
        self.alphaf[i] = h[6,-1] #initial thrust direction (rad)
        self.hist.append(h) #current trajectory
        
        return mf
    
    def get_failures(self, arr):
        
        """
        returns the number of failures, the number of runs, the failure rate and the values in the specified
        array for which the optimization failed
        """
        
        #number of failures and failure rate
        nb_runs = np.size(self.failures)
        nb_fail = np.count_nonzero(self.failures)
        fail_rate = nb_fail/nb_runs
        self.fail_summary = (nb_runs, nb_fail, fail_rate)
        
        if nb_fail>0:
            self.arr_fail = np.take(arr, np.nonzero(self.failures)) #values in the given array corresponding to failures        
            return self.fail_summary, self.arr_fail
        else:
            return self.fail_summary
                 
    def get_results(self):
        
        """
        returns a dictionary with the following keys:
            
            h0:                 initial circular parking orbit altitude (m)
            hp:                 intermediate orbit periapsis altitude (m)
            hf:                 final altitude (m)
            r0:                 initial circular parking orbit radius (m)
            rp:                 intermediate orbit periapsis radius (m)
            rf:                 final radius (m)
            a0:                 intermediate orbit semimajor axis (m)
            Isp:                specific impulse (s)
            twr:                thrust/initial weight ratio
            F:                  thrust (N)
            tof:                time of flight (s)
            m0:                 initial spacecraft mass (kg)
            m_final:            final spacecraft mass (kg)
            m_prop:             propellant mass (kg)
            prop_frac:          propellant fraction
            alpha0:             initial thrust direction (rad)
            alphaf:             final thrust direction (rad)
            hist:               optimal trajectory states and controls histories
            idx_opt:            array index corresponding to the optimal trajectory
            failures:           recorded failures matrix
            fail_summary:       tuple with number of runs, number of failures and failures rate
            
        and optionally:
            
            arr_fail:           array with values for which the optimization failed
            m_park:             initial spacecraft mass in the parking orbit (kg)
            deorbit_prop:       propellant mass required for de-orbit burn (kg)
            deorbit_dv:         impulsive delta-V required for de-orbit burn (m/s)
        """
        
        d = descentAnalyzer.get_results(self)
        d1 = self.get_idx_fail()
        d.update(d1)
        
        return d
    
    def get_idx_fail(self):
        
        """
        returns a dictionary with the following keys:
            
            idx_opt:            array index corresponding to the optimal trajectory
            failures:           recorded failures matrix
            fail_summary:       tuple with number of runs, number of failures and failures rate
            
        and optionally:
            
            arr_fail:           array with values for which the optimization failed
        """
        
        d = {}
        d['idx_opt'] = self.idx_opt
        d['failures'] = self.failures
        d['fail_summary'] = self.fail_summary
        
        if self.fail_summary[1]>0: #at least one failure
            d['arr_fail'] = self.arr_fail
            
        return d
        
    def set_results(self, d):
        
        """
        saves as class attributes the results for an already computed trajectory stored in a given dictionary d
        """
        
        descentAnalyzer.set_results(self, d)
        self.set_idx_fail(d)
        
        
    def set_idx_fail(self, d):
        
        """
        saves as class attributes the results for an already computed trajectory stored in a given dictionary d
        """
        
        self.idx_opt = d['idx_opt']
        self.failures = d['failures']
        self.fail_summary = d['fail_summary']
        
    def display_deorbit(self):
        
        """
        prints out the following optimal trajectory's parameters:
            m_park:             initial spacecraft mass in the parking orbit (kg)
            m0:                 spacecraft mass after deorbit burn (kg)
            deorbit_prop:       propellant mass required for de-orbit burn (kg)
            deorbit_dv:         impulsive delta-V required for de-orbit burn (m/s)
        """

        print('')
        print('{:^54s}'.format('De-orbit burn from initial parking orbit'))
        print('{:<30s}{:>24.2f}'.format('initial mass (kg)', self.m_park))
        print('\nfinal mass (kg)')
        print(self.m0)
        print('\npropellant mass (kg)')
        print(self.deorbit_prop)
        print('\nimpulsive dV (km/s)')
        print(self.dv/1e3)
        print('')
        
    def display_summary(self):
        
        """
        prints out the following optimal trajectory's parameters:
            
            H:                  initial circular parking orbit altitude (km)
            hp:                 intermediate orbit periapsis altitude (km)
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
            
        and optionally:
            
            m_park:             initial spacecraft mass in the parking orbit (kg)
            m0:                 spacecraft mass after deorbit burn (kg)
            deorbit_prop:       propellant mass required for de-orbit burn (kg)
            deorbit_dv:         impulsive delta-V required for de-orbit burn (m/s)
        """
                
        #intermediate orbit different from initial parking orbit
        if self.settings['deorbit']:
            self.display_deorbit()
            
        print('')
        print('{:^74s}'.format('Optimal soft landing trajectory'))
        print('{:<50s}{:>24.2f}'.format('circular parking orbit altitude (km)', self.H/1e3))
        print('{:<50s}{:>24.2f}'.format('specific impulse (s)', self.Isp))
        print('{:<50s}{:>24.2f}'.format('thrust/initial weight ratio', self.twr))
        print('{:<50s}{:>24.2f}'.format('thrust magnitude (N)', self.F))
        print('\nintermediate orbit periapsis altitude (km)')
        print(self.hp/1e3)
        print('\ntime of flight (s)')
        print(self.tof)
        print('\ninitial spacecraft mass (kg)')
        print(self.m0)
        print('\nfinal spacecraft mass (kg)')
        print(self.m_final)
        print('\npropellant mass (kg)')
        print(self.m_prop)
        print('\nfinal/initial mass ratio (%)')
        print(self.m_final/self.m0*100)
        print('\npropellant fraction (%)')
        print(self.prop_frac*100)
        print('\ninitial thrust direction (deg)')
        print(self.alpha0*180/np.pi)
        print('\nfinal thrust direction (deg)')
        print(self.alphaf*180/np.pi)
        print('')
                
    def run_optimizer(self):
        
        """
        creates an instance of the class descentOptimizer and solves the optimal control problem
        for every periapsis altitude value storing the optimal trajectory and all the parameters as class attributes
        """
        
        self.init_arrays(np.size(self.hp)) #initialize the matrices to store the different results
        m_opt=-1 #initialize a fake final mass
        
        optimizer = descentOptimizer(self.const, self.settings) #create an optimizer instance
        
        execution_start = time() #measure the execution time
        
        #solve the optimal control problem for every couple (Isp, twr)
        for i in range(np.size(self.hp)):
                
            print("\nperiapsis altitude: " + str(self.hp[i]/1e3) + " km\n") #current periapsis altitude (km)
            
            #set appropriate BCs depending on current periapsis altitude
            self.bcs[0,0] = self.rp[i]
            self.bcs[3,0] = self.vp[i]
            self.bcs[4] = [self.m0[i], self.m0[i]/100]
                
            failed = optimizer.run_optimizer(self.F, self.w, self.bcs, self.t0[i]) #run the optimizer
                
            if not failed: #current optimization successful
                    
                #retrieve the phase, the trajectory and the final mass
                p = optimizer.get_pbm()
                h = self.get_history(p)
                mf = self.fill_arrays(h, i)
                    
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
        self.get_failures(self.hp)
        print(str(self.fail_summary[1]) + " failures in " + str(self.fail_summary[0]) + " runs")
        print("Failure rate: " + str(self.fail_summary[2]*100) + " %\n")
        
        #save optimal trajectory
        self.idx_opt = idx_opt
        
        #display the results
        self.display_summary()
        self.plot_all(self.hist, self.idx_opt, self.hp)
        
    def plot_all(self, hist, idx_opt, hp):
        
        """
        calls all the required functions to plot the following quantities:
            
            states_plot(h):                 state variables histories for the optimal trajectory
            alt_plot(h):                    altitude profile vs true anomaly
            conrtol_plot(h):                control history for the optimal trajectory
            trajectory_plot(h):             descent trajectory in the xy plane
            final_mass_periapsis():         final spacecraft mass vs periapsis altitude
            param_plot(hist, param, d):     thrust direction and altitude profile for varying periapsis altitudes
        """
        
        h = hist[idx_opt] #states and control histories for the optimal trajectory
        rp = hp[idx_opt] + self.R #periapsis radius for the optimal trajectory (m)
        
        print("\n\nOptimal soft landing powered descent trajectory from a " + str(self.H/1e3) + "x" +
              str(self.hp[self.idx_opt]/1e3) + " km intermediate orbit\n")
        
        self.states_plot(h) #state variables vs time
        self.alt_plot(h) #altitude profile vs true anomaly
        self.control_plot(h) #control vs time
        self.trajectory_plot(h, rp) #trajectory in the xy plane
        self.final_mass_periapsis() #final mass vs periapsis altitude
        
        d = {'title_alpha':'Thrust direction for different periapsis altitudes',
             'title_alt':'Altitude profile for different periapsis altitudes', 'label':' km'}
        
        self.param_plot(hist, hp/1e3, d) #thrust direction and altitude profile for varying periapsis altitudes
                
        plt.show()
        
    def final_mass_periapsis(self):
        
        """
        plots the final mass as function of periapsis altitude
        """
        
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        ax.plot(self.hp/1e3, self.m_final, 'o', color='r')
        ax.set_xlabel('Intermediate orbit periapsis altitude (km)')
        ax.set_ylabel('final mass (kg)')
        ax.set_title('Final spacecraft mass')
        ax.grid()
            
    def param_plot(self, hist, param, d):
        
        #thrust direction alpha (deg)
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('alpha (deg)')
        ax.set_title(d['title_alpha'])
        ax.grid()
        
        i=0
        for h in hist:
            ax.plot(h[0], h[6]*(180/np.pi), label = str(param[i]) + d['label'])
            i+=1
            
        ax.legend()
            
        #altitude profile vs time (km)
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('h (km)')
        ax.set_title(d['title_alt'])
        ax.grid()
        
        i=0
        for h in hist:
            alt = (h[1]-self.R)/1e3 #altitude (km)
            ax.plot(h[0], alt, label = str(param[i]) + d['label'])
            i+=1
            
        ax.legend()
            
        #altitude profile vs true anomaly (km)
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.set_xlabel('true anomaly (deg)')
        ax.set_ylabel('h (km)')
        ax.set_title(d['title_alt'])
        ax.grid()
        
        i=0
        for h in hist:
            alt = (h[1]-self.R)/1e3 #altitude (km)
            ax.plot(h[2]*180/np.pi, alt, label = str(param[i]) + d['label'])
            i+=1
            
        ax.legend()


class descentAnalyzerThrust(descentAnalyzerPeriapsis): #single phase, multiple thrust values trajectories
            
    def display_summary(self):
        
        """
        prints out the following optimal trajectory's parameters:
            
            H:                  initial circular parking orbit altitude (km)
            hp:                 intermediate orbit periapsis altitude (km)
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
            
        and optionally:
            
            m_park:             initial spacecraft mass in the parking orbit (kg)
            m0:                 spacecraft mass after deorbit burn (kg)
            deorbit_prop:       propellant mass required for de-orbit burn (kg)
            deorbit_dv:         impulsive delta-V required for de-orbit burn (m/s)
        """
                
        #intermediate orbit different from initial parking orbit
        if self.settings['deorbit']:
            self.display_deorbit()
            
        print('')
        print('{:^74s}'.format('Optimal soft landing trajectory'))
        print('{:<50s}{:>24.2f}'.format('circular parking orbit altitude (km)', self.H/1e3))
        print('{:<50s}{:>24.2f}'.format('intermediate orbit periapsis altitude (km)', self.hp/1e3))
        print('{:<50s}{:>24.2f}'.format('specific impulse (s)', self.Isp))
        print('{:<50s}{:>24.2f}'.format('initial spacecraft mass (kg)', self.m0))
        print('\nthrust/initial weight ratio')
        print(self.twr)
        print('\nthrust magnitude (N)')
        print(self.F)
        print('\ntime of flight (s)')
        print(self.tof)
        print('\nfinal spacecraft mass (kg)')
        print(self.m_final)
        print('\npropellant mass (kg)')
        print(self.m_prop)
        print('\nfinal/initial mass ratio (%)')
        print(self.m_final/self.m0*100)
        print('\npropellant fraction (%)')
        print(self.prop_frac*100)
        print('\ninitial thrust direction (deg)')
        print(self.alpha0*180/np.pi)
        print('\nfinal thrust direction (deg)')
        print(self.alphaf*180/np.pi)
        print('')
            
    def run_optimizer(self):
        
        """
        creates an instance of the class onePhaseDescentOptimizer and solves the optimal control problem
        for every periapsis altitude value storing the optimal trajectory and all the parameters as class attributes
        """
        
        self.init_arrays(np.size(self.F)) #initialize the matrices to store the different results
        m_opt=-1 #initialize a fake final mass
        
        optimizer = descentOptimizer(self.const, self.settings) #create an optimizer instance
        
        execution_start = time() #measure the execution time
        
        for i in range(np.size(self.F)):
                
            print("\nthrust value: " + str(self.F[i]) + " N\n") #current thrust value (N)
            
            #set appropriate BCs depending on current settings
            self.bcs[4] = [self.m0, self.m0/100]
                
            failed = optimizer.run_optimizer(self.F[i], self.w, self.bcs, self.t0[i]) #run the optimizer
                
            if not failed: #current optimization successful
                    
                #retrieve the phase, the trajectory and the final mass
                p = optimizer.get_pbm()
                h = self.get_history(p)
                mf = self.fill_arrays(h, i)
                    
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
        self.get_failures(self.F)
        print(str(self.fail_summary[1]) + " failures in " + str(self.fail_summary[0]) + " runs")
        print("Failure rate: " + str(self.fail_summary[2]*100) + " %\n")
                
        #save optimal trajectory
        self.idx_opt = idx_opt
        
        #display the results
        self.display_summary()
        self.plot_all(self.hist, self.idx_opt, self.F, self.rp)
                
    def plot_all(self, hist, idx_opt, F, rp):
        
        """
        calls all the required functions to plot the following quantities:
            
            states_plot(h):                 state variables histories for the optimal trajectory
            alt_plot(h):                    altitude profile vs true anomaly
            control_plot(h):                control history for the optimal trajectory
            trajectory_plot(h):             descent trajectory in the xy plane
            final_mass_periapsis():         final spacecraft mass vs periapsis altitude
            param_plot(hist, param, d):     thrust direction and altitude profile for varying periapsis altitudes
        """
        
        h = hist[idx_opt] #states and control histories for the optimal trajectory
        
        print("\n\nOptimal soft landing powered descent trajectory with " + str(self.F[self.idx_opt]) + " N of thrust")
        
        self.states_plot(h) #state variables vs time
        self.alt_plot(h) #altitude profile vs true anomaly
        self.control_plot(h) #optimal control vs time
        self.trajectory_plot(h, rp) #trajectory in the xy plane
        self.final_mass_thrust() #final mass vs thrust
        
        d = {'title_alpha':'Thrust direction for different thrust values',
             'title_alt':'Altitude profile for different thrust values', 'label':' kN'}
        
        self.param_plot(hist, F/1e3, d) #thrust direction and altitude profile for varying thrust values
                
        plt.show()
        
    def final_mass_thrust(self):
        
        """
        plots the final mass as function of thrust
        """
        
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        ax.plot(self.F, self.m_final, 'o', color='r')
        ax.set_xlabel('thrust (N)')
        ax.set_ylabel('final mass (kg)')
        ax.set_title('Final spacecraft mass')
        ax.grid()
                    
class descentAnalyzerThrottle(descentAnalyzer):
    
    def set_params(self, Isp, twr, hp, klim, t0):
        
        """
        defines the Isp, twr, exaust velocity and thrust as well as the intermediate orbit periapsis radius
        and the time of flight initial guess before computing the BCs for the TPBVP calling the method
        deorbit_burn()
        
        inputs:
            
            Isp:    specific impulse (s)
            twr:    thrust/initial weight ratio
            hp:     intermediate orbit periapsis altitude (m)
            klim:   throttle limits (kmin, kmax)
            t0:     time of flight guess (s)
            
        variables stored as class attributes:
            
            rp:     intermediate orbit periapsis radius (m)
            v0:     circular parking orbit tangential velocity (m/s)
            vp:     intermediate orbit tangential velocity at periapsis (m/s)
            w:      exaust velocity (m/s)
            F:      thrust (N)
        """
        self.klim = klim
        descentAnalyzer.set_params(self, Isp, twr, hp, t0)
        
        
    def set_bcs(self):
        
        """
        computes the BCs for the TPBVP storing them in the class attribute bcs defined as [r, theta, u, v, m, alpha, klim]
        """
        
        descentAnalyzer.set_bcs(self)
        
        k = np.array([self.klim]) #throttle limits
        self.bcs = np.concatenate((self.bcs, k))
                    
    def run_optimizer(self):
        
        """
        creates an instance of the class descentOptimizerThrottle and solves the optimal control problem
        storing the following results as class attributes:
            
            trj:    implicitly obtained trajectory
            hist:   array with the states and control variables history [t, r, theta, u, v, m, alpha]
        """
        
        optimizer = descentOptimizerThrottle(self.const, self.settings) #create an optimizer instance
        
        execution_start = time() #measure the execution time
        
        failed = optimizer.run_optimizer(self.F, self.w, self.bcs, self.t0) #solve the optimal control problem
                
        execution_end = time()
        elapsed_time = execution_end-execution_start
        
        if not failed:
            
            print("\nOptimization done in " + str(elapsed_time) + " s\n") 
            
            self.p = optimizer.get_pbm()
            self.hist = sstoAnalyzerThrottle.get_history(self, self.p)         
            self.get_scalars(self.hist) #retrieve the optimal trajectory final values
            
            self.display_summary()
            self.plot_all(self.hist, self.rp)
                                    
        else:
            print("\nOptimization failed!\n")
            
    def plot_all(self, h, rp):
        
        """
        calls the following class methods to display the different plots:
            
            states_plot(h):             altitude, true anomaly, radial and tangential velocity vs time
            alt_plot(h):                altitude profile vs true anomaly
            control_plot(h):            thrust direction vs time
            trajectory_plot(h, rf):     descent tajectory and target LLO in the xy plane
        """
        
        if np.nanmin(self.hist[7])>0.0: #minimum throttle greater than zero
            self.states_plot(h) #state variables vs time
            self.alt_plot(h) #altitude profile vs true anomaly
        else: #minimum throttle equal to zero
            sstoAnalyzerThrottle.states_plot(self, h) #state variables vs time
            sstoAnalyzerThrottle.alt_plot(self, h) #altitude profile vs true anomaly
            
        sstoAnalyzerThrottle.control_plot(self, h) #controls vs time
        self.trajectory_plot(h, rp) #trajectory in the xy plane
        
        plt.show()