# -*- coding: utf-8 -*-
"""
The script defines the classes required to set up, solve and display the results for the optimal control problem
of finding the most fuel-efficient ascent trajectory from the Moon surface to a specified LLO

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

from Optimizers.ssto_optimizer import sstoOptimizer, sstoOptimizerThrottle, sstoOptimizer2p

class sstoAnalyzer: #single phase with constant thrust
    
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
        
        self.g0 = const['g0'] #standard gravity acceleration (m/s^2)
        self.mu = const['mu'] #lunar standard gravitational parameter (m^3/s^2)
        self.R = const['R'] #lunar radius (m)
        
        #target orbit altitude (m)
        self.H = const['H']
        if const['lunar_radii']:
            self.H *= self.R
            
        self.rf = self.R + self.H #target orbit radius (m)
        self.vf = np.sqrt(self.mu/self.rf) #target orbit tangential velocity (m/s)
        
        self.m0 = const['m0'] #initial spacecraft mass (kg)
        self.W0 = self.m0*(self.mu/self.R**2) #initial spacecraft weight on the Moon surface (N)
        
        #boundary conditions assuming [r, theta, u, v, m, alpha]
        self.bcs = np.array([[self.R, self.rf], [0, np.pi/2], [0, 0], [0, self.vf], [self.m0, self.m0/100], [-np.pi/2, np.pi/2]])
    
        self.const = const
        self.settings = settings
            
    def set_params(self, Isp, twr, t0):
        
        """
        defines the Isp, twr, w and F values for which solve the optimization problem and
        an initial guess for the required time of flight where:
            
            Isp:    specific impulse (s)
            twr:    thrust/initial weight ratio
            w:      exaust velocity (m/s)
            F:      thrust (N)
            t0:     time of flight initial guess and bounds defined as (lb, tof, ub) (s)
        """
        
        self.Isp = Isp #specific impulse (s)
        self.twr = twr #thrust/initial weight ratio
        self.w = self.Isp*self.g0 #exaust velocity (m/s)
        self.F = self.twr*self.W0 #thrust (N)
        self.t0 = t0 #time of flight initial guess and bounds (s)
        
    def set_results(self, d):
        
        """
        saves as class attributes the results for an already computed trajectory stored in a given dictionary d
        """
        
        self.H = d['H']
        self.rf = d['rf']
        self.Isp = d['Isp']
        self.twr = d['twr']
        self.tof = d['tof']
        self.m0 = d['m0']
        self.m_final = d['m_final']
        self.m_prop = d['m_prop']
        self.prop_frac = d['prop_frac']
        self.alpha0 = d['alpha0']
        self.hist = d['hist']
        
        try:
            self.hist_exp = d['hist_exp']
            self.err = d['err']
            self.max_err = d['max_err']
            self.rel_err = d['rel_err']
        except:
            pass

    def get_history(self, pbm):
        
        """
        retreives the states and control histories from a given Problem class instance
        and returns them in a unique array hist defined as [t, r, theta, u, v, m, alpha]
        """
        
        t = pbm.get_val('phase0.timeseries.time_phase').flatten()
        
        r = pbm.get_val('phase0.timeseries.states:r').flatten()
        theta = pbm.get_val('phase0.timeseries.states:theta').flatten()
        u = pbm.get_val('phase0.timeseries.states:u').flatten()
        v = pbm.get_val('phase0.timeseries.states:v').flatten()
        m = pbm.get_val('phase0.timeseries.states:m').flatten()
        
        alpha = pbm.get_val('phase0.timeseries.controls:alpha').flatten()
        
        hist = np.array([t, r, theta, u, v, m, alpha])
        
        return hist

    def get_scalars(self, hist):
        
        """
        retrieves from a given histories array hist the following quantities:
            
            m_final:        final spaecraft mass (kg)
            m_prop:         propellant mass (kg)
            prop_frac:      propellant fraction
            tof:            time of flight (s)
            alpha0:         initial thrust direction (rad)
        """

        self.m_final = hist[5,-1] #final mass (kg)
        self.m_prop = self.m0 - self.m_final #propellant mass (kg)
        self.prop_frac = self.m_prop/self.m0 #propellant fraction
        
        self.tof = hist[0,-1] #time of flight (s)
        self.alpha0 = hist[6,0] #initial thrust direction (rad)
        
    def get_results(self):
        
        """
        returns a dictionary with the following keys:
            
            H:              target orbit altitude (m)
            rf:             final orbit radius (m)
            Isp:            specific impulse (s)
            twr:            thrust/initial weight ratio
            tof:            time of flight (s)
            m0:             initial spacecraft mass (kg)
            m_final:        final spacecraft mass (kg)
            m_prop:         propellant mass (kg)
            prop_frac:      propellant fraction
            alpha0:         initial thrust direction (rad)
            hist:           state variables and controls histories array
        """
        
        d = {'H':self.H, 'rf':self.rf, 'Isp':self.Isp, 'twr':self.twr, 'tof':self.tof, 'm0':self.m0,
             'm_final':self.m_final, 'm_prop':self.m_prop, 'prop_frac':self.prop_frac, 'alpha0':self.alpha0,
             'hist':self.hist}
        
        try:
            d['hist_exp'] = self.hist_exp
            d['err'] = self.err
            d['max_err'] = self.max_err
            d['rel_err'] = self.rel_err
        except:
            pass
        
        return d
        
    def display_summary(self):
        
        """
        prints out a summary of the optimal trajectory's parameters
        """
        
        print('')
        
        #print optimizer settings
        try:
            print('{:^54s}'.format('Optimizer settings'))
            print('{:<30s}{:>24s}'.format('NLP solver', self.settings['solver']))
            print('{:<30s}{:>24s}'.format('transcription', self.settings['transcription']))
            print('{:<30s}{:>24d}'.format('number of segments', self.settings['num_seg']))
            print('{:<30s}{:>24d}'.format('transcription order', self.settings['transcription_order']))
            print('{:<30s}{:>24d}'.format('alpha rate2 continuity', self.settings['alpha_rate2_cont']))
            print('{:<30s}{:>24d}'.format('accurate guess', self.settings['acc_guess']))
        except:
            pass
        
        print('')
    
        #print ascent trajectory data
        print('{:^54s}'.format('Optimal ascent trajectory to LLO'))
        print('{:<30s}{:>24.2f}'.format('orbit altitude (km)', self.H/1e3))
        print('{:<30s}{:>24.2f}'.format('specific impulse (s)', self.Isp))
        print('{:<30s}{:>24.2f}'.format('initial thrust/wight ratio', self.twr))
        print('{:<30s}{:>24.2f}'.format('time of flight (s)', self.tof))
        print('{:<30s}{:>24.2f}'.format('final/initial mass ratio (%)', self.m_final/self.m0*100))
        print('{:<30s}{:>24.2f}'.format('propellant fraction (%)', self.prop_frac*100))
        print('{:<30s}{:>24.2f}'.format('initial thrust direction (deg)', self.alpha0*180/np.pi))
        print('')
                
        try:
            self.display_error()
        except:
            pass
                
    def display_error(self):
        
        print('\nError between the implicit and explicit trajectories')
        print('\nmaximum absolute error:')
        print(self.max_err)
        print('\nmaximum relative error:')
        print(self.rel_err)
        print('')
        
    def plot_all(self, h, h_exp, rf):
        
        """
        calls the following class methods to display the different plots:
            
            states_plot(h, h_exp):      altitude, angle, radial and tangential velocity vs time
            alt_plot(h, h_exp):         altitude profile vs angle
            control_plot(h):            thrust direction vs time
            trajectory_plot(h, rf):     ascent trajectory and target LLO in the xy plane
        """

        self.states_plot(h, h_exp) #state variables vs time
        self.alt_plot(h, h_exp) #altitude profile vs angle
        self.control_plot(h) #control vs time
        self.trajectory_plot(h, rf) #trajectory in the xy plane
        
        plt.show()
        
    def states_plot(self, h, h_exp=None):
        
        """
        plots the state variables (h, theta, u, v) vs time
        """
        
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        
        #explicit simulation
        if h_exp is not None:
            axs[0,0].plot(h_exp[0], (h_exp[1]-self.R)/1e3, color='g')
            axs[1,0].plot(h_exp[0], h_exp[2]*(180/np.pi), color='g')
            axs[0,1].plot(h_exp[0], h_exp[3]/1e3, color='g')
            axs[1,1].plot(h_exp[0], h_exp[4]/1e3, color='g')
        
        #implicit solution
        
        #altitude (km)
        axs[0,0].plot(h[0], (h[1]-self.R)/1e3, 'o', color='b')
        axs[0,0].set_xlabel('time (s)')
        axs[0,0].set_ylabel('h (km)')
        axs[0,0].set_title('Altitude')
        axs[0,0].grid()

        #angle (deg)
        axs[1,0].plot(h[0], h[2]*(180/np.pi), 'o', color='b')
        axs[1,0].set_xlabel('time (s)')
        axs[1,0].set_ylabel('theta (deg)')
        axs[1,0].set_title('Angle')
        axs[1,0].grid()

        #radial velocity (km/s)
        axs[0,1].plot(h[0], h[3]/1e3, 'o', color='b')
        axs[0,1].set_xlabel('time (s)')
        axs[0,1].set_ylabel('u (km/s)')
        axs[0,1].set_title('Radial velocity')
        axs[0,1].grid()

        #tangential velocity (km/s)
        axs[1,1].plot(h[0], h[4]/1e3, 'o', color='b')
        axs[1,1].set_xlabel('time (s)')
        axs[1,1].set_ylabel('v (km/s)')
        axs[1,1].set_title('Tangential velocity')
        axs[1,1].grid()
        

    def control_plot(self, h):
        
        """
        plots the thrust direction vs time
        """
        
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
                    
        ax.plot(h[0], h[6]*(180/np.pi), 'o', color='r')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('alpha (deg)')
        ax.set_title('Thrust direction')
        ax.grid()
            
    def alt_plot(self, h, h_exp=None):
        
        """
        plots the altitude profile vs angle
        """
        
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        #explicit simulation
        if h_exp is not None:
            alt_exp = (h_exp[1]-self.R)/1e3 #altitude (km)
            ta_exp = h_exp[2]*180/np.pi #angle (deg)
            ax.plot(ta_exp, alt_exp, color='g')
        
        #implicit solution
        alt = (h[1]-self.R)/1e3 #altitude (km)
        ta = h[2]*180/np.pi #angle (deg)
        
        ax.plot(ta, alt, 'o', color='b')
        ax.set_xlabel('theta (deg)')
        ax.set_ylabel('h (km)')
        ax.set_title('Altitude profile')
        ax.grid()

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
            
        #axis limits and ticks
        limit = np.ceil(r_orbit/1e3)*1e3
        ticks = np.linspace(-limit, limit, 9)
        
        #true anomaly vector to plot the surface of the Moon and the final orbit (rad)
        theta = np.linspace(0, 2*np.pi, 200)
    
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(r_moon*np.cos(theta), r_moon*np.sin(theta), label='Moon surface')
        ax.plot(r_orbit*np.cos(theta), r_orbit*np.sin(theta), label='Target orbit')
        ax.plot(x_ascent, y_ascent, label='Ascent trajectory')
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
        
        self.settings['acc_guess'] = False
        return sstoOptimizer(self.const, self.settings)
        
    def run_optimizer(self):
        
        """
        solves the optimal control problem storing the following results as class attributes:
            
            pbm:    OpenMDAO Problem class instance corresponding to the obtained optimal solution
            ph:     Dymos Phase class instance corresponding to the obtained optimal solution
            hist:   array with the states and control variables history [t, r, theta, u, v, m, alpha]
        """
        
        optimizer =  self.set_optimizer()
        
        execution_start = time() #measure the execution time
        
        failed = optimizer.run_optimizer(self.F, self.w, self.bcs, self.t0) #solve the optimal control problem
                
        execution_end = time()
        elapsed_time = execution_end-execution_start
        
        if not failed:
            
            print("\nOptimization done in " + str(elapsed_time) + " s\n") 
            
            #retrieve the results
            self.pbm = optimizer.get_pbm()
            self.ph = optimizer.get_phase()
            self.hist = self.get_history(self.pbm)            
            self.get_scalars(self.hist)
            
            #explicitly simulate the obtained trajectory (optional)
            if self.settings['exp_sim']:
                self.run_exp_sim()
            else:
                self.hist_exp = None
            
            #display the results
            self.display_summary()
            self.plot_all(self.hist, self.hist_exp, self.rf)
            return self.pbm, self.ph
                                    
        else:
            print("\nOptimization failed!\n")
            
    def run_exp_sim(self):
        
        """
        explicitly integrate the trajectory with the implicitly obtained controls histories
        """
        
        #explicitly simulate the trajectory
        self.p_exp = self.ph.simulate(times_per_seg=None, atol=1e-12, rtol=1e-12)
        self.hist_exp = self.get_history(self.p_exp)
        
        #evaluate the error between the two trajectories
        self.err = self.hist[1:]-self.hist_exp[1:]
        self.max_err = np.nanmax(np.absolute(self.err), axis=1)
        self.rel_err = self.max_err/np.nanmax(self.hist[1:], axis=1)

                
class sstoAnalyzerThrottle(sstoAnalyzer): #single phase with varying thrust
                
    def set_params(self, Isp, twr, klim, t0):
        
        """
        defines the Isp, twr, w and F values for which solve the optimization problem,
        the throttle limits and an initial guess for the required time of flight where:
            
            Isp:        specific impulse (s)
            twr:        thrust/initial weight ratio
            w:          exaust velocity (m/s)
            F:          thrust (N)
            klim:       throttle limits (kmin, kmax)
            t0:         time of flight initial guess as (lb, tof, ub) (s)
        """
        
        sstoAnalyzer.set_params(self, Isp, twr, t0)
        
        k = np.array([klim]) #throttle limits
        self.bcs = np.concatenate((self.bcs, k)) #phase BCs
            
    def get_history(self, pbm):
        
        """
        retreives the states and control histories from a given problem p and returns them
        in a unique array hist defined as [t, r, theta, u, v, m, alpha, k]
        """
        
        hist0 = sstoAnalyzer.get_history(self, pbm) #without throttle
        
        k = pbm.get_val('phase0.timeseries.controls:k').flatten() #throttle
        nb = np.size(k) #number of nodes
        
        k[k<1e-5]=0. #set to zero all throttle values below a specified threshold
        hist0[6][k==0.]=np.nan #remove thrust direction in the nodes where the throttle is zero
                
        hist = np.concatenate((hist0, np.reshape(k, (1, nb))))
        
        return hist
    
    def display_summary(self):
        
        """
        prints out a summary of the optimal trajectory parameters
        """
        
        sstoAnalyzer.display_summary(self)
        
        lb = self.hist[0][self.hist[7]>0][-1]
        print('')
        print('{:<30s}{:>24.2f}'.format('last burn (s)', lb))
        print('')
        
    
    def plot_all(self, h, h_exp, rf):
        
        """
        calls the following class methods to display the different plots:
            
            states_plot(h, h_exp, colors):      altitude, angle, radial and tangential velocity vs time
            alt_plot(h, h_exp, colors):         altitude profile vs angle
            control_plot(h):                    thrust direction vs time
            trajectory_plot(h, rf):             ascent tajectory and target LLO in the xy plane
        """
        
        if np.min(self.hist[7])>0.0: #minimum throttle greater than zero
            colors = ('b', 'r')
        else:
            colors = ('r', 'b')

        self.states_plot(h, h_exp, colors) #state variables vs time
        self.alt_plot(h, h_exp, colors) #altitude profile vs angle
        self.control_plot(h) #controls vs time
        self.trajectory_plot(h, rf) #trajectory in the xy plane
        
        plt.show()
    
    def states_plot(self, h, h_exp=None, colors=('r', 'b')):
        
        """
        plots the optimal state variables (h, theta, u, v) vs time
        """
        
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        k = h[7] #throttle values
        
        #explicit simulation
        if h_exp is not None:
            axs[0,0].plot(h_exp[0], (h_exp[1]-self.R)/1e3, color='g')
            axs[1,0].plot(h_exp[0], h_exp[2]*(180/np.pi), color='g')
            axs[0,1].plot(h_exp[0], h_exp[3]/1e3, color='g')
            axs[1,1].plot(h_exp[0], h_exp[4]/1e3, color='g')
        
        #implicit solution
        
        #altitude (km)
        axs[0,0].plot(h[0][k==0.], (h[1]-self.R)[k==0.]/1e3, 'o', color=colors[1], label='coasting')
        axs[0,0].plot(h[0][k!=0.], (h[1]-self.R)[k!=0.]/1e3, 'o', color=colors[0], label='powered')
        axs[0,0].set_xlabel('time (s)')
        axs[0,0].set_ylabel('h (km)')
        axs[0,0].set_title('Altitude')
        axs[0,0].grid()
        axs[0,0].legend(loc='best')

        #angle (deg)
        axs[1,0].plot(h[0][k==0.], h[2][k==0.]*(180/np.pi), 'o', color=colors[1], label='coasting')
        axs[1,0].plot(h[0][k!=0.], h[2][k!=0.]*(180/np.pi), 'o', color=colors[0], label='powered')
        axs[1,0].set_xlabel('time (s)')
        axs[1,0].set_ylabel('theta (deg)')
        axs[1,0].set_title('Angle')
        axs[1,0].grid()
        axs[1,0].legend(loc='best')

        #radial velocity (km/s)
        axs[0,1].plot(h[0][k==0.], h[3][k==0.]/1e3, 'o', color=colors[1], label='coasting')
        axs[0,1].plot(h[0][k!=0.], h[3][k!=0.]/1e3, 'o', color=colors[0], label='powered')
        axs[0,1].set_xlabel('time (s)')
        axs[0,1].set_ylabel('u (km/s)')
        axs[0,1].set_title('Radial velocity')
        axs[0,1].grid()
        axs[0,1].legend(loc='best')

        #tangential velocity (km/s)
        axs[1,1].plot(h[0][k==0.], h[4][k==0.]/1e3, 'o', color=colors[1], label='coasting')
        axs[1,1].plot(h[0][k!=0.], h[4][k!=0.]/1e3, 'o', color=colors[0], label='powered')
        axs[1,1].set_xlabel('time (s)')
        axs[1,1].set_ylabel('v (km/s)')
        axs[1,1].set_title('Tangential velocity')
        axs[1,1].grid()
        axs[1,1].legend(loc='best')
            
    def control_plot(self, h):
        
        """
        plots the thrust direction and throttle values vs time
        """
        
        #thrust direction alpha (deg)
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(h[0], h[6]*(180/np.pi), 'o', color='r')
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('alpha (deg)')
        axs[0].set_title('Thrust direction')
        axs[0].grid()
        
        #throttle
        axs[1].plot(h[0], h[7], 'o', color='r')
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('throttle')
        axs[1].set_title('Instantaneous thrust vs maximum thrust')
        axs[1].set_ylim((-0.1, 1.1))
        axs[1].grid()
                    
    def alt_plot(self, h, h_exp=None, colors=('r', 'b')):
        
        """
        plots the altitude profile vs angle
        """
        
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        if h_exp is not None:
            alt_exp = (h_exp[1]-self.R)/1e3 #altitude (km)
            ta_exp = h_exp[2]*180/np.pi #angle (deg)
            ax.plot(ta_exp, alt_exp, color='g')
        
        alt = (h[1]-self.R)/1e3 #altitude (km)
        ta = h[2]*180/np.pi #angle (deg)
        k = h[7] #throttle values
        
        ax.plot(ta[k==0], alt[k==0], 'o', color=colors[1], label='coasting')
        ax.plot(ta[k!=0], alt[k!=0], 'o', color=colors[0], label='powered')
        ax.set_xlabel('theta (deg)')
        ax.set_ylabel('h (km)')
        ax.set_title('Altitude profile')
        ax.grid()
        ax.legend(loc='best')
        
    def set_optimizer(self):
        
        """
        returns an instance of the class sstoOptimizer
        """
        
        return sstoOptimizerThrottle(self.const, self.settings)
        

class sstoAnalyzer2p(sstoAnalyzer): #two phases with constant thrust
    
    def __init__(self, const, settings):
        
        """
        creates a new class instance given the following inputs:
            
            const, a dictionary with the following keys:
            
                g0:             standard gravity acceleration (m/s^2)
                mu:             lunar standard gravitational parameter (m^3/s^2)
                R:              lunar radius (m)
                H:              target LLO altitude (lunar_radii or m)
                h0:             initial altitude (m)
                hs:             switch altitude (m)
                us:             switch radial velocity (m/s)
                m0:             initial spacecraft mass (kg)
                m0_prop:        initial propellant mass (kg)
                lunar_radii:    orbit altitude H expressed in lunar radii (True) or in meters (False)
                
            settings, a dictionary with the following keys:
                
                solver:                         NLP solver
                tol:                            NLP stopping tolerance
                maxiter:                        maximum number of iterations
                transcription:                  transcription method
                num_seg_horiz:                  number of segments for horizontal phase
                transcription_order_horiz:      order of interpolating polynomials for horizontal phase
                num_seg_vert:                   number of segments for vertical phase
                transcription_order_vert:       order of interpolating polynomials for vertical phase
                scalers:                        optimizer scalers for (time, r, theta, u, v, m)
                defect_scalers:                 optimizer defect scalers for (time, r, theta, u, v, m)
                top_level_jacobian:             jacobian format used by openMDAO
                dynamic_simul_derivs:           accounts for the sparsity while computing the jacobian
                compressed:                     compressed transcription to reduce the number of variables
                debug:                          check partial derivatives defined in the ODEs
                acc_guess:                      use or not an accurate initial guess
                alpha_rate2_cont:               impose or not the continuity of the second derivative of alpha
                duration_bounds:                impose or not the phase duration bounds specified in t0
                exp_sim:                        explicit integration using the optimal control profile
            
        and defining the following quantities:
            
            r0:     initial radius (m)
            rs:     switch radius (m)
            rf:     target LLO radius (m)
            vf:     target LLO tangential velocity (m/s)
            W0:     initial spacecraft weight on the Moon surface (N)
            bcs:    array with all the required boundary conditions
        """
        
        sstoAnalyzer.__init__(self, const, settings)
        
        self.h0 = const['h0'] #initial altitude (m)
        self.hs = const['hs'] #switch altitude (m)        
        self.us = const['us'] #switch radial velocity (m/s)
        self.m0_prop = const['m0_prop'] #initial propellant mass (kg)
        
        self.r0 = self.R + self.h0 #initial radius (m)
        self.rs = self.R + self.hs #switch radius (m)
        self.bcs = np.array([[self.r0, self.rs, self.rf], [0.0, 0.0, np.pi/6], [0.0, self.us, 0.0],
                             [0.0, 0.0, self.vf], [self.m0, self.m0-self.m0_prop/5, self.m0-self.m0_prop],
                             [-np.pi/2, np.pi/2, 0]])
          
    def get_history_vert(self, p):
        
        """
        retreives the states and control histories for the vertical-rise phase from a given
        trajectory trj and returns them in a tuple defined as (t, r, u, m, nb)
        """
        
        t = p.get_val('trj.vert.timeseries.time').flatten()
        r = p.get_val('trj.vert.timeseries.states:r').flatten()
        u = p.get_val('trj.vert.timeseries.states:u').flatten()
        m = p.get_val('trj.vert.timeseries.states:m').flatten()
        nb = np.size(t)
                
        return t, r, u, m, nb
    
    def get_history_horiz(self, p):
        
        """
        retreives the states and control histories for the free-attitude phase from a given
        trajectory trj and returns them in a tuple defined as (t, r, theta, u, v, m, alpha, nb)
        """
        
        t = p.get_val('trj.horiz.timeseries.time').flatten()
        r = p.get_val('trj.horiz.timeseries.states:r').flatten()
        theta = p.get_val('trj.horiz.timeseries.states:theta').flatten()
        u = p.get_val('trj.horiz.timeseries.states:u').flatten()
        v = p.get_val('trj.horiz.timeseries.states:v').flatten()
        m = p.get_val('trj.horiz.timeseries.states:m').flatten()
        alpha = p.get_val('trj.horiz.timeseries.controls:alpha').flatten()
        nb = np.size(t)
        
        return t, r, theta, u, v, m, alpha, nb
    
    
    def get_history(self, p):
                
        """
        retreives the states and control histories from a given trajectory trj and returns them
        in a unique array hist defined as [t, r, theta, u, v, m, alpha]
        """
        
        #state and control variables in the vertical-rise phase
        tv, rv, uv, mv, nbv = self.get_history_vert(p)
        thetav = np.zeros(nbv)
        vv = np.zeros(nbv)
        alphav = np.ones(nbv)*np.pi/2
        histv = np.array([tv, rv, thetav, uv, vv, mv, alphav])
        
        #state and control variables in the free-attitude ascent phase
        th, rh, thetah, uh, vh, mh, alphah, nbh = self.get_history_horiz(p)
        histh = np.array([th, rh, thetah, uh, vh, mh, alphah])
        
        #state and control variables throughout the whole trajectory        
        hist = np.concatenate((histv, histh), 1)
        
        #switch time, altitude and radial velocity
        ts = tv[-1]
        hs = rv[-1] - self.R
        us = uv[-1]
        
        return hist, ts, hs, us, nbv, nbh
        
    def get_results(self):
        
        """
        returns a dictionary with the following keys:

            H:              final orbit altitude (m)
            rf:             final orbit radius (m)
            Isp:            specific impulse (s)
            twr:            thrust/initial weight ratio
            tof:            time of flight (s)
            m0:             initial spacecraft mass (kg)
            m_final:        final spacecraft mass (kg)
            m_prop:         propellant mass (kg)
            prop_frac:      propellant fraction
            alpha0:         initial thrust direction (rad)
            hist:           optimal trajectory states and controls histories
            
            hs:             switch altitude (m)
            rs:             switch radius (m)
            ts:             switch time (s)
            us:             switch radial velocity (m/s)
            nbv:            number of collocation points in the vertical rise phase
            nbh:            number of collocation points in the attitude-free phase
        """
        
        d = sstoAnalyzer.get_results(self)
        d1 = {'hs':self.hs, 'rs':self.rs, 'ts':self.ts, 'us':self.us, 'nbv':self.nbv, 'nbh':self.nbh}
        d.update(d1)
        
        return d
    
    def set_results(self, d):
        
        """
        saves as class attributes the results for an already computed trajectory given in the dictionary d
        """
        
        sstoAnalyzer.set_results(self, d)
        
        self.hs = d['hs']
        self.rs = d['rs']
        self.ts = d['ts']
        self.us = d['us']
        self.nbv = d['nbv']
        self.nbh = d['nbh']
        
    def display_summary(self):
        
        """
        prints out a summary of the optimal trajectory parameters
        """
        
        sstoAnalyzer.display_summary(self)
        
        print('')
        print('{:<30s}{:>24.2f}'.format('switch altitude (m)', self.hs))
        print('{:<30s}{:>24.2f}'.format('switch time (s)', self.ts))
        print('{:<30s}{:>24.2f}'.format('switch radial velocity (m/s)', self.us))
        print('')
        
    def plot_all(self, h, rf, n):
        
        """
        calls the following class methods to display the different plots:
            
            states_plot(h, n, colors, labels):      altitude, angle, radial and tangential velocities vs time
            alt_plot(h, n, colors, labels):         altitude profile vs angle
            control_plot(h):                        thrust direction vs time
            trajectory_plot(h, rf):                 ascent trajectory in the xy plane
        """
        
        colors = ['r', 'b']
        labels = ['vertical', 'attitude-free']
        
        self.states_plot(h, n, colors, labels) #state variables vs time
        self.alt_plot(h, n, colors, labels) #altitude profile vs angle
        self.control_plot(h) #control vs time
        self.trajectory_plot(h, rf) #trajectory in the xy plane
        
        plt.show()
        
    def states_plot(self, h, n, colors, labels):
        
        """
        plots the state variables [h, theta, u, v] vs time
        """
        
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        
        #altitude (km)
        axs[0,0].plot(h[0,:n], (h[1,:n]-self.R)/1e3, 'o', color=colors[0], label=labels[0], zorder=2)
        axs[0,0].plot(h[0,n:], (h[1,n:]-self.R)/1e3, 'o', color=colors[1], label=labels[1], zorder=1)
        axs[0,0].set_xlabel('time (s)')
        axs[0,0].set_ylabel('h (km)')
        axs[0,0].set_title('Altitude')
        axs[0,0].grid()
        axs[0,0].legend(loc='best')

        #angle (deg)
        axs[1,0].plot(h[0,:n], h[2,:n]*(180/np.pi), 'o', color=colors[0], label=labels[0], zorder=2)
        axs[1,0].plot(h[0,n:], h[2,n:]*(180/np.pi), 'o', color=colors[1], label=labels[1], zorder=1)
        axs[1,0].set_xlabel('time (s)')
        axs[1,0].set_ylabel('theta (deg)')
        axs[1,0].set_title('Angle')
        axs[1,0].grid()
        axs[1,0].legend(loc='best')

        #radial velocity (km/s)
        axs[0,1].plot(h[0,:n], h[3,:n]/1e3, 'o', color=colors[0], label=labels[0], zorder=2)
        axs[0,1].plot(h[0,n:], h[3,n:]/1e3, 'o', color=colors[1], label=labels[1], zorder=1)
        axs[0,1].set_xlabel('time (s)')
        axs[0,1].set_ylabel('u (km/s)')
        axs[0,1].set_title('Radial velocity')
        axs[0,1].grid()
        axs[0,1].legend(loc='best')

        #tangential velocity (km/s)
        axs[1,1].plot(h[0,:n], h[4,:n]/1e3, 'o', color=colors[0], label=labels[0], zorder=2)
        axs[1,1].plot(h[0,n:], h[4,n:]/1e3, 'o', color=colors[1], label=labels[1], zorder=1)
        axs[1,1].set_xlabel('time (s)')
        axs[1,1].set_ylabel('v (km/s)')
        axs[1,1].set_title('Tangential velocity')
        axs[1,1].grid()
        axs[1,1].legend(loc='best')
            
    def alt_plot(self, h, n, colors, labels):
        
        """
        plots the altitude profile vs angle
        """
        
        alt = (h[1]-self.R)/1e3 #altitude (km)
        ta = h[2]*180/np.pi #angle (deg)
        
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        ax.plot(ta[:n], alt[:n], 'o', color=colors[0], label=labels[0], zorder=2)
        ax.plot(ta[n:], alt[n:], 'o', color=colors[1], label=labels[1], zorder=1)
        ax.set_xlabel('theta (deg)')
        ax.set_ylabel('h (km)')
        ax.set_title('Altitude profile')
        ax.grid()
        ax.legend(loc='best')
                        
    def run_optimizer(self):
        
        """
        creates an instance of the class sstoOptimizer2p and solves the optimal control problem
        storing the following results as class attributes:
            
            trj:            implicitly obtained trajectory
            hist:           array with the states and control variables history [t, r, theta, u, v, m, alpha]
        """
        
        optimizer = sstoOptimizer2p(self.const, self.settings)
        
        execution_start = time() #measure the execution time
        
        failed = optimizer.run_optimizer(self.F, self.w, self.bcs, self.t0) #solve the optimal control problem
                
        execution_end = time()
        elapsed_time = execution_end-execution_start
        
        if not failed:
            
            print("\nOptimization done in " + str(elapsed_time) + " s\n") 
            
            self.p = optimizer.get_pbm()
            self.hist, self.ts, self.hs, self.us, self.nbv, self.nbh = self.get_history(self.p)            
            self.get_scalars(self.hist)
            
            self.display_summary()
            self.plot_all(self.hist, self.rf, self.nbv)
            
            return self.p
                                    
        else:
            print("\nOptimization failed!\n")
            
         
class sstoAnalyzerMatrix(sstoAnalyzer): #single phase with constant thrust, multiple Isp and twr values
    
    def init_matrix(self):
        
        """
        initializes the following matrices:
            
            tof:            time of flights matrix (s)
            m_final:        final mass matrix (kg)
            m_prop:         propellant masses matrix (kg)
            prop_frac:      propellant fractions matrix
            alpha0:         initial thrust direction matrix (rad)
            failures:       matrix to keep track of the optimizer failures
        """
        
        rows = np.size(self.twr) #number of rows
        cols = np.size(self.Isp) #number of columns
        
        self.tof = np.zeros((rows, cols))
        self.m_final = np.zeros((rows, cols))
        self.m_prop = np.zeros((rows, cols))
        self.prop_frac = np.zeros((rows, cols))
        self.alpha0 = np.zeros((rows, cols))
        self.failures = np.zeros((rows, cols))
        
    def fill_matrix(self, hist, i, j):
        
        """
        fills the matrices defined in init_matrix() with the values given by the histories array hist
        and returns the final mass
        """
        
        mf = hist[5,-1] #final spacecraft mass (kg)
        
        self.tof[i,j] = hist[0,-1] #time of flight (s)
        self.m_final[i,j] =  mf #final spacecraft mass (kg)
        self.m_prop[i,j] = self.m0 - mf#propellant mass (kg)
        self.prop_frac[i,j] = self.m_prop[i,j]/self.m0 #propellant fraction
        self.alpha0[i,j] = hist[6,0] #initial thrust direction (rad)
        
        return mf
    
    def get_failures(self):
        
        """
        returns the number of failures, the number of runs, the failure rate and
        two array with the Isp and twr values for which the optimization failed
        """
        
        #number of failures and failure rate
        nb_runs = np.size(self.failures)
        nb_fail = np.count_nonzero(self.failures)
        fail_rate = nb_fail/nb_runs
        self.fail_summary = (nb_runs, nb_fail, fail_rate)
        
        #Isp and twr arrays corresponding to failures
        if nb_fail>0:
            
            arr_fail = np.nonzero(self.failures)
            self.Isp_fail = np.take(self.Isp, arr_fail[1])
            self.twr_fail = np.take(self.twr, arr_fail[0])
            
            return self.fail_summary, self.Isp_fail, self.twr_fail
        
        else:
            
            return self.fail_summary
                
    def remove_failures(self):
        
        """
        removes the values corresponding to failures from all the defined matrices
        """
        
        self.m_final[self.failures==1.0] = np.nan
        self.m_prop[self.failures==1.0] = np.nan
        self.prop_frac[self.failures==1.0] = np.nan
        self.tof[self.failures==1.0] = np.nan
        self.alpha0[self.failures==1.0] = np.nan
            
    def get_results(self):
        
        """
        returns a dictionary with the following keys:
            
            H:                  target orbit altitude (m)
            rf:                 final orbit radius (m)
            Isp:                specific impulses array (s)
            twr:                thrust/initial weight ratios array
            tof:                time of flights matrix (s)
            m0:                 initial spacecraft mass (kg)
            m_final:            final mass matrix (kg)
            m_prop:             propellant masses matrix (kg)
            prop_frac:          propellant fractions matrix
            alpha0:             initial thrust direction matrix (rad)
            hist:               optimal states variables and controls histories
            
            idx_opt:            optimal trajectory indexes (twr, Isp)
            failures:           recorded failures matrix
            fail_summary:       tuple with number of runs, number of failures and failures rate
            Isp_fail:           array with Isp values corresponding to failures
            twr_fail:           array with twr values corresponding to failures
        """
        
        self.remove_failures()
        d = sstoAnalyzer.get_results(self)
        
        d['idx_opt'] = self.idx_opt
        d['failures'] = self.failures
        d['fail_summary'] = self.fail_summary
        
        if self.fail_summary[1]>0: #at least one failure
            d['Isp_fail'] = self.Isp_fail
            d['twr_fail'] = self.twr_fail
        
        return d
    
    def run_optimizer(self):
        
        """
        creates an instance of the class sstoOptimizer and solves the optimal control problem for every
        couple (Isp, twr) storing the optimal trajectory and parameters as class attributes
        """
                
        self.init_matrix() #initialize the matrices to store the different results
        m_opt=-1 #initialize a fake final mass
        
        optimizer = self.set_optimizer() #create an optimizer instance
        
        execution_start = time() #measure the execution time
        
        #solve the optimal control problem for every couple (Isp, twr)
        for i in range(np.size(self.twr)):
            for j in range(np.size(self.Isp)):
                
                print("\nIsp: " + str(self.Isp[j]) + " s\ttwr: " + str(self.twr[i]) + "\n") #current Isp, twr values
                
                failed = optimizer.run_optimizer(self.F[i], self.w[j], self.bcs, self.t0)
                
                if not failed: #current optimization successful
                    
                    p = optimizer.get_pbm()
                    hist = self.get_history(p)
                    m_final = self.fill_matrix(hist, i, j)
                    
                    #compare the current final mass with its maximum value to determine if the solution is better
                    if m_final>m_opt:
                        m_opt = m_final
                        hist_opt = hist
                        idx_opt = (i,j) #(twr, Isp)
                    
                else:
                    self.failures[i,j] = 1.0 #current optimization failed, update corresponding matrix
                    
        #print out the execution time
        execution_end = time()
        elapsed_time = execution_end-execution_start
        print("\nOptimization done in " + str(elapsed_time) + " s\n")
        
        #print out the failure rate
        self.get_failures()
        print(str(self.fail_summary[1]) + " failures in " + str(self.fail_summary[0]) + " runs")
        print("Failure rate: " + str(self.fail_summary[2]*100) + " %\n")
        
        #save optimal trajectory
        self.hist = hist_opt
        self.idx_opt = idx_opt
        
        #display the results
        self.display_summary()
        self.plot_all(self.hist, self.rf)
        
    def display_summary(self):
        
        """
        prints out a summary of the optimal trajectory parameters
        """
        
        print('')
        print('{:^54s}'.format('Optimal ascent trajectory to LLO'))
        print('{:<30s}{:>24.2f}'.format('orbit altitude (km)', self.H/1e3))
        print('{:<30s}{:>24.2f}'.format('specific impulse (s)', self.Isp[self.idx_opt[1]]))
        print('{:<30s}{:>24.2f}'.format('initial thrust/wight ratio', self.twr[self.idx_opt[0]]))
        print('{:<30s}{:>24.2f}'.format('time of flight (s)', self.tof[self.idx_opt[0], self.idx_opt[1]]))
        print('{:<30s}{:>24.2f}'.format('final/initial mass ratio (%)', self.m_final[self.idx_opt[0], self.idx_opt[1]]/self.m0*100))
        print('{:<30s}{:>24.2f}'.format('propellant fraction (%)', self.prop_frac[self.idx_opt[0], self.idx_opt[1]]*100))
        print('{:<30s}{:>24.2f}'.format('initial thrust direction (deg)', self.alpha0[self.idx_opt[0], self.idx_opt[1]]*180/np.pi))
        print('')
                        
    def plot_all(self, h, rf):
        
        """
        calls the following class methods to display the different plots:
            
            prop_frac_contour():        propellant fraction as function of twr and Isp
            states_plot(h):             altitude, angle, radial and tangential velocity vs time
            alt_plot(h):                altitude profile vs angle
            control_plot(h):            thrust direction vs time
            trajectory_plot(h, rf):     ascent trajectory and target LLO in the xy plane
        """
                
        if np.size(self.Isp)>1 and np.size(self.twr)>1: #propellant fraction as function of twr and Isp
            self.prop_frac_contour()

        self.states_plot(h) #state variables vs time
        self.alt_plot(h) #altitude profile vs angle
        self.control_plot(h) #control vs time
        self.trajectory_plot(h, rf) #trajectory in the xy plane
        
        plt.show()

    def prop_frac_contour(self):
        
        """
        plots a contour of the required propellant fraction as function of
        thrust/initial weight ratio and specific impulse
        """
        
        [X,Y]=np.meshgrid(self.twr, self.Isp, indexing='ij')

        fig, ax = plt.subplots()
        cs = ax.contour(X,Y,self.prop_frac, 25)
        ax.clabel(cs)
        ax.set_xlabel('Thrust/initial weight ratio')
        ax.set_ylabel('Isp (s)')
        ax.set_title('Propellant fraction')