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
from mpl_toolkits import mplot3d

from Optimizers.ssto_optimizer3D import sstoOptimizer_3D
from Analyzers.ssto_analyzer import sstoAnalyzer
from Utils.keplerian_orbit import KeplerianOrbit

class sstoAnalyzer_3D(sstoAnalyzer):
    
    def __init__(self, const, settings):
        
        self.g0 = const['g0'] #standard gravity acceleration (m/s^2)
        self.mu = const['mu'] #lunar standard gravitational parameter (m^3/s^2)
        self.R = const['R']   #lunar radius (m)
        self.m0 = const['m0'] #initial spacecraft mass (kg)
        self.W0 = self.m0*(self.mu/self.R**2) #initial spacecraft weight on the Moon surface (N)
    
        self.const = const
        self.settings = settings
            
    def set_params(self, Isp, twr, kmin, t0, u0):
        
        """
        defines the Isp, twr and kmin values for which solve the optimization problem and
        an initial guess for the required time of flight and thrust direction where:
            
            Isp:        specific impulse (s)
            twr:        thrust/initial weight ratio
            kmin:       minimum throttle value
            t0:         time of flight initial guess and bounds defined as (lb, tof, ub) (s)
            u0:         thrust direction guess defined as [[ux0, uy0, uz0], [uxf, uyf, uzf]]
        """
        
        self.Isp = Isp #specific impulse (s)
        self.twr = twr #thrust/initial weight ratio
        self.Tmax = self.twr*self.W0 #maximum thrust (N)
        self.Tmin = self.Tmax*kmin #minimum thrust (N)
        self.u0 = u0 #thrust direction initial guess
        self.t0 = t0 #time of flight initial guess and bounds (s)
                
    def set_initial_position(self, R0):
        
        """
        specifies the initial spacecraft position as R=[x, y, z]
        """
        
        self.R0 = R0
        
    def set_final_coe(self, coe, angle_unit='rad'):
        
        """
        specifies the 5 orbital elements (a, e, i, raan, aop) for the target orbit and an initial guess for the target true anomaly
        and computes the corresponding state vector, specific angular momentum vector and eccentricity vector
        """
        
        self.coe = coe
        self.finalKepOrb = KeplerianOrbit(self.mu)
        self.finalKepOrb.set_coe_vector(coe, angle_unit=angle_unit)
        self.Rf, self.Vf = self.finalKepOrb.get_sv()
        self.Ef = self.finalKepOrb.get_eccentricity_vector()
        self.Hf = self.finalKepOrb.get_angular_momentum()
        
    def run_optimizer(self):
        
        """
        solves the optimal control problem storing the following results as class attributes:
            
            pbm:    OpenMDAO Problem class instance corresponding to the obtained optimal solution
            ph:     Dymos Phase class instance corresponding to the obtained optimal solution
            hist:   array with the states and control variables history [t, x, y, z, vx, vy, vz, m, Tx, Ty, Tz]
        """
        
        optimizer =  sstoOptimizer_3D(self.const, self.settings)
        
        execution_start = time() #measure the execution time
        
        if self.settings['final_bcs'] == 'coe': #impose final BCs on COE, not recommended
            failed = optimizer.run_optimizer_coe(self.Isp, self.Tmin, self.Tmax, self.u0, self.t0, self.R0, self.Rf, self.Vf, self.coe)
        elif self.settings['final_bcs'] == 'he': #impose final BCs on H and E vectors
            failed = optimizer.run_optimizer_he(self.Isp, self.Tmin, self.Tmax, self.u0, self.t0, self.R0, self.Rf, self.Vf, self.Hf, self.Ef)
        else:
            print("\nFinal BCs type not recognized! Choose one between coe and he\n")
        
        execution_end = time()
        elapsed_time = execution_end-execution_start
        
        if not failed:
            
            print("\nOptimization done in " + str(elapsed_time) + " s\n") 
            
            #retrieve the results
            self.pbm = optimizer.get_pbm()
            self.ph = optimizer.get_phase()
            self.hist, self.R_sol, self.V_sol = self.get_history(self.pbm)
            self.coe_sol, self.H_sol, self.E_sol = self.check_insertion(self.R_sol, self.V_sol)
            self.get_scalars(self.hist)
            
            #explicitly simulate the obtained trajectory (optional)
            if self.settings['exp_sim']:
                self.run_exp_sim()
            else:
                self.hist_exp = None
                
            #display the results
            self.display_summary()
            self.plot_all(self.hist, self.hist_exp)
                        
            return self.pbm, self.ph
                                    
        else:
            print("\nOptimization failed!\n")
            
    def run_exp_sim(self):
        
        """
        explicitly integrate the trajectory with the implicitly obtained controls histories
        """
        
        #explicitly simulate the trajectory
        self.p_exp = self.ph.simulate(times_per_seg=None, atol=1e-12, rtol=1e-12)
        self.hist_exp, self.R_exp, self.V_exp = self.get_history(self.p_exp)
        
        #evaluate the error between the two trajectories
        self.err = self.hist[1:]-self.hist_exp[1:]
        self.max_err = np.nanmax(np.absolute(self.err), axis=1)
        self.rel_err = self.max_err/np.nanmax(self.hist[1:], axis=1)

    def set_results(self, d):
        
        """
        saves as class attributes the results for an already computed trajectory stored in a given dictionary d
        """
        
        self.R0 = d['R0']
        self.coe = d['coe']
        self.coe_sol = d['coe_sol']
        self.Isp = d['Isp']
        self.twr = d['twr']
        self.tof = d['tof']
        self.m0 = d['m0']
        self.m_final = d['m_final']
        self.m_prop = d['m_prop']
        self.prop_frac = d['prop_frac']
        self.hist = d['hist']
        
        try:
            self.Ef = d['Ef']
            self.Hf = d['Hf']
            self.E_sol = d['E_sol']
            self.H_sol = d['H_sol']
        except:
            pass
        
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
        and returns them in a unique array hist defined as [t, r, x, y, z, vx, vy, vz, m, Tx, Ty, Tz, T, u]
        """
        
        t = pbm.get_val('phase0.timeseries.time_phase').flatten()
        
        x = pbm.get_val('phase0.timeseries.states:x').flatten()
        y = pbm.get_val('phase0.timeseries.states:y').flatten()
        z = pbm.get_val('phase0.timeseries.states:z').flatten()
        
        vx = pbm.get_val('phase0.timeseries.states:vx').flatten()
        vy = pbm.get_val('phase0.timeseries.states:vy').flatten()
        vz = pbm.get_val('phase0.timeseries.states:vz').flatten()
        
        m = pbm.get_val('phase0.timeseries.states:m').flatten()
        
        ux = pbm.get_val('phase0.timeseries.controls:ux').flatten()
        uy = pbm.get_val('phase0.timeseries.controls:uy').flatten()
        uz = pbm.get_val('phase0.timeseries.controls:uz').flatten()
        
        r = pbm.get_val('phase0.timeseries.r').flatten()
        u = pbm.get_val('phase0.timeseries.u2').flatten()**0.5
        
        T = pbm.get_val('phase0.timeseries.controls:T').flatten()
        
        Tx = T*ux
        Ty = T*uy
        Tz = T*uz
        
        hist = np.array([t, r, x, y, z, vx, vy, vz, m, Tx, Ty, Tz, T, u])
        
        #final state vector
        R = np.array([x[-1], y[-1], z[-1]])
        V = np.array([vx[-1], vy[-1], vz[-1]])
        
        return hist, R, V
    
    def check_insertion(self, R, V):
                
        self.solKepOrb = KeplerianOrbit(self.mu)
        self.solKepOrb.set_sv(R, V)
        coe_sol = self.solKepOrb.get_coe_vector(angle_unit='deg')
        H_sol = self.solKepOrb.get_angular_momentum()
        E_sol = self.solKepOrb.get_eccentricity_vector()
        
        return coe_sol, H_sol, E_sol
        
    def get_scalars(self, hist):
        
        """
        retrieves from a given histories' array hist the following quantities:
            
            m_final:        final spaecraft mass (kg)
            m_prop:         propellant mass (kg)
            prop_frac:      propellant fraction
            tof:            time of flight (s)
        """

        self.m_final = hist[8,-1] #final mass (kg)
        self.m_prop = self.m0 - self.m_final #propellant mass (kg)
        self.prop_frac = self.m_prop/self.m0 #propellant fraction
        self.tof = hist[0,-1] #time of flight (s)
        
    def get_results(self):
        
        """
        returns a dictionary with the following keys:
            
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
        
        d = {'R0':self.R0, 'coe':self.coe, 'Isp':self.Isp, 'twr':self.twr, 'tof':self.tof, 'm0':self.m0,
             'm_final':self.m_final, 'm_prop':self.m_prop, 'prop_frac':self.prop_frac, 'hist':self.hist,
             'coe_sol':self.coe_sol, 'Hf':self.Hf, 'Ef':self.Ef, 'H_sol':self.H_sol, 'E_sol':self.E_sol}
        
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
        except:
            pass
        
        print('')
        
        #print target and achieved LLO classical orbital elements
        head = ['semimajor axis (km)', 'eccentricity', 'inclination (deg)', 'right ascension (deg)',
                'argument of periapsis (deg)', 'true anomaly (deg)']
        
        print('{:^54s}'.format('Target LLO'))
        print('{:<30s}{:>12s}{:>12s}'.format('Orbital parameter','Target','Achieved'))
        print('{:<30s}{:>12.2f}{:>12.2f}'.format(head[0], self.coe[0]/1e3, self.coe_sol[0]/1e3))
        for i in range(1,6):
            print('{:<30s}{:>12.2f}{:>12.2f}'.format(head[i], self.coe[i], self.coe_sol[i]))
        print('')
        
        #print target and achieved LLO specific angular momentum and eccentricity vector H, E
        headH = ['hx (km^2/s)', 'hy (km^2/s)', 'hz (km^2/s)']
        headE = ['ex', 'ey', 'ez']
        try:
            for i in range(3):
                print('{:<30s}{:>12.2e}{:>12.2e}'.format(headH[i], self.Hf[i]/1e6, self.H_sol[i]/1e6))
            for j in range(3):
                print('{:<30s}{:>12.2e}{:>12.2e}'.format(headE[j], self.Ef[j], self.E_sol[j]))
        except:
            pass
        
        print('')
    
        #print ascent trajectory data
        print('{:^54s}'.format('Optimal ascent trajectory to LLO'))
        print('{:<30s}{:>24.2f}'.format('specific impulse (s)', self.Isp))
        print('{:<30s}{:>24.2f}'.format('initial thrust/wight ratio', self.twr))
        print('{:<30s}{:>24.2f}'.format('time of flight (s)', self.tof))
        print('{:<30s}{:>24.2f}'.format('final/initial mass ratio (%)', self.m_final/self.m0*100))
        print('{:<30s}{:>24.2f}'.format('propellant fraction (%)', self.prop_frac*100))
        print('')
        
        #print error between implicit solution and explicit simulation
        try:
            self.display_error()
        except:
            pass
        
    def plot_all(self, h, h_exp):
        
        """
        calls the following class methods to display the different plots:
            
            r_plot(h, h_exp):           position components vs time
            v_plot(h, h_exp):           velocity components vs time
            alt_plot(h, h_exp):         altitude profile vs time
            thrust_plot(h):             thrust components vs time
            trajectory_plot(h):         ascent trajectory and target LLO in 3D
        """

        self.r_plot(h, h_exp) #position components vs time
        self.v_plot(h, h_exp) #velocity components vs time
        self.alt_plot(h, h_exp) #altitude profile vs time
        self.thrust_plot(h) #thrust components vs time
        self.trajectory_plot(h) #trajectory in 3D
        
        plt.show()
        
    def r_plot(self, h, h_exp=None):
        
        """
        plots the state variables (x, y, z) vs time
        """
        
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        
        #explicit simulation
        if h_exp is not None:
            axs[0].plot(h_exp[0], h_exp[2]/1e3, color='g')
            axs[1].plot(h_exp[0], h_exp[3]/1e3, color='g')
            axs[2].plot(h_exp[0], h_exp[4]/1e3, color='g')
        
        #implicit solution
        axs[0].plot(h[0], h[2]/1e3, 'o', color='b')
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('x (km)')
        axs[0].set_title('position along X')
        axs[0].grid()

        axs[1].plot(h[0], h[3]/1e3, 'o', color='b')
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('y (km)')
        axs[1].set_title('position along Y')
        axs[1].grid()
        
        axs[2].plot(h[0],  h[4]/1e3, 'o', color='b')
        axs[2].set_xlabel('time (s)')
        axs[2].set_ylabel('z (km)')
        axs[2].set_title('position along Z')
        axs[2].grid()
        
    def v_plot(self, h, h_exp=None):
        
        """
        plots the state variables (vx, vy, vz) vs time
        """
        
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        
        #explicit simulation
        if h_exp is not None:
            axs[0].plot(h_exp[0], h_exp[5]/1e3, color='g')
            axs[1].plot(h_exp[0], h_exp[6]/1e3, color='g')
            axs[2].plot(h_exp[0], h_exp[7]/1e3, color='g')
        
        #implicit solution
        axs[0].plot(h[0], h[5]/1e3, 'o', color='b')
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('vx (km/s)')
        axs[0].set_title('velocity along X')
        axs[0].grid()

        axs[1].plot(h[0], h[6]/1e3, 'o', color='b')
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('vy (km/s)')
        axs[1].set_title('velocity along Y')
        axs[1].grid()
        
        axs[2].plot(h[0],  h[7]/1e3, 'o', color='b')
        axs[2].set_xlabel('time (s)')
        axs[2].set_ylabel('vz (km/s)')
        axs[2].set_title('velocity along Z')
        axs[2].grid()
        
    def alt_plot(self, h, h_exp=None):
        
        """
        plots the altitude profile vs time
        """
        
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        #explicit simulation
        if h_exp is not None:
            alt_exp = (h_exp[1]-self.R)/1e3 #altitude (km)
            ax.plot(h_exp[0], alt_exp, color='g')
        
        #implicit solution
        alt = (h[1]-self.R)/1e3 #altitude (km)
        ax.plot(h[0], alt, 'o', color='b')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('h (km)')
        ax.set_title('Altitude profile')
        ax.grid()

    def thrust_plot(self, h):
        
        """
        plots the thrust components vs time
        """
        
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
                    
        axs[0].plot(h[0], h[9], 'o', color='r')
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('Tx (N)')
        axs[0].set_title('Thrust component along X')
        axs[0].grid()
        
        axs[1].plot(h[0], h[10], 'o', color='r')
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('Ty (N)')
        axs[1].set_title('Thrust component along Y')
        axs[1].grid()
        
        axs[2].plot(h[0], h[11], 'o', color='r')
        axs[2].set_xlabel('time (s)')
        axs[2].set_ylabel('Tz (N)')
        axs[2].set_title('Thrust component along Z')
        axs[2].grid()
            
    def trajectory_plot(self, h):
        
        """
        plots the ascent trajectory in 3D
        """
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        if self.settings['visible_moon'] == True:
        
            #Moon surface (km)
            r_moon = self.R/1e3
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = r_moon * np.outer(np.cos(u), np.sin(v))
            y = r_moon * np.outer(np.sin(u), np.sin(v))
            z = r_moon * np.outer(np.ones(np.size(u)), np.cos(v))
    
            # Plot the surface
            ax.plot_surface(x, y, z, color='0.5', alpha=0.2)
            ax.get_autoscale_on()
            ax.get_frame_on()
            
            
        #target orbit
        n = 100
        ta = np.linspace(0.0, 360.0, n)
        x_tgt = np.zeros(n)
        y_tgt = np.zeros(n)
        z_tgt = np.zeros(n)
        kepOrb = KeplerianOrbit(self.mu)
        
        for i in range(n):
            coe = np.concatenate((self.coe[:5], [ta[i]]))
            kepOrb.set_coe_vector(coe, angle_unit='deg')
            R, V = kepOrb.get_sv()            
            x_tgt[i] = R[0]/1e3
            y_tgt[i] = R[1]/1e3
            z_tgt[i] = R[2]/1e3
            
        #ascent trajectory
        ax.plot3D(h[2]/1e3, h[3]/1e3, h[4]/1e3, label='ascent trajectory')
        ax.plot3D(x_tgt, y_tgt, z_tgt, label='target orbit')
        ax.scatter3D(h[2,0]/1e3, h[3,0]/1e3, h[4,0]/1e3, color='k', label='launch site')
        ax.scatter3D(h[2,-1]/1e3, h[3,-1]/1e3, h[4,-1]/1e3, color='r', label='insertion point')

        #axes limits
        m = np.max(np.concatenate((x_tgt, y_tgt, z_tgt)))
        ax.set_xlim(-m*1.1, m*1.1)
        ax.set_ylim(-m*1.1, m*1.1)
        ax.set_zlim(-m*1.1, m*1.1)
        
        #title, legend, labels
        ax.legend(bbox_to_anchor=(1, 1), loc=2)
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel('z (km)')
        ax.set_title('Optimal ascent trajectory')