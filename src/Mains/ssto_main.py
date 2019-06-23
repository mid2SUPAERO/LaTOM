# -*- coding: utf-8 -*-
"""
The script defines the required constants and settings to numerically solve an optimal control problem
for the optimal ascent trajectory from the lunar surface to a circular Low Lunar Orbit (LLO)

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
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from Analyzers.ssto_analyzer import sstoAnalyzer, sstoAnalyzerThrottle, sstoAnalyzerMatrix
from Utils.ssto_matlab_interface import matlab_interface

"""
choose one of the following:
    
    (1) Isp and twr single values, constant thrust (no throttle)
    (2) Isp and twr single values, varying thrust (throttle)
    
    (3) saved data for single Isp, twr with constant thrust
    (4) saved data for single Isp, twr with varying thrust
    
    (5) Isp and twr custom multiple values
    (6) Isp and twr multiple values from analytical results
    (7) saved data for multiple Isp, twr with constant thrust
"""

ch = 7

#constants' dictionary
const = {}

const['g0'] = 9.80665 #standard gravity acceleration (m/s^2)
const['mu'] = 4.902800066e12 #lunar standard gravitational parameter (m^3/s^2)
const['R'] = 1737.4e3 #lunar radius (m)
const['H'] = 0.05 #target orbit altitude
const['m0'] = 1.0 #initial spacecraft mass (kg)
const['lunar_radii'] = True #LLO altitude defined in lunar radii

#optimizer settings
solver = 'ipopt' #NLP solver
acc_guess = True #use an accurate initial guess provided by sstoGuess (varying thrust only)
alpha_rate2_cont = False #enforce rate2_continuity for control alpha
duration_bounds = True
debug = True #check the partial derivatives defined in the ODEs
exp_sim = True #explicit simulation with optimal control profile

#SLSQP specific settings
tol = 1e-6 #stopping tolerance
maxiter = 5000 #maximum number of iterations

#transcription settings
transcription = 'gauss-lobatto'
num_seg = 10
transcription_order = 3

#scaling parameters
scalers = (1e-2, 1e-6, 1, 1e-3, 1e-3, 1) #scalers for (time, r, theta, u, v, m)
defect_scalers = (10, 10, 1, 10, 10, 1) #defect scalers for (time, r, theta, u, v, m)

settings = {'solver':solver.upper(), 'tol':tol, 'maxiter':maxiter, 'transcription':transcription, 'num_seg':num_seg,
            'transcription_order':transcription_order, 'scalers':scalers, 'defect_scalers':defect_scalers,
            'top_level_jacobian':'csc', 'dynamic_simul_derivs':True, 'compressed':True, 'debug':debug, 'acc_guess':acc_guess,
            'alpha_rate2_cont':alpha_rate2_cont, 'duration_bounds':duration_bounds, 'exp_sim':exp_sim}

if ch==1: #Isp and twr single values, constant thrust (no throttle)
    
    Isp = 450. #Isp (s)
    twr = 2.1 #thrust/initial weight ratio
    t0 = np.array([0.1, 0.5, 1.5])*1e3 #time of flight initial guess and bounds (s)
    
    a = sstoAnalyzer(const, settings)
    a.set_params(Isp, twr, t0)
    
    p, ph = a.run_optimizer()
    
    a.run_exp_sim()
    
    d = a.get_results()
    #savemat("../data/2D_case1", d)
    
elif ch==2: #Isp and twr single values, varying thrust (throttle)
    
    Isp = 450. #Isp (s)
    twr = 2.1 #thrust/initial weight ratio
    klim = (0.0, 1.0) #throttle limits
    #t0 = np.array([3., 4., 5.])*1e3 #time of flight initial guess and bounds (s)
    t0 = np.array([2.0, 4.0, 5.0])*1e3
    
    settings['num_seg'] = 100
    settings['transcription_order'] = 3
    settings['scalers'] = (1e-3, 1e-6, 1, 1e-3, 1e-3, 1) #scalers for (time, r, theta, u, v, m)
    settings['defect_scalers'] = (10, 10, 1, 10, 10, 1) #defect scalers for (time, r, theta, u, v, m)
    
    a = sstoAnalyzerThrottle(const, settings)
    a.set_params(Isp, twr, klim, t0)
    
    p, ph = a.run_optimizer()
    
    d = a.get_results()
    #savemat("../data/2D_case3", d)
        
elif ch==3: #saved data for single Isp, twr with constant thrust
    
    d = loadmat("../data/2D_case1", squeeze_me=True, mat_dtype=True)
    a = sstoAnalyzer(const, settings)
    a.set_results(d)
    
    a.display_summary()
    a.plot_all(a.hist, a.hist_exp, a.rf)
    
elif ch==4: #saved data for single Isp, twr with varying thrust
    
    d = loadmat("../data/2D_case2", squeeze_me=True, mat_dtype=True)
    a = sstoAnalyzerThrottle(const, settings)
    
    Isp = 450. #Isp (s)
    twr = 2.1 #thrust/initial weight ratio
    klim = (0.0, 1.0) #throttle limits
    #t0 = np.array([3., 4., 5.])*1e3 #time of flight initial guess and bounds (s)
    t0 = np.array([2.0, 4.0, 5.0])*1e3
    
    a.set_params(Isp, twr, klim, t0)
    a.set_results(d)
    
    a.display_summary()
    a.plot_all(a.hist, a.hist_exp, a.rf)
    
elif ch==5: #Isp and twr custom multiple values
    
    Isp = np.arange(300,501,10)
    twr = np.arange(1.2, 5.2, 0.2)
    t0 = np.array([0.1, 0.5, 1.5])*1e3 #time of flight initial guess and bounds (s)
    
    am = sstoAnalyzerMatrix(const, settings)
    am.set_params(Isp, twr, t0)
    
    am.run_optimizer() #run the optimizer
    
    d = am.get_results() #retrieve a dictionary with all the obtained results
    
    #savemat('../data/matrix_isp_300-500_twr_12-50_no_t_bounds_ipopt', d)
    
elif ch==6: #Isp and twr multiple values from analytical results
    
    #create a Matlab interface instance loading the results saved in the specified .mat file
    keys_mat = ('Isp_mat', 'twr_mat', 'prop_frac_mat', 'tof_mat', 'alpha0_mat')
    mi_mat = matlab_interface("../data/matlab_results_full.mat", keys=keys_mat)
    
    #set Isp and twr limits and return the correspondig arrays
    Isp_lim = (100., 600.)
    twr_lim = (1.0, 4.0)
    mi_mat.set_data_lim(Isp_lim, twr_lim)
    Isp, twr = mi_mat.get_Isp_twr()
    
    dt = 1000.0 #time of flight initial guess (s)
    
    am = sstoAnalyzerMatrix(const, settings)
    am.set_params(Isp, twr, dt)
    
    am.run_optimizer() #run the optimizer
    
    d = am.get_results() #retrieve a dictionary with all the obtained results
    
    #savemat('data/matrix_isp_all_twr4_lgl_5_11_new', d)
    mi = matlab_interface(d, mat_file=False) #create a Matlab interface instance with the obtained results
    
elif ch==7: #saved data for multiple Isp, twr
    
    #create a Matlab interface instance loading the results obtained in Dymos and retrieve the Isp and twr limits
    mi = matlab_interface('../data/matrix_isp_all_twr4_lgl_5_11_new.mat')
    Isp_lim, twr_lim = mi.get_lim()
    
    #create a Matlab interface instance loading the results obtained in Matlab and set the Isp and twr limits
    keys_mat = ('Isp_mat', 'twr_mat', 'prop_frac_mat', 'tof_mat', 'alpha0_mat')
    mi_mat = matlab_interface("../data/matlab_results_full.mat", keys=keys_mat)
    mi_mat.set_data_lim(Isp_lim, twr_lim)
    
    #maximum errors    
    err = mi.get_errors(mi_mat, display=True)
    
    #error plot
    z = np.abs(mi.data['prop_frac']-mi_mat.data['prop_frac_mat'])
    mi.generic_contour(z, 'Propellant fraction absolute error (x $10^-5$)\ndirect vs indirect methods', scale=1e5)
    
    #contour plots
    mi.data_contour('prop_frac', 'Propellant fraction - numerical solution')
    mi_mat.data_contour(keys_mat[2], 'Propellant fraction - analytical solution')
    mi.data_contour('tof', 'Time of flight (s) - numerical solution')
    mi_mat.data_contour(keys_mat[3], 'Time of flight (s) - analytical solution')
    mi.data_contour('alpha0', 'Initial thrust direction (deg) - numerical solution', scale=180/np.pi)
    mi_mat.data_contour(keys_mat[4], 'Initial thrust direction (deg) - analytical solution',
                        scale=180/np.pi)

    #create an sstoAnalyzer instance corresponding to the optimal trajectory saved in the Matlab interface
    a = sstoAnalyzer(const, settings)
    a.set_results(mi.get_opt_data())
    
    a.display_summary()
    a.plot_all(a.hist, None, a.rf)
    
    plt.show()  