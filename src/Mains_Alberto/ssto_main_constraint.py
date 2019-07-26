# -*- coding: utf-8 -*-
"""
The script defines the required constants and settings to numerically solve an optimal control problem
for the optimal ascent trajectory from the lunar surface to a circular Low Lunar Orbit (LLO) with a
constrained minimum safe altitude

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
from scipy.io import savemat, loadmat
from Analyzers.ssto_analyzer_constraint import sstoAnalyzerConstraint

#   (1) compute
#   (2) dispaly
ch = 1

const = {} #constants' dictionary

const['g0'] = 9.80665 #standard gravity acceleration (m/s^2)
const['mu'] = 4.902800066e12 #lunar standard gravitational parameter (m^3/s^2)
const['R'] = 1737.4e3 #lunar radius (m)
const['H'] = 0.05 #target orbit altitude
const['m0'] = 1.0 #initial spacecraft mass (kg)
const['lunar_radii'] = True #LLO altitude defined in lunar radii

const['hmin'] = 5e3 #minimum safe altitude
const['mc'] = 100. #path constraint shape

#optimizer settings
solver = 'ipopt' #NLP solver
solver_unconstrained = solver #NLP solver for the unconstrained optimal control problem
run_unconstrained = False #solve the unconstrained optimal control problem first and use it as initial guess
acc_guess = True #use an accurate initial guess provided by sstoGuess (varying thrust only)
alpha_rate2_cont = False #enforce rate2_continuity for control alpha
duration_bounds = True #impose bounds on the phase duration (minimum and maximum values)
debug = True #check the partial derivatives defined in the ODEs and explicitly simulate the obtained trajectory
exp_sim = True #explicit simulation to verify the results

#SLSQP specific settings
tol = 1e-6 #stopping tolerance
maxiter = 5000 #maximum number of iterations

#transcription settings
transcription = 'gauss-lobatto'
num_seg = 200
transcription_order = 3

#scaling parameters
scalers = (1e-3, 1e-6, 1, 1e-3, 1e-3, 1) #scalers for (time, r, theta, u, v, m)
defect_scalers = (10, 10, 1, 10, 10, 1) #defect scalers for (time, r, theta, u, v, m)

settings = {'solver':solver.upper(), 'solver_unconstrained':solver_unconstrained.upper(), 'tol':tol, 'maxiter':maxiter,
            'transcription':transcription, 'num_seg':num_seg, 'transcription_order':transcription_order, 'scalers':scalers,
            'defect_scalers':defect_scalers, 'top_level_jacobian':'csc', 'dynamic_simul_derivs':True, 'compressed':True,
            'debug':debug, 'acc_guess':acc_guess, 'alpha_rate2_cont':alpha_rate2_cont, 'duration_bounds':duration_bounds,
            'run_unconstrained':run_unconstrained, 'exp_sim':exp_sim}

#parameters
Isp = 450. #Isp (s)
twr = 2.1 #thrust/initial weight ratio
klim = (0.0, 1.0) #throttle limits
t0 = np.array([3., 4., 7.])*1e3 #time of flight initial guess and bounds (s)

if ch==1: #compute trajectory
    a = sstoAnalyzerConstraint(const, settings)
    a.set_params(Isp, twr, klim, t0) 
    p, ph = a.run_optimizer()    
    d = a.get_results()
    #savemat('../data_constraint/2D_case4',d)

elif ch==2: #saved data
    
    d = loadmat("../data_constraint/ipopt_200_new", squeeze_me=True, mat_dtype=True)
    a = sstoAnalyzerConstraint(const, settings)
    a.set_params(Isp, twr, klim, t0)
    a.set_results(d)
    
    a.display_summary()
    a.plot_all(a.hist, a.hist_exp, a.rf)
