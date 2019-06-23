# -*- coding: utf-8 -*-
"""
The script ssto_main.py defines the required constants and settings to numerically solve
the optimal control problem for a two-phases ascent trajectory from the Moon surface to a
circular Low Lunar Orbit (LLO) with an initial constrained vertical rise

referenced papers:
    
        Ma et al. "A unified trajectory optimization framework for lunar ascent",
    Advances in Engineering Software, 2016
        Ma et al. "Three-Dimensional Trajectory Optimization for Lunar Ascent Using Gauss Pseudospectral Method",
    AIAA Guidance, Navigation, and Control Conference, 2016

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

#adjust path
import sys

if '..' not in sys.path:
    sys.path.append('..')

from scipy.io import loadmat, savemat

from Analyzers.ssto_analyzer import sstoAnalyzer2p

"""
choose between:
    (1) 10s vertical rise phase
    (2) 500m height vertical rise phase
    (3) saved trajectory
"""
ch = 2

const = {} #constants' dictionary

const['g0'] = 9.80 #standard gravity acceleration (m/s^2)
const['mu'] = 4.9028e12 #lunar standard gravitational parameter (m^3/s^2)
const['R'] = 1738e3 #lunar radius (m)
const['m0'] = 4869 #initial spacecraft mass (kg)
const['m0_prop'] = 2500 #initial propellant mass (kg)
const['h0'] = 4.05 #initial altitude (m)
const['H'] = 1789.44e3 - const['R'] #target orbit altitude (m)
const['lunar_radii'] = False #LLO altitude defined in lunar radii

Isp = 309.50 #specific impulse (s)
F = 15435.33 #thrust (N)
W0 = const['m0']*(const['mu']/const['R']**2) #initial spacecraft weight (N)
twr = F/W0 #initial thrust/weight ratio

#optimizer settings
solver = 'ipopt' #NLP solver
debug = True #check the partial derivatives defined in the ODEs

#SLSQP specific settings
tol = 1e-6 #stopping tolerance
maxiter = 5000 #maximum number of iterations

transcription = 'gauss-lobatto'
num_seg_vert = 3
transcription_order_vert = 3
num_seg_horiz = 15
transcription_order_horiz = 3
maxiter = 500
scalers = (1e-2, 1e-6, 1, 1e-3, 1e-3, 1e-3) #scalers for (time, r, theta, u, v, m)
defect_scalers = (1, 10, 1, 1, 10, 10) #scalers for (time, r, theta, u, v, m)

settings = {'solver':solver.upper(), 'transcription':transcription, 'num_seg_vert':num_seg_vert, 'num_seg_horiz':num_seg_horiz,
            'transcription_order_vert':transcription_order_vert, 'transcription_order_horiz':transcription_order_horiz,
            'maxiter':maxiter, 'tol':tol, 'top_level_jacobian':'csc', 'dynamic_simul_derivs':True, 'compressed':True,
            'scalers':scalers, 'defect_scalers':defect_scalers, 'debug':debug, 'acc_guess':False, 'exp_sim':False}

if ch==1: #fixed duration
    
    settings['fixed'] = 'time'
    const['hs'] = 100 #transition altitude (m)
    const['us'] = 15.0 #transition radial velocity (m/s)
    t0 = (0.0, 10.0, 500.0, 1000.0) #time of flight guesses (s)
    
    a = sstoAnalyzer2p(const, settings)
    a.set_params(Isp, twr, t0)
    
    a.run_optimizer()
    d = a.get_results()
    savemat("../data/ma_fix_time_ipopt", d)
    
if ch==2: #fixed altitude

    settings['fixed'] = 'alt'
    
    const['hs'] = 500 #transition altitude (m)
    const['us'] = 40.0 #transition radial velocity (m/s)
    t0 = (0.0, 25.0, 470.0, 1000.0) #time of flight guesses (s)
    
    a = sstoAnalyzer2p(const, settings)
    a.set_params(Isp, twr, t0)
    
    a.run_optimizer()
    d = a.get_results()
    savemat("../data/ma_fix_alt_ipopt", d)      