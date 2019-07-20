# -*- coding: utf-8 -*-
"""
The script main_remesh.py defines the required constants and settings to numerically solve
the optimal control problem for the optimal two-phases descent trajectory from a circular
Low Lunar Orbit (LLO) to the Moon surface

referenced papers:
    
    Remesh et al. "Fuel Optimum Lunar Soft Landing Trajectory Design Using Different Solution Schemes"
    Ramanan et al. "Analysis of optimal strategies for soft landing on the Moon from lunar parking orbits"

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

#adjust path
import sys

if '..' not in sys.path:
    sys.path.append('..')

import numpy as np

from scipy.io import loadmat, savemat

from Analyzers.descent_analyzer2p import descentAnalyzer2p, descentAnalyzerThrust2p, descentAnalyzerThrottle2p

"""
choose one of the following:
    (1) single thrust (Remesh)
    (2) multiple thrust levels (Remesh)
    (3) single thrust (Ramanan)
    (4) varying thrust (Ramanan)
    
    (5) single thrust saved trajectory
    (6) multiple thrust levels saved trajectories
"""
ch = 5

const = {} #constants' dictionary

const['g0'] = 9.80665 #standard gravity acceleration (m/s^2)
const['mu'] = 4.902800476e12 #lunar standard gravitational parameter (m^3/s^2)
const['R'] = 1738e3 #lunar radius (m)

const['m0'] = 880.0 #initial spacecraft mass (kg)
W0 = const['m0']*(const['mu']/const['R']**2) #initial spacecraft weight (N)
Isp = 315.0 #Isp (s)

const['H'] = 100e3 #parking orbit altitude (m)
const['hf'] = 3.0 #final altitude (m)
const['uf']  = -5.0 #final radial velocity (m/s)
const['vf']  = 0.0 #final tangential velocity (m/s)

#NLP solver settings
solver = 'slsqp' #NLP solver
acc_guess = False #use an accurate initial guess provided by sstoGuess (varying thrust only)
alpha_rate2_cont = True #enforce rate2_continuity for control alpha
debug = True #check the partial derivatives defined in the ODEs

#SLSQP specific settings
tol = 1e-6
maxiter = 1000

#transcription settings (23, 5)
transcription = 'gauss-lobatto'
num_seg_horiz = 1
transcription_order_horiz = 23
num_seg_vert = 1
transcription_order_vert = 5

#NLP scaling settings
scalers = (1e-2, 1e-6, 1, 1e-2, 1e-3, 1e-2) #scalers for (time, r, theta, u, v, m)
defect_scalers = (1, 10, 1, 1, 10, 1) #scalers for (time, r, theta, u, v, m)

settings = {'solver':solver.upper(), 'tol':tol, 'maxiter':maxiter, 'transcription':transcription, 'num_seg_horiz':num_seg_horiz,
            'transcription_order_horiz':transcription_order_horiz, 'num_seg_vert':num_seg_vert,
            'transcription_order_vert':transcription_order_vert, 'scalers':scalers, 'defect_scalers':defect_scalers,
            'top_level_jacobian':'csc', 'dynamic_simul_derivs':True, 'compressed':True, 'debug':debug, 'acc_guess':acc_guess,
            'alpha_rate2_cont':alpha_rate2_cont}


if ch==1: #single thrust (Remesh)
    
    F = 880.0 #thrust (N)
    twr = F/W0 #initial thrust/weight ratio
    hp = 15e3 #intermediate orbit periapsis altitude (m)
    hs = 3.003e3 #switch altitude between horizontal and vertical braking phase (m)
    t0 = (1640.0, 1740.0) #time of flight guesses (s)
    
    settings['deorbit'] = True
    
    a = descentAnalyzer2p(const, settings)
    a.set_params(Isp, twr, hp, hs, t0)
    
    a.run_optimizer()
    d = a.get_results()
    #savemat("data/remesh_lgl_1_23", d)
    
if ch==2: #multiple thrust levels (Remesh)
    
    F = np.arange(1.0, 3.5, 0.5)*const['m0'] #thrust array (N)
    twr = F/W0 #initial thrust/weight ratio array
    hp = 15e3 #intermediate orbit periapsis altitude (m)
    hs = 3.003e3 #switch altitude between horizontal and vertical braking phase (m)    
    t0 = np.array([[1640.0, 1740.0],[900.0, 1000.0],[650.0,700.0],[500.0,550.0],[400.0,450.0]]) #tof guesses
    #t0 = np.array([[1700.0, 1800.0],[1200.0, 1300.0],[650.0,750.0],[530.0,570.0],[450.0,470.0]])
    
    settings['deorbit'] = True
    
    at = descentAnalyzerThrust2p(const, settings)
    at.set_params(Isp, twr, hp, hs, t0)
    
    at.run_optimizer()
    d = at.get_results()
    #savemat("data/remesh_thrust_lgl_1_23", d)
    
if ch==3: #single thrust (Ramanan)
    
    #modify some constants
    const['m0'] = 300.0 #initial spacecraft mass (kg)
    const['hf'] = 3.0 #final altitude (m)
    const['uf']  = 0.0 #final radial velocity (m/s)
    
    W0 = const['m0']*(const['mu']/const['R']**2) #initial spacecraft weight (N)
    Isp = 310.0 #Isp (s)
    
    F = 440.0 #thrust (N)
    twr = F/W0 #initial thrust/weight ratio
    hp = 15e3 #intermediate orbit periapsis altitude (m)
    hs = 3e3 #switch altitude between horizontal and vertical braking phase (m)
    #t0 = (1000.0, 1100.0) #time of flight guesses (s)
    t0 = (950., 1000.)
    
    settings['deorbit'] = True
    
    a = descentAnalyzer2p(const, settings)
    a.set_params(Isp, twr, hp, hs, t0)
    
    a.run_optimizer()
    d = a.get_results()
    #savemat("data/ramanan_2p_alt4_lgl_1_23", d)
    
if ch==4: #varying thrust (Ramanan)
    
    #modify some constants
    scalers = (1e-3, 1e-6, 1, 1e-2, 1e-3, 1e-2) #scalers for (time, r, theta, u, v, m)
    defect_scalers = (10, 10, 1, 1, 10, 1) #scalers for (time, r, theta, u, v, m)
    const['m0'] = 300.0 #initial spacecraft mass (kg)
    const['hf'] = 3.0 #final altitude (m)
    const['uf']  = 0.0 #final radial velocity (m/s)
    
    W0 = const['m0']*(const['mu']/const['R']**2) #initial spacecraft weight (N)
    Isp = 310.0 #Isp (s)
    
    F = 440.0 #thrust (N)
    twr = F/W0 #initial thrust/weight ratio
    klim = (0.1, 1.0) #throttle limits
    hp = 100e3 #intermediate orbit periapsis altitude (m)
    hs = 3e3 #transition altitude between horizontal and vertical braking phase (m)
    t0 = (5000.0, 5100.0) #time of flight guess (s)
    
    settings['deorbit'] = False
    
    a = descentAnalyzerThrottle2p(const, settings)
    a.set_params(Isp, twr, hp, hs, klim, t0)
    
    a.run_optimizer()
    d = a.get_results()
    #savemat("data/ramanan_single_deorbit", d)
    
if ch==5: #single thrust saved trajectory
    
    #fid = "../data/remesh_lgl_1_23"
    fid = "../data/ramanan_2p_alt3_lgl_1_23"
    #fid = "../data/ramanan_2p_alt4_lgl_1_23"
    d = loadmat(fid, squeeze_me=True, mat_dtype=True)
    settings['deorbit'] = True
    
    a = descentAnalyzer2p(const, settings)
    
    a.set_results(d)
    a.display_summary()
    a.plot_all(a.hist, a.rp, a.nbh, 'b', 'r')
    
if ch==6: #multiple thrust levels saved trajectories
    
    d = loadmat("../data/remesh_thrust_lgl_1_23", squeeze_me=True, mat_dtype=True)
    settings['deorbit'] = True
    
    a = descentAnalyzerThrust2p(const, settings)
    
    a.set_results(d)
    a.display_summary()
    a.plot_all(a.hist, a