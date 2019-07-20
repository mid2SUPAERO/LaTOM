# -*- coding: utf-8 -*-
"""
The script main_ramanan.py defines the required constants and settings to numerically solve
the optimal control problem for the optimal single-phase descent trajectory from a circular
Low Lunar Orbit (LLO) to the Moon surface

referenced papers:
    
    Mukundan et al. "Optimal Moon Landing Trajectory Design with Solid and Liquid Propulsion using SQP"
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
from scipy.io import savemat, loadmat

from Analyzers.descent_analyzer import descentAnalyzer, descentAnalyzerThrust, descentAnalyzerPeriapsis, descentAnalyzerThrottle

"""
choose one of the following:
    (1) single constant thrust and periapsis altitude (Ramanan)
    (2) single varying thrust and periapsis altitude (Ramanan)
    
    (3) multiple constant thrust levels and single periapsis altitude (Mukundan)
    (4) single constant thrust and multiple periapsis altitudes (Ramanan)
    
    (5) single constant thrust and periapsis altitude saved trajectory
    (6) single varying thrust and periapsis altitude saved trajectory
    (7) multiple thrust levels and single periapsis altitude saved trajectories
    (8) single thrust and multiple periapsis altitudes saved trajectories
"""
ch = 5

const = {} #constants' dictionary

const['g0'] = 9.80665 #standard gravity acceleration (m/s^2)
const['mu'] = 4.902800476e12 #lunar standard gravitational parameter (m^3/s^2)
const['R'] = 1738.0e3 #lunar radius (m)

const['m0'] = 300.0 #initial spacecraft mass (kg)
W0 = const['m0']*(const['mu']/const['R']**2) #initial spacecraft weight (N)
Isp = 310.0 #Isp (s)

const['H'] = 100e3 #parking orbit altitude (m)
const['hf'] = 3.0 #final altitude (m)
const['uf']  = 0.0 #final radial velocity (m/s)
const['vf'] = 0.0 #final tangential velocity (m/s)

#NLP solver settings
solver = 'slsqp' #NLP solver
acc_guess = False #use an accurate initial guess provided by sstoGuess (varying thrust only)
alpha_rate2_cont = True #enforce rate2_continuity for control alpha
debug = True #check the partial derivatives defined in the ODEs

#SLSQP specific settings
tol = 1e-6
maxiter = 1000

#transcription settings
transcription = 'gauss-lobatto'
num_seg = 5
transcription_order = 13

#NLP scaling settings
scalers = (1e-2, 1e-6, 1, 1e-2, 1e-3, 1e-2) #scalers for (time, r, theta, u, v, m)
defect_scalers = (1, 10, 1, 1, 10, 1) #scalers for (time, r, theta, u, v, m)

settings = {'solver':solver.upper(), 'tol':tol, 'maxiter':maxiter, 'transcription':transcription, 'num_seg':num_seg,
            'transcription_order':transcription_order, 'scalers':scalers, 'defect_scalers':defect_scalers,
            'top_level_jacobian':'csc', 'dynamic_simul_derivs':True, 'compressed':True, 'debug':debug, 'acc_guess':acc_guess,
            'alpha_rate2_cont':alpha_rate2_cont}


if ch==1: #single thrust and periapsis altitude (Ramanan)
    
    hp = 15e3 #intermediate orbit periapsis altitude (m) 
    F = 10000.0 #thrust (N)
    twr = F/W0 #initial thrust/weight ratio
    t0 = 40.0 #time of flight guess (s)
    
    settings['deorbit'] = True #consider or not the impulsive de-orbit burn
    
    a = descentAnalyzer(const, settings)
    a.set_params(Isp, twr, hp, t0)
    
    p = a.run_optimizer()
    d = a.get_results()
    #savemat("data/ramanan_lgl_5_13", d)
    
if ch==2: #single varying thrust and periapsis altitude (Ramanan)
    
    hp = 100e3 #intermediate orbit periapsis altitude (m) 
    F = 440.0 #thrust (N)
    twr = F/W0 #initial thrust/weight ratio
    klim = (0.0, 1.0) #throttle limits (kmin, kmax)
    t0 = 1000.0 #time of flight guess (s)
    
    settings['deorbit'] = False #consider or not the impulsive de-orbit burn
    settings['num_seg'] = 1
    settings['transcription_order'] = 29
    
    a = descentAnalyzerThrottle(const, settings)
    a.set_params(Isp, twr, hp, klim, t0)
    
    p = a.run_optimizer()
    d = a.get_results()
    #savemat("data/ramanan_lgl_1_29_k00", d)
    
if ch==3: #multiple thrust levels and single periapsis altitude (Mukundan)
    
    hp = 15e3 #periapsis altitude (m)
    F = np.array([0.44, 0.88, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0])*1e3 #thrust array (N)
    twr = F/W0 #initial thrust/weight ratio
    t0 = np.array([1000.0, 500.0, 300.0, 150.0, 120.0, 80.0, 40.0, 40.0, 30.0]) #time of flight guess (s)
    
    settings['deorbit'] = True #consider or not the impulsive de-orbit burn
    
    at = descentAnalyzerThrust(const, settings)
    at.set_params(Isp, twr, hp, t0)
    
    at.run_optimizer()
    d = at.get_results()
    #savemat("data/mukundan_thrust_lgl_5_13", d)
    
if ch==4: #single thrust and multiple periapsis altitudes (Ramanan)

    hp1 = np.array([100., 50.])
    hp2 = np.arange(25.0, 0.0, -5.0)
    hp = np.concatenate((hp1,hp2))*1e3 #periapsis altitudes array (m)
    
    F = 440.0 #thrust (N)
    twr = F/W0 #initial thrust/weight ratio
    t0 = np.ones(np.size(hp))*1e3 #time of flight guess (s)
    t0[1:4] = 900
    
    settings['deorbit'] = True #consider or not the impulsive de-orbit burn
    
    ap = descentAnalyzerPeriapsis(const, settings)
    ap.set_params(Isp, twr, hp, t0)
    
    ap.run_optimizer()
    d = ap.get_results()
    #savemat("data/ramanan_hp_lgl_5_13", d)
    
if ch==5: #single thrust and periapsis altitude saved trajectory
    
    d = loadmat("../data/ramanan_lgl_5_13", squeeze_me=True, mat_dtype=True)
    settings['deorbit'] = True
    
    a = descentAnalyzer(const, settings)
    
    a.set_results(d)
    a.display_summary()
    a.plot_all(a.hist, a.idx_opt, a.F, a.rp, a.nbh, 'b', 'r')
    
if ch==6: #single varying thrust and periapsis altitude saved trajectory
    
    d = loadmat("../data/ramanan_lgl_1_29_k00", squeeze_me=True, mat_dtype=True)
    settings['deorbit'] = True
    
    a = descentAnalyzerThrottle(const, settings)
    
    a.set_results(d)
    a.display_summary()
    a.plot_all(a.hist, a.rp)
    
if ch==7: #multiple thrust levels and single periapsis altitude saved trajectories
    
    d = loadmat("../data/mukundan_thrust_lgl_5_13", squeeze_me=True, mat_dtype=True)
    settings['deorbit'] = True
    
    a = descentAnalyzerThrust(const, settings)
    
    a.set_results(d)
    a.display_summary()
    a.plot_all(a.hist, a.idx_opt, a.F, a.rp)
    
if ch==8: #single thrust and multiple periapsis altitudes saved trajectories
    
    d = loadmat("../data/ramanan_hp_lgl_5_13", squeeze_me=True, mat_dtype=True)
    settings['deorbit'] = True
    
    a = descentAnalyzerPeriapsis(const, settings)
    
    a.set_results(d)
    a.display_summary()
    a.plot_all(a.hist, a.idx_opt, a.hp)