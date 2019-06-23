# -*- coding: utf-8 -*-
"""
The script defines the required constants and settings to numerically solve an optimal control problem
for the optimal ascent trajectory from the lunar surface to a circular Low Lunar Orbit (LLO) in 3D

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
from Analyzers.ssto_analyzer3D import sstoAnalyzer_3D

#   (1) compute
#   (2) display
ch = 2

const = {} #constants' dictionary

const['g0'] = 9.80665 #standard gravity acceleration (m/s^2)
const['mu'] = 4.902800066e12 #lunar standard gravitational parameter (m^3/s^2)
R = 1737.4e3 #lunar radius (m)
const['R'] = R
const['m0'] = 1.0 #initial spacecraft mass (kg)

#optimizer settings
solver = 'ipopt' #NLP solver
attitude_rate_bounds = False #path constraints on attitude rate
duration_bounds = True #bounds on the phase duration (minimum and maximum values)
debug = True #check the partial derivatives defined in the ODEs and explicitly simulate the obtained trajectory
exp_sim = True #explicit simulation to verify the results
visible_moon = True # plot the moon on the trajectory plot

#SLSQP specific settings
tol = 1e-6 #stopping tolerance
maxiter = 5000 #maximum number of iterations

#transcription settings
transcription = 'gauss-lobatto'
num_seg = 20
transcription_order = 3

#scaling parameters
scalers = (1e-2, 1e-6, 1e-3, 1) #scalers for (time, r, v, m)
defect_scalers = (10, 10, 10, 1) #defect scalers for (time, r, v, m)

final_bcs = 'he' #impose final BCs on classical orbital elements (coe) or specific angular momentum and eccentricity vectors (he)

settings = {'solver':solver.upper(),'tol':tol, 'maxiter':maxiter, 'transcription':transcription, 'num_seg':num_seg,
            'transcription_order':transcription_order, 'scalers':scalers, 'defect_scalers':defect_scalers, 'top_level_jacobian':'csc',
            'dynamic_simul_derivs':True, 'compressed':True, 'debug':debug, 'duration_bounds':duration_bounds,
            'attitude_rate_bounds':attitude_rate_bounds, 'final_bcs':final_bcs, 'exp_sim':exp_sim, 'visible_moon':visible_moon}

#parameters
Isp = 450. #Isp (s)
twr = 2.1 #thrust/initial weight ratio
kmin = 0.0  #minimum thrust throttle / If kmin=1 constant thrust

#initial guess
t0 = np.array([2., 4., 6.])*1e3 #time of flight initial guess and bounds (s)
u0 = np.array([[0.8, 0.2, 0.0],[0.01, 0.01, 0.0]]) #thrust direction initial guess as [[ux0, uy0, uz0], [uxf, uyf, uzf]]
R0 = np.array([R, 0.0, 0.0]) #initial position R0 defined as (x0, y0, z0)

#target orbit coe defined as (a, e, i, raan, w, ta)
rp = R*1.05
e = 0.0
a = rp/(1-e)
coe = np.array([a, e, 0.0, 0.0, 0.0, 180.])

if ch==1: #compute trajectory
    a3d = sstoAnalyzer_3D(const, settings)
    a3d.set_params(Isp, twr, kmin, t0, u0) 
    a3d.set_initial_position(R0)
    a3d.set_final_coe(coe, angle_unit='deg')
    p, ph = a3d.run_optimizer()    
    d = a3d.get_results()
    #savemat('../data_3D/3Dtrajectory_case10',d)

elif ch==2: #saved data
    
    d = loadmat("../data_3D/3Dtrajectory_case9", squeeze_me=True, mat_dtype=True)
    a3d = sstoAnalyzer_3D(const, settings)
    a3d.set_params(Isp, twr, kmin, t0, u0)
    a3d.set_results(d)
    
    a3d.display_summary()
    a3d.plot_all(a3d.hist, a3d.hist_exp)
