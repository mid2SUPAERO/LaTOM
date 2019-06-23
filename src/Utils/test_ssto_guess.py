#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 08:19:26 2019

@authors:
    Alberto Fossa'
    Giuliana Miceli
    
Test the class sstoGuess
"""

import sys

if '..' not in sys.path:
    sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from time import time
from Utils.ssto_guess import sstoGuess

def fit_data(x,y,order=3,nb_points=100):
    
    z = np.polyfit(x, y, order)
    f = np.poly1d(z)

    x_fit = np.linspace(x[0], x[-1], nb_points)
    y_fit = f(x_fit)
    
    return x_fit, y_fit
    
#constants
g0 = 9.80665 #standard gravity acceleration (m/s^2)
mu = 4.902800066e12 #lunar standard gravitational parameter (m^3/s^2)
R = 1737.4e3 #lunar radius (m)
H_rel = 0.05 #target orbit altitude (lunar radii)
m0 = 1.0 #initial spacecraft mass (kg)
const = {'g0':g0, 'mu':mu, 'R':R, 'H':H_rel, 'm0':m0, 'lunar_radii':True}

#parameters
Isp = 450.0 #specific impulse (s)
twr = 2.1 #initial thrust/weight ratio

H = R*H_rel #target orbit altitude (m)
W0 = m0*(mu/R**2) #initial weight (N)
F = W0*twr #thrust (N)
w = g0*Isp #exaust velocity (m/s)

#test
N = np.arange(200,201,1)
nb_sol = np.size(N)
M = np.zeros(nb_sol)

for i in range(0,nb_sol):
    
    #run sstoGuess
    print('\nNumber of points:', N[i])
    t0 = time()
    trj = sstoGuess(const, F, w)
    tof, s_t = trj.compute_tof()
    t_vec = np.linspace(0,tof,N[i])
    s_sg, s_ht = trj.compute_trajectory(t=t_vec)
    t1 = time()

    #elapsed time and final mass
    dt = t1 - t0
    M[i]=trj.m_final
    print("\nInitial guess computed in " + str(dt) + " s")
    print("Final mass: " + str(trj.m_final/m0*100) + " %")
    
#trajectory plot
trj.plot_all()

if len(N)>2:
    
    #fit data
    xf, yf = fit_data(N, M)

    #plot final mass vs nb of nodes
    fig, ax = plt.subplots()
    ax.plot(N,M,'bo')
    ax.plot(xf, yf, 'r')
    ax.set_xlabel('number of nodes')
    ax.set_ylabel('final/initial mass')
    plt.show()
