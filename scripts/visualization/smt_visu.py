"""
SMT Surrogate Model visualization
=================================

This example loads an SMT SurrogateModel stored in `latom.data.smt`, predicts additional outputs based on existing
data and plots the corresponding response surface.

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.surrogate.smt_surrogates import SurrogateModel

fid_lhs = 'asc_const_lhs.pkl'  # file ID in latom.data.smt for surrogate model obtained with Latin Hypercube sampling
fid_full = 'asc_const_full.pkl'  # file ID in latom.data.smt for surrogate model obtained with Full-Factorial sampling
kind = 'prop'  # quantity to display, 'prop' for propellant fraction or 'final' for final/initial mass ratio

# surrogate modeling method (first argument) must be chosen among IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC
sm_lhs = SurrogateModel('KRG', rec_file=fid_lhs)  # instantiate surrogate model for LHS
sm_full = SurrogateModel('LS', rec_file=fid_full)  # instantiate surrogate model for FF

twr = np.linspace(2, 3, 5)  # twr values for prediction [-]
isp = np.linspace(300, 400, 5)  # Isp values for prediction [s]
m_prop = sm_lhs.evaluate(isp, twr)  # predicted propellant fraction [-]
print(m_prop)

# response surfaces
sm_lhs.plot(2500, kind=kind, nb_lines=40)
sm_full.plot(kind=kind)
