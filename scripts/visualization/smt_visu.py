"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.surrogate.smt_surrogates import SurrogateModel

fid_lhs = 'desc_vland_lhs.pkl'
fid_full = 'asc_const_full.pkl'
kind = 'prop'  # 'final'

sm_lhs = SurrogateModel('KRG', rec_file=fid_lhs)
sm_full = SurrogateModel('LS', rec_file=fid_full)

twr = np.linspace(2, 3, 5)
isp = np.linspace(300, 400, 5)
m_prop = sm_lhs.evaluate(isp, twr)
print(m_prop)

sm_lhs.plot(2500, kind=kind, nb_lines=40)
sm_full.plot(kind=kind)
