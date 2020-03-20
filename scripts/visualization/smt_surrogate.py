"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.surrogate.surrogate_2d import SurrogateModel

fid_lhs = 'asc_vtoff_lhs.pkl'
fid_full = 'desc_const_full.pkl'
kind = 'prop'  # 'final'

# 'IDW', 'KPLS', 'KPLSK', 'KRG', 'LS', 'QP', 'RBF', 'RMTB', 'RMTC'
sm_lhs = SurrogateModel('KRG', rec_file=fid_lhs)
sm_full = SurrogateModel('LS', rec_file=fid_full)

sm_lhs.plot(2500, kind=kind)
sm_full.plot(kind=kind)
