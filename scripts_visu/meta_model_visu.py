"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.surrogate.meta_models import TwoDimAscConstMetaModel

rec_file = 'desc_var_mm.pkl'

a = TwoDimAscConstMetaModel(rec_file=rec_file, vec_size=1)

a.p['twr'] = [2.55]
a.p['Isp'] = [375]

a.p.run_model()
a.plot()

print('Total failures: ' + str(np.sum(a.failures)))
print(a.p['mm.m_prop'])
