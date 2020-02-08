"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.surrogate.meta_models import TwoDimAscConstMetaModel

rec_file = 'meta_model_test.pkl'

a = TwoDimAscConstMetaModel(rec_file=rec_file, vec_size=2)

a.p['twr'] = [2., 2.5]
a.p['Isp'] = [350., 400.]

a.p.run_model()
a.plot()

print('Total failures: ' + str(np.sum(a.failures)))
print(a.p['mm.m_prop'])
