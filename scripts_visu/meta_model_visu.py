"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.surrogate.meta_models import TwoDimAscConstMetaModel

rec_file = 'meta_model_test.pkl'

a = TwoDimAscConstMetaModel(rec_file=rec_file)

a.p['twr'] = 2.
a.p['Isp'] = 350.

a.p.run_model()

print(a.p['mm.m_prop'])
