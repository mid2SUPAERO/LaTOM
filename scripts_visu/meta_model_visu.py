"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.surrogate.meta_models import TwoDimAscConstMetaModel

rec_file = 'mm.pkl'

a = TwoDimAscConstMetaModel(rec_file=rec_file, vec_size=2)

a.p['twr'] = [2., 2.5]
a.p['Isp'] = [350., 400.]

a.p.run_model()
a.plot()

print(a.p['mm.m_prop'])
