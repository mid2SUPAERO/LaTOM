"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.surrogate.meta_models import MetaModel

# MetaModel settings
distributed = False  # variables distributed across multiple processes
extrapolate = False  # extrapolation for out-of-bounds inputs

# interpolation method among slinear, lagrange2, lagrange3, cubic, akima, scipy_cubic, scipy_slinear, scipy_quintic
interp_method = 'slinear'

training_data_gradients = True  # compute gradients wrt output training data
vec_size = 5  # number of points to evaluate at once
rec_file = 'asc_const_mm.pkl'  # name of the file on which the solution is serialized
kind = 'final'  # quantity to display, prop for propellant fraction or final for final/initial mass ratio

a = MetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
              training_data_gradients=training_data_gradients, vec_size=vec_size, rec_file=rec_file)

a.p['twr'] = [1.9]
a.p['Isp'] = [345.]

a.p.run_model()
a.plot(nb_lines=50, log_scale=False, kind=kind)

print('Total failures: ' + str(np.sum(a.failures)))
print(a.p['mm.m_prop'])

# energy = a.d['energy']
# print(np.max(energy))

