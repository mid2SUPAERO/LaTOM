"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.primary import Moon
from rpfm.surrogate.meta_models_llo2heo import TwoDimLLO2ApoMetaModel

# MetaModel settings
distributed = False  # variables distributed across multiple processes
extrapolate = False  # extrapolation for out-of-bounds inputs
interp_method = 'scipy_cubic'  # interpolation method
training_data_gradients = True  # compute gradients wrt output training data
vec_size = 1  # number of points to evaluate at once
nb_samp = (2, 2)  # number of samples on which the actual solution is computed
rec_file = 'test.pkl'  # name of the file on which the solution is serialized

moon = Moon()  # central attracting body

# trajectory
llo_alt = 100e3  # initial LLO altitude [m]
heo_rp = 3150e3  # target HEO periselene radius [m]
heo_period = 6.5655*86400  # target HEO period [s]

# grid limits
isp = [250., 500.]  # specific impulse [s]
twr = [1.1, 4.0]  # initial thrust/weight ratio [-]

# NLP
method = 'gauss-lobatto'
segments = 20
order = 3
solver = 'SNOPT'
snopt_opts = {'Major feasibility tolerance': 1e-8, 'Major optimality tolerance': 1e-8,
              'Minor feasibility tolerance': 1e-8}

mm = TwoDimLLO2ApoMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                            training_data_gradients=training_data_gradients, vec_size=vec_size)
mm.sampling(moon, twr, isp, llo_alt, None, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
            rec_file=rec_file, t=heo_period, rp=heo_rp)
