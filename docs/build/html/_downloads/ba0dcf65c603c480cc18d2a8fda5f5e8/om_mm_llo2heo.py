"""
OpenMDAO MetaModel for LLO to HEO transfers
===========================================

This example computes the sampling grid and training points to assemble an OpenMDAO MetaModel for an LLO to HEO transfer
modeled as a finite departure burn to leave the initial LLO, a ballistic arc and a final impulsive burn to inject at the
apoapsis of the target HEO.
For each specific impulse value included in the grid, a continuation method can be employed to obtain all corresponding
solutions for decreasing thrust/weight ratio values.

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from latom.utils.pickle_utils import save
from latom.utils.primary import Moon
from latom.surrogate.om_metamodels_llo2heo import TwoDimLLO2ApoMetaModel, TwoDimLLO2ApoContinuationMetaModel

# MetaModel settings
continuation = True  # use continuation method over thrust/weight ratios
log_scale = False  # thrust/weight ratios equally spaced in logarithmic scale
distributed = False  # variables distributed across multiple processes
extrapolate = False  # extrapolation for out-of-bounds inputs
interp_method = 'scipy_cubic'  # interpolation method
training_data_gradients = True  # compute gradients wrt output training data
vec_size = 1  # number of points to evaluate at once
nb_samp = (50, 50)  # number of samples on which the actual solution is computed as (twr, Isp)
rec_file = 'example.pkl'  # name of the file in latom.data.metamodels in which the solution is serialized
rec_file_obj = 'example.pkl'  # name of the file in the working directory in which the object is serialized

moon = Moon()  # central attracting body

# trajectory
llo_alt = 100e3  # initial LLO altitude [m]
heo_rp = 3150e3  # target HEO periselene radius [m]
heo_period = 6.5655*86400  # target HEO period [s]

# grid limits
isp = [250., 495.]  # specific impulse [s]
twr = [0.05, 3.]  # initial thrust/weight ratio [-]

# NLP
transcription_method = 'gauss-lobatto'
segments = 200
order = 3
solver = 'IPOPT'
snopt_opts = {'Major feasibility tolerance': 1e-8, 'Major optimality tolerance': 1e-8,
              'Minor feasibility tolerance': 1e-8}

if continuation:
    mm = TwoDimLLO2ApoContinuationMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                                            training_data_gradients=training_data_gradients, vec_size=vec_size)
    mm.sampling(moon, twr, isp, llo_alt, None, transcription_method, segments, order, solver, nb_samp,
                snopt_opts=snopt_opts, rec_file=rec_file, t=heo_period, rp=heo_rp, log_scale=log_scale)
else:
    mm = TwoDimLLO2ApoMetaModel(distributed=distributed, extrapolate=extrapolate, method=interp_method,
                                training_data_gradients=training_data_gradients, vec_size=vec_size)
    mm.sampling(moon, twr, isp, llo_alt, None, transcription_method, segments, order, solver, nb_samp,
                snopt_opts=snopt_opts, rec_file=rec_file, t=heo_period, rp=heo_rp)

mm.plot()
save(mm, rec_file_obj)
