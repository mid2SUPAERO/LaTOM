"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.pickle_utils import save
from rpfm.utils.primary import Moon
from rpfm.surrogate.meta_models_llo2heo import TwoDimLLO2ApoMetaModel, TwoDimLLO2ApoContinuationMetaModel

# MetaModel settings
continuation = True  # use continuation method over thrust/weight ratios
log_scale = False  # thrust/weight ratios equally spaced in logarithmic scale
distributed = False  # variables distributed across multiple processes
extrapolate = False  # extrapolation for out-of-bounds inputs
interp_method = 'scipy_cubic'  # interpolation method
training_data_gradients = True  # compute gradients wrt output training data
vec_size = 1  # number of points to evaluate at once
nb_samp = (50, 50)  # number of samples on which the actual solution is computed as (twr, Isp)
rec_file = 'llo2apo_mm_log.pkl'  # name of the file on which the solution is serialized
rec_file_obj = 'llo2apo_mm_log_all.pkl'  # name of the file on which the object is serialized

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

# mm.plot()
save(mm, rec_file_obj)
