"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.primary import Moon
from rpfm.surrogate.meta_models_llo2heo import TwoDimLLO2ApoMetaModel

# MetaModel settings
rec_file = 'mm_test.pkl'  # name of the file on which the solution is serialized
interp_method = 'cubic'  # interpolation method
nb_samp = (5, 5)  # number of samples on which the actual solution is computed

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

mm = TwoDimLLO2ApoMetaModel(interp_method=interp_method)
mm.sampling(moon, twr, isp, llo_alt, None, method, segments, order, solver, nb_samp, snopt_opts=snopt_opts,
            rec_file=rec_file, t=heo_period, rp=heo_rp)
