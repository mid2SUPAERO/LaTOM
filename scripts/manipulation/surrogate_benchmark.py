"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.surrogate.om_metamodels import MetaModel
from latom.surrogate.smt_surrogates import SurrogateModel
from latom.utils.spacecraft import Spacecraft
from latom.utils.primary import Moon
from latom.analyzer.analyzer_2d import TwoDimAscConstAnalyzer
from latom.utils.pickle_utils import load, save
from latom.data.data import dirname


def generate_benchmark_points(lims, nb, mm_set, lhs_set, full_set=None, atol=1e-8, rtol=1e-5):
    points = []
    while len(points) < nb:
        p = np.random.uniform(lims[0], lims[-1])
        p_mm = np.isclose(p, mm_set, atol=atol, rtol=rtol).any()
        p_lhs = np.isclose(p, lhs_set, atol=atol, rtol=rtol).any()
        if full_set is not None:
            p_full = np.isclose(p, full_set, atol=atol, rtol=rtol).any()
        else:
            p_full = False
        if (not p_mm) and (not p_lhs) and (not p_full):
            points.append(p)
    return np.asarray(points)


def solve_nlp(body, sc):

    tr = TwoDimAscConstAnalyzer(body, sc, 100e3, np.pi/2, 500, None, 'gauss-lobatto', 60, 3, 'SNOPT', u_bound='lower',
                                check_partials=False, snopt_opts=snopt_opts)
    print(f"\nIsp: {sc.Isp:.6f} s\ttwr: {sc.twr:.6f}")
    f = tr.run_driver()
    tr.nlp.exp_sim()
    tr.get_solutions(explicit=True, scaled=False)
    print('Failed: ' + str(f))
    print(tr)
    return 1. - tr.states[-1, -1]


def compute_solutions(body, isp_list, twr_list):

    m_prop_list = []
    for i in range(np.size(isp_list)):
        sc = Spacecraft(isp_list[i], twr_list[i], g=body.g)
        m_prop_i = solve_nlp(body, sc)
        m_prop_list.append(m_prop_i)
    return np.asarray(m_prop_list)


# actual solution settings
moon = Moon()
snopt_opts = {'Major feasibility tolerance': 1e-12, 'Major optimality tolerance': 1e-12,
              'Minor feasibility tolerance': 1e-12}

# file names
fid_txt = 'surrogates_benchmark.txt'  # text file for summary
fid_real = dirname + '/asc_const_benchmark20.pkl'  # actual NLP solutions

# common metamodels settings
solve = False  # recompute actual solutions
store = False  # store computed actual solutions
multiple = True  # compare multiple surrogate modeling methods and interpolation schemes
compute_err = True  # compute the error
case = 'asc_const'  # study case between asc_const, asc_var, asc_vtoff, desc_const, desc_var, desc_vland
kind = None  # quantity to be displayed between prop (propellant fraction) or final (final/initial mass ratio)
nb_points = 20  # number of benchmark points

# MetaModel settings
distributed = False  # variables distributed across multiple processes
extrapolate = False  # extrapolation for out-of-bounds inputs
training_data_gradients = True  # compute gradients wrt output training data
if multiple:
    interp_method = ['slinear', 'lagrange2', 'lagrange3', 'cubic', 'akima', 'scipy_cubic', 'scipy_slinear',
                     'scipy_quintic']
else:
    interp_method = ['slinear']

# SurrogateModel settings
if multiple:
    train_method_lhs = ['KRG', 'LS', 'QP']
    train_method_full = ['IDW', 'LS', 'QP', 'RBF', 'RMTB', 'RMTC']
else:
    train_method_lhs = ['KRG']  # latin hypercube between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC
    train_method_full = ['LS']  # full-factorial between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC

# names of the files containing the serialized solutions
fid_mm = case + '_mm.pkl'
fid_lhs = case + '_lhs.pkl'
if case not in ['asc_vtoff', 'desc_vland']:
    fid_full = case + '_full.pkl'
else:
    fid_full = None

# write summary of interpolation error
if compute_err:
    fid = open(fid_txt, 'w+')
else:
    fid = None

for im_mm in interp_method:
    for tm_lhs in train_method_lhs:
        for tm_full in train_method_full:

            # MetaModel instance
            mm = MetaModel(distributed=distributed, extrapolate=extrapolate, method=im_mm,
                           training_data_gradients=training_data_gradients, vec_size=nb_points, rec_file=fid_mm)

            # SurrogateModel instance(s)
            sm_lhs = SurrogateModel(tm_lhs, rec_file=fid_lhs)
            if fid_full is not None:
                sm_full = SurrogateModel(tm_full, rec_file=fid_full)
            else:
                sm_full = None

            # benchmark points
            if compute_err and (not solve):
                d = load(fid_real)
                isp = d['Isp']
                twr = d['twr']
            else:
                if sm_full is not None:
                    isp = generate_benchmark_points(mm.limits[0], nb_points, mm.Isp, sm_lhs.x_samp[:, 0],
                                                    sm_full.x_samp[:, 0], atol=1e-6)
                    twr = generate_benchmark_points(mm.limits[1], nb_points, mm.twr, sm_lhs.x_samp[:, 1],
                                                    sm_full.x_samp[:, 1], atol=1e-6)
                else:
                    isp = generate_benchmark_points(mm.limits[0], nb_points, mm.Isp, sm_lhs.x_samp[:, 0], atol=1e-6)
                    twr = generate_benchmark_points(mm.limits[1], nb_points, mm.twr, sm_lhs.x_samp[:, 1], atol=1e-6)
                d = {}

            # MetaModel evaluation
            mm.p['Isp'] = isp
            mm.p['twr'] = twr
            mm.p.run_model()
            m_prop_mm = mm.p['mm.m_prop']

            # SurrogateModel evaluation(s)
            m_prop_lhs = sm_lhs.evaluate(isp, twr)
            if sm_full is not None:
                m_prop_full = sm_full.evaluate(isp, twr)
            else:
                m_prop_full = None

            # actual solutions
            if solve:
                m_prop = compute_solutions(moon, isp, twr)
            elif compute_err:
                m_prop = d['m_prop']
            else:
                m_prop = None

            # errors
            if compute_err:
                err_mm = np.fabs(m_prop.flatten() - m_prop_mm.flatten())
                err_lhs = np.fabs(m_prop.flatten() - m_prop_lhs.flatten())
                if m_prop_full is not None:
                    err_full = np.fabs(m_prop.flatten() - m_prop_full.flatten())
                else:
                    err_full = None
                fid.write(f"\n{'mm':4s}\t{im_mm:20s}\t{np.min(err_mm):.16e}\t{np.max(err_mm):.16e}")
                fid.write(f"\n{'lhs':4s}\t{tm_lhs:20s}\t{np.min(err_lhs):.16e}\t{np.max(err_lhs):.16e}")
                fid.write(f"\n{'full':4s}\t{tm_full:20s}\t{np.min(err_full):.16e}\t{np.max(err_full):.16e}")
            if store:
                d = {'Isp': isp, 'twr': twr, 'm_prop': m_prop}
                save(d, fid_real)

if fid is not None:
    fid.close()
