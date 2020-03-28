.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_computation_smt_surf2llo.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_computation_smt_surf2llo.py:


SMT Surrogate Model for Moon to LLO and LLO to Moon transfers
=============================================================

This example computes the sampling grid and training points to assemble and train an SMT Surrogate Model for a Moon to
LLO or LLO to Moon transfer with constant or variable thrust and optional minimum safe altitude.

@authors: Alberto FOSSA' Giuliana Elena MICELI


.. code-block:: default


    import numpy as np

    from latom.surrogate.smt_surrogates import TwoDimAscConstSurrogate, TwoDimAscVarSurrogate, TwoDimAscVToffSurrogate, \
        TwoDimDescVertSurrogate, TwoDimDescConstSurrogate, TwoDimDescVarSurrogate, TwoDimDescVLandSurrogate
    from latom.utils.primary import Moon

    rec_file = 'example.pkl'  # name of the file in latom.data.smt in which the solution is serialized

    # transfer type among the followings:
    # ac: ascent constant, av: ascent variable, as: ascent vertical takeoff
    # dc: descent constant, dv: descent variable, ds: descent vertical landing, d2p: two-phases descent vertical landing
    kind = 'ac'

    # SurrogateModel settings
    samp_method = 'lhs'  # sampling scheme, 'lhs' for Latin Hypercube or 'full' for Full-Factorial
    nb_samp = 10  # total number of samples, must be a perfect square if 'full' is chosen as sampling scheme
    criterion = 'm'  # sampling criterion (Latin Hypercube only)
    train_method = 'KRG'  # surrogate modeling method among IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC
    nb_eval = 100  # number of points to plot the response surface, must be a perfect square (Latin Hypercube only)

    moon = Moon()  # central attracting body

    # trajectory
    alt = 100e3  # final orbit altitude [m]
    theta = np.pi / 2  # guessed spawn angle [rad]
    tof = 1000  # guessed time of flight [s]
    tof_desc_2p = (1000, 100)  # guessed time of flight (descent with 2 phases) [s]
    t_bounds = None  # time of flight bounds [-]
    alt_p = 15e3  # perigee altitude (descent 2 phases and constant only) [m]
    alt_switch = 3e3  # switch altitude (descent 2 phases only) [m]
    alt_safe = 5e3  # minimum safe altitude (ascent and descent safe only) [m]
    slope = 10.  # slope of the constraint on minimum safe altitude (ascent and descent safe only) [-]
    isp_lim = (250., 500.)  # specific impulse lower and upper limits [s]
    twr_lim = (1.1, 4.)  # initial thrust/weight ratio lower and upper limits [-]

    # NLP
    method = 'gauss-lobatto'
    segments_asc = 20
    segments_desc_2p = (10, 10)
    order = 3
    solver = 'SNOPT'
    snopt_opts = {'Major feasibility tolerance': 1e-8, 'Major optimality tolerance': 1e-8,
                  'Minor feasibility tolerance': 1e-8}

    if kind == 'ac':
        sm = TwoDimAscConstSurrogate(train_method)
        sm.sampling(moon, isp_lim, twr_lim, alt, theta, tof, t_bounds, method, segments_asc, order, solver, nb_samp,
                    samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts)
    elif kind == 'av':
        sm = TwoDimAscVarSurrogate(train_method)
        sm.sampling(moon, isp_lim, twr_lim, alt, t_bounds, method, segments_asc, order, solver, nb_samp,
                    samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts)
    elif kind == 'as':
        sm = TwoDimAscVToffSurrogate(train_method)
        sm.sampling(moon, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, segments_asc, order, solver, nb_samp,
                    samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts)
    elif kind == 'd2p':
        sm = TwoDimDescVertSurrogate(train_method)
        sm.sampling(moon, isp_lim, twr_lim, alt, alt_p, alt_switch, theta, tof_desc_2p, t_bounds, method, segments_desc_2p,
                    order, solver, nb_samp, samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts)
    elif kind == 'dc':
        sm = TwoDimDescConstSurrogate(train_method)
        sm.sampling(moon, isp_lim, twr_lim, alt, alt_p, theta, tof, t_bounds, method, segments_asc, order, solver, nb_samp,
                    samp_method=samp_method, criterion=criterion, snopt_opts=snopt_opts)
    elif kind == 'dv':
        sm = TwoDimDescVarSurrogate(train_method)
        sm.sampling(moon, isp_lim, twr_lim, alt, t_bounds, method, segments_asc, order, solver, nb_samp,
                    samp_method=samp_method, criterion=criterion)
    elif kind == 'ds':
        sm = TwoDimDescVLandSurrogate(train_method)
        sm.sampling(moon, isp_lim, twr_lim, alt, alt_safe, -slope, t_bounds, method, segments_asc, order, solver, nb_samp,
                    samp_method=samp_method, criterion=criterion)
    else:
        raise ValueError('kind must be ac, av, as or d2p, dc, dv, ds')

    sm.save(rec_file)

    if samp_method == 'lhs':
        sm.train(train_method)
        sm.plot(nb_eval=nb_eval)
    else:
        sm.plot()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_examples_computation_smt_surf2llo.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: smt_surf2llo.py <smt_surf2llo.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: smt_surf2llo.ipynb <smt_surf2llo.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
