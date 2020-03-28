"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from smt.sampling_methods import LHS, FullFactorial
from smt.surrogate_models import IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC

from latom.utils.spacecraft import Spacecraft, ImpulsiveBurn
from latom.utils.keplerian_orbit import TwoDimOrb
from latom.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP, TwoDimDescTwoPhasesNLP,\
    TwoDimDescConstNLP, TwoDimDescVarNLP, TwoDimDescVLandNLP
from latom.guess.guess_2d import HohmannTransfer
from latom.plots.response_surfaces import RespSurf
from latom.data.smt.data_smt import dirname_smt
from latom.utils.pickle_utils import save, load


class SurrogateModel:
    """`SurrogateModel` class sets up a surrogate model object defined in the Surrogate Modelling Toolbox (SMT) [2]_
    package to compute and exploit data for different transfer trajectories.

    The model inputs are the spacecraft specific impulse `Isp` and initial thrust/weight ratio `twr` while the model
    output is the propellant fraction `m_prop`.

    Parameters
    ----------
    train_method : str
        Training method among ``IDW``, ``KPLS``, ``KPLSK``, ``KRG``, ``LS``, ``QP``, ``RBF``, ``RMTB``, ``RMTC``
    rec_file : str or ``None``, optional
        Name of the file in `latom.data.smt` where the surrogate model is stored or ``None`` if a new model has to
        be built. Default is ``None``

    Attributes
    ----------
    limits : ndarray
        Sampling grid limits in terms of minimum and maximum `Isp` and `twr`
    x_samp : ndarray
        Sampling points as `Isp, twr` tuples
    m_prop : ndarray
        Propellant fraction on training points [-]
    failures : ndarray
        Matrix of boolean to verify each NLP solution has converged
    d : dict
        Dictionary that contain all the information to reconstruct a meta model
    trained : IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB or RMTC
        Surrogate model object defined by SMT

    References
    ----------
    .. [2] M. A. Bouhlel and J. T. Hwang and N. Bartoli and R. Lafage and J. Morlier and J. R. R. A. Martins.
        A Python surrogate modeling framework with derivatives. Advances in Engineering Software, 2019.

    """

    def __init__(self, train_method, rec_file=None):
        """Initializes `SurrogateModel` class. """

        self.limits = self.x_samp = self.m_prop = self.failures = self.d = None
        self.trained = None

        if rec_file is not None:
            self.load(rec_file)
            self.train(train_method)

    @staticmethod
    def abs_path(rec_file):
        """Returns the absolute path of the file where the surrogate model is stored.

        Parameters
        ----------
        rec_file : str
            Name of the file in `latom.data.smt` where the surrogate model is stored

        Returns
        -------
        fid : str
            Full path where the surrogate model is stored

        """

        return '/'.join([dirname_smt, rec_file])

    def load(self, rec_file):
        """Loads stored data to instantiate a surrogate model.

        Parameters
        ----------
        rec_file : str
            Name of the file in `latom.data.smt` where the surrogate model is stored

        """

        self.d = load(self.abs_path(rec_file))
        self.limits = self.d['limits']
        self.x_samp = self.d['x_samp']
        self.m_prop = self.d['m_prop']
        self.failures = self.d['failures']

    def save(self, rec_file):
        """Saves data corresponding to a surrogate model instance.

        Parameters
        ----------
        rec_file : str
            Name of the file in `latom.data.smt` where the surrogate model is stored

        """

        d = {'limits': self.limits, 'x_samp': self.x_samp, 'm_prop': self.m_prop, 'failures': self.failures}
        save(d, self.abs_path(rec_file))

    def compute_grid(self, isp_lim, twr_lim, nb_samp, samp_method='full', criterion='m'):
        """Compute the sampling grid fro given `Isp` and `twr` limits and sampling scheme.

        Parameters
        ----------
        isp_lim : iterable
            Specific impulse lower and upper bounds [s]
        twr_lim : iterable
            Thrust/weight ratio lower and upper bounds [-]
        nb_samp : int
            Total number of samples. Must be a perfect square if ``full`` is chosen as `samp_method`
        samp_method : str, optional
            Sampling scheme, ``lhs`` for Latin Hypercube Sampling or ``full`` for Full-Factorial Sampling.
            Default is ``full``
        criterion : str, optional
            Criterion used to construct the LHS design among ``center``, ``maximin``, ``centermaximin``,
            ``correlation``, ``c``, ``m``, ``cm``, ``corr``, ``ese``. ``c``, ``m``, ``cm`` and ``corr`` are
            abbreviations of ``center``, ``maximin``, ``centermaximin`` and ``correlation``, ``respectively``
            Default is ``m``

        """

        self.limits = np.vstack((np.asarray(isp_lim), np.asarray(twr_lim)))

        if samp_method == 'lhs':
            samp = LHS(xlimits=self.limits, criterion=criterion)
        elif samp_method == 'full':
            samp = FullFactorial(xlimits=self.limits)
        else:
            raise ValueError('samp_method must be either lhs or full')

        self.x_samp = samp(nb_samp)
        self.m_prop = np.zeros((nb_samp, 1))
        self.failures = np.zeros((nb_samp, 1))

    @staticmethod
    def solve(nlp, i):
        """Solve the i-th NLP problem.

        Parameters
        ----------
        nlp : NLP
            NLP object
        i : int
            Current iteration

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        print(f"\nIteration {i}\nIsp: {nlp.sc.Isp:.6f} s\ttwr: {nlp.sc.twr:.6f}")
        f = nlp.p.run_driver()
        print("\nFailure: {0}".format(f))

        if isinstance(nlp.phase_name, str):
            phase_name = nlp.phase_name
        else:
            phase_name = nlp.phase_name[-1]

        m_prop = 1.0 - nlp.p.get_val(phase_name + '.timeseries.states:m')[-1, -1]
        nlp.cleanup()

        return m_prop, f

    def train(self, train_method, **kwargs):
        """Trains the surrogate model with given training data.

        Parameters
        ----------
        train_method : str
            Training method among ``IDW``, ``KPLS``, ``KPLSK``, ``KRG``, ``LS``, ``QP``, ``RBF``, ``RMTB``, ``RMTC``
        kwargs : dict
            Additional keyword arguments supported by SMT objects

        """

        if train_method == 'IDW':
            self.trained = IDW(**kwargs)
        elif train_method == 'KPLS':
            self.trained = KPLS(**kwargs)
        elif train_method == 'KPLSK':
            self.trained = KPLSK(**kwargs)
        elif train_method == 'KRG':
            self.trained = KRG(**kwargs)
        elif train_method == 'LS':
            self.trained = LS(**kwargs)
        elif train_method == 'QP':
            self.trained = QP(**kwargs)
        elif train_method == 'RBF':
            self.trained = RBF(**kwargs)
        elif train_method == 'RMTB':
            self.trained = RMTB(xlimits=self.limits, **kwargs)
        elif train_method == 'RMTC':
            self.trained = RMTC(xlimits=self.limits, **kwargs)
        else:
            raise ValueError('train_method must be one between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC')

        self.trained.set_training_values(self.x_samp, self.m_prop)
        self.trained.train()

    def evaluate(self, isp, twr):
        """Evaluate the surrogate model in the given set of points.

        Parameters
        ----------
        isp : float or iterable
            Specific impulse on evaluation points [s]
        twr : float or iterable
            Thrust/weight ratio on evaluation points [-]

        Returns
        -------
        m_eval : float or iterable
            Propellant fraction on evaluation points [-]

        """

        if isinstance(isp, float):
            isp = [isp]
        if isinstance(twr, float):
            twr = [twr]

        x_eval = np.hstack((np.reshape(isp, (len(isp), 1)), np.reshape(twr, (len(twr), 1))))
        m_eval = self.trained.predict_values(x_eval)

        return m_eval

    def compute_matrix(self, nb_eval=None):
        """Compute structured matrices for `Isp`, `twr` and `m_prop` to display the training data on a response surface.

        Parameters
        ----------
        nb_eval : int or ``None``
            Number of points included in the matrix if Latin Hypercube Sampling has been used or ``None``.
            Default is ``None``

        Returns
        -------
        isp : ndarray
            Matrix of specific impulses [s]
        twr : ndarray
            Matrix of thrust/weight ratios [-]
        m_mat : ndarray
            Matrix of propellant fractions [-]

        """

        if nb_eval is not None:  # LHS
            samp_eval = FullFactorial(xlimits=self.limits)
            x_eval = samp_eval(nb_eval)
            m_prop_eval = self.trained.predict_values(x_eval)

        else:  # Full-Factorial
            nb_eval = np.size(self.m_prop)
            x_eval = deepcopy(self.x_samp)
            m_prop_eval = deepcopy(self.m_prop)

        isp = np.unique(x_eval[:, 0])
        twr = np.unique(x_eval[:, 1])
        n = int(np.sqrt(nb_eval))
        m_mat = np.reshape(m_prop_eval, (n, n))

        return isp, twr, m_mat

    def plot(self, nb_eval=None, nb_lines=50, kind='prop'):
        """Plot the response surface corresponding to the loaded surrogate model.

        Parameters
        ----------
        nb_eval : int or ``None``
            Number of points included in the matrix if Latin Hypercube Sampling has been used or ``None``.
            Default is ``None``
        nb_lines : int, optional
            Number of contour lines. Default is ``50``
        kind : str
            ``prop`` to display the propellant fraction `m_prop`, ``final`` to display the final spacecraft mass
            `1 - m_prop`

        """

        isp, twr, m_mat = self.compute_matrix(nb_eval=nb_eval)

        if kind == 'prop':
            surf_plot = RespSurf(isp, twr, m_mat, 'Propellant fraction', nb_lines=nb_lines)
        elif kind == 'final':
            surf_plot = RespSurf(isp, twr, (1 - m_mat), 'Final/initial mass ratio', nb_lines=nb_lines)
        else:
            raise ValueError('kind must be either prop or final')

        surf_plot.plot()

        plt.show()


class TwoDimAscConstSurrogate(SurrogateModel):
    """`TwoDimAscConstSurrogate` implements `SurrogateModel` for a two-dimensional ascent trajectory at constant thrust.

    """

    def sampling(self, body, isp_lim, twr_lim, alt, theta, tof, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='lhs', criterion='m', snopt_opts=None, u_bound='lower'):
        """Compute a new set of training data starting from a given sampling grid.

        Parameters
        ----------
        body : Primary
            Central attracting body
        isp_lim : iterable
            Specific impulse lower and upper limits [s]
        twr_lim : iterable
            Thrust/weight ratio lower and upper limits [-]
        alt : float
            Orbit altitude [m]
        theta : float
            Guessed spawn angle [rad]
        tof : float
            Guessed time of flight [s]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int or iterable
            Number of segments in which each phase is discretized
        order : int or iterable
            Transcription order within each phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        nb_samp : iterable
            Total number of sampling points
        samp_method : str, optional
            Sampling scheme, ``lhs`` for Latin Hypercube Sampling or ``full`` for Full-Factorial Sampling.
            Default is ``lhs``
        criterion : str, optional
            Criterion used to construct the LHS design among ``center``, ``maximin``, ``centermaximin``,
            ``correlation``, ``c``, ``m``, ``cm``, ``corr``, ``ese``. ``c``, ``m``, ``cm`` and ``corr`` are
            abbreviations of ``center``, ``maximin``, ``centermaximin`` and ``correlation``, ``respectively``.
            Default is ``m``
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``lower``

        """

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimAscConstNLP(body, sc, alt, theta, (-np.pi/2, np.pi/2), tof, t_bounds, method, nb_seg, order,
                                    solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimAscVarSurrogate(SurrogateModel):
    """`TwoDimAscVarSurrogate` implements `SurrogateModel` for a two-dimensional ascent trajectory with variable thrust.

    """

    def sampling(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='lhs', criterion='m', snopt_opts=None, u_bound='lower'):
        """Compute a new set of training data starting from a given sampling grid.

        Parameters
        ----------
        body : Primary
            Central attracting body
        isp_lim : iterable
            Specific impulse lower and upper limits [s]
        twr_lim : iterable
            Thrust/weight ratio lower and upper limits [-]
        alt : float
            Orbit altitude [m]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int or iterable
            Number of segments in which each phase is discretized
        order : int or iterable
            Transcription order within each phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        nb_samp : iterable
            Total number of sampling points
        samp_method : str, optional
            Sampling scheme, ``lhs`` for Latin Hypercube Sampling or ``full`` for Full-Factorial Sampling.
            Default is ``lhs``
        criterion : str, optional
            Criterion used to construct the LHS design among ``center``, ``maximin``, ``centermaximin``,
            ``correlation``, ``c``, ``m``, ``cm``, ``corr``, ``ese``. ``c``, ``m``, ``cm`` and ``corr`` are
            abbreviations of ``center``, ``maximin``, ``centermaximin`` and ``correlation``, ``respectively``.
            Default is ``m``
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``lower``

        """

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimAscVarNLP(body, sc, alt, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg, order, solver,
                                  'powered', snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimAscVToffSurrogate(SurrogateModel):
    """`TwoDimAscVToffSurrogate` implements `SurrogateModel` for a two-dimensional ascent trajectory with variable
    thrust and minimum safe altitude.

    """

    def sampling(self, body, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver,
                 nb_samp, samp_method='lhs', criterion='m', snopt_opts=None, u_bound='lower'):
        """Compute a new set of training data starting from a given sampling grid.

        Parameters
        ----------
        body : Primary
            Central attracting body
        isp_lim : iterable
            Specific impulse lower and upper limits [s]
        twr_lim : iterable
            Thrust/weight ratio lower and upper limits [-]
        alt : float
            Orbit altitude [m]
        alt_safe : float
            Asymptotic minimum safe altitude [m]
        slope : float
            Minimum safe altitude slope close to the launch site [-]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int or iterable
            Number of segments in which each phase is discretized
        order : int or iterable
            Transcription order within each phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        nb_samp : iterable
            Total number of sampling points
        samp_method : str, optional
            Sampling scheme, ``lhs`` for Latin Hypercube Sampling or ``full`` for Full-Factorial Sampling.
            Default is ``lhs``
        criterion : str, optional
            Criterion used to construct the LHS design among ``center``, ``maximin``, ``centermaximin``,
            ``correlation``, ``c``, ``m``, ``cm``, ``corr``, ``ese``. ``c``, ``m``, ``cm`` and ``corr`` are
            abbreviations of ``center``, ``maximin``, ``centermaximin`` and ``correlation``, ``respectively``.
            Default is ``m``
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``lower``

        """

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimAscVToffNLP(body, sc, alt, alt_safe, slope, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg,
                                    order, solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimDescConstSurrogate(SurrogateModel):
    """`TwoDimDescConstSurrogate` implements `SurrogateModel` for a two-dimensional descent trajectory at constant
    thrust.

    """

    def sampling(self, body, isp_lim, twr_lim, alt, alt_p, theta, tof, t_bounds, method, nb_seg, order, solver, nb_samp,
                 samp_method='lhs', criterion='m', snopt_opts=None, u_bound='upper'):
        """Compute a new set of training data starting from a given sampling grid.

        Parameters
        ----------
        body : Primary
            Central attracting body
        isp_lim : iterable
            Specific impulse lower and upper limits [s]
        twr_lim : iterable
            Thrust/weight ratio lower and upper limits [-]
        alt : float
            Orbit altitude [m]
        alt_p : float
            Periapsis altitude where the final powered descent is initiated [m]
        theta : float
            Guessed spawn angle [rad]
        tof : float
            Guessed time of flight [s]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int or iterable
            Number of segments in which each phase is discretized
        order : int or iterable
            Transcription order within each phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        nb_samp : iterable
            Total number of sampling points
        samp_method : str, optional
            Sampling scheme, ``lhs`` for Latin Hypercube Sampling or ``full`` for Full-Factorial Sampling.
            Default is ``lhs``
        criterion : str, optional
            Criterion used to construct the LHS design among ``center``, ``maximin``, ``centermaximin``,
            ``correlation``, ``c``, ``m``, ``cm``, ``corr``, ``ese``. ``c``, ``m``, ``cm`` and ``corr`` are
            abbreviations of ``center``, ``maximin``, ``centermaximin`` and ``correlation``, ``respectively``.
            Default is ``m``
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``upper``

        """

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        ht = HohmannTransfer(body.GM, TwoDimOrb(body.GM, a=(body.R + alt), e=0.0),
                             TwoDimOrb(body.GM, a=(body.R + alt_p), e=0.0))

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            deorbit_burn = ImpulsiveBurn(sc, ht.dva)
            nlp = TwoDimDescConstNLP(body, deorbit_burn.sc, alt_p, ht.transfer.vp, theta, (0, 3/2*np.pi),  tof,
                                     t_bounds, method, nb_seg, order, solver, 'powered', snopt_opts=snopt_opts,
                                     u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimDescVarSurrogate(SurrogateModel):
    """`TwoDimDescVarSurrogate` implements `SurrogateModel` for a two-dimensional descent trajectory with variable
    thrust.

    """

    def sampling(self, body, isp_lim, twr_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp, samp_method='lhs',
                 criterion='m', snopt_opts=None, u_bound='upper'):
        """Compute a new set of training data starting from a given sampling grid.

        Parameters
        ----------
        body : Primary
            Central attracting body
        isp_lim : iterable
            Specific impulse lower and upper limits [s]
        twr_lim : iterable
            Thrust/weight ratio lower and upper limits [-]
        alt : float
            Orbit altitude [m]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int or iterable
            Number of segments in which each phase is discretized
        order : int or iterable
            Transcription order within each phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        nb_samp : iterable
            Total number of sampling points
        samp_method : str, optional
            Sampling scheme, ``lhs`` for Latin Hypercube Sampling or ``full`` for Full-Factorial Sampling.
            Default is ``lhs``
        criterion : str, optional
            Criterion used to construct the LHS design among ``center``, ``maximin``, ``centermaximin``,
            ``correlation``, ``c``, ``m``, ``cm``, ``corr``, ``ese``. ``c``, ``m``, ``cm`` and ``corr`` are
            abbreviations of ``center``, ``maximin``, ``centermaximin`` and ``correlation``, ``respectively``.
            Default is ``m``
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``upper``

        """

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimDescVarNLP(body, sc, alt, (0.0, 3/2*np.pi), t_bounds, method, nb_seg, order, solver, 'powered',
                                   snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimDescVLandSurrogate(SurrogateModel):
    """`TwoDimDescVLandSurrogate` implements `SurrogateModel` for a two-dimensional descent trajectory with variable
    thrust and constrained minimum altitude.

    """

    def sampling(self, body, isp_lim, twr_lim, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver,
                 nb_samp, samp_method='lhs', criterion='m', snopt_opts=None, u_bound='upper'):
        """Compute a new set of training data starting from a given sampling grid.

        Parameters
        ----------
        body : Primary
            Central attracting body
        isp_lim : iterable
            Specific impulse lower and upper limits [s]
        twr_lim : iterable
            Thrust/weight ratio lower and upper limits [-]
        alt : float
            Orbit altitude [m]
        alt_safe : float
            Asymptotic minimum safe altitude [m]
        slope : float
            Minimum safe altitude slope close to the launch site [-]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int or iterable
            Number of segments in which each phase is discretized
        order : int or iterable
            Transcription order within each phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        nb_samp : iterable
            Total number of sampling points
        samp_method : str, optional
            Sampling scheme, ``lhs`` for Latin Hypercube Sampling or ``full`` for Full-Factorial Sampling.
            Default is ``lhs``
        criterion : str, optional
            Criterion used to construct the LHS design among ``center``, ``maximin``, ``centermaximin``,
            ``correlation``, ``c``, ``m``, ``cm``, ``corr``, ``ese``. ``c``, ``m``, ``cm`` and ``corr`` are
            abbreviations of ``center``, ``maximin``, ``centermaximin`` and ``correlation``, ``respectively``.
            Default is ``m``
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``upper``

        """

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            nlp = TwoDimDescVLandNLP(body, sc, alt, alt_safe, slope, (0.0, 3/2*np.pi), t_bounds, method, nb_seg, order,
                                     solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)


class TwoDimDescVertSurrogate(SurrogateModel):
    """`TwoDimDescVertSurrogate` implements `SurrogateModel` for a two-dimensional descent trajectory at constant
    thrust with final vertical descent phase.

    """

    def sampling(self, body, isp_lim, twr_lim, alt, alt_p, alt_switch, theta, tof, t_bounds, method, nb_seg, order,
                 solver, nb_samp, samp_method='lhs', criterion='m', snopt_opts=None):
        """Compute a new set of training data starting from a given sampling grid.

        Parameters
        ----------
        body : Primary
            Central attracting body
        isp_lim : iterable
            Specific impulse lower and upper limits [s]
        twr_lim : iterable
            Thrust/weight ratio lower and upper limits [-]
        alt : float
            Orbit altitude [m]
        alt_p : float
            Periapsis altitude where the final powered descent is initiated [m]
        alt_switch : float
            Altitude at which the final vertical descent is triggered [m]
        theta : float
            Guessed spawn angle [rad]
        tof : iterable
            Guessed time of flight for the two phases [s]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int or iterable
            Number of segments in which each phase is discretized
        order : int or iterable
            Transcription order within each phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        nb_samp : iterable
            Total number of sampling points
        samp_method : str, optional
            Sampling scheme, ``lhs`` for Latin Hypercube Sampling or ``full`` for Full-Factorial Sampling.
            Default is ``lhs``
        criterion : str, optional
            Criterion used to construct the LHS design among ``center``, ``maximin``, ``centermaximin``,
            ``correlation``, ``c``, ``m``, ``cm``, ``corr``, ``ese``. ``c``, ``m``, ``cm`` and ``corr`` are
            abbreviations of ``center``, ``maximin``, ``centermaximin`` and ``correlation``, ``respectively``.
            Default is ``m``
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None

        """

        self.compute_grid(isp_lim, twr_lim, nb_samp, samp_method=samp_method, criterion=criterion)

        ht = HohmannTransfer(body.GM, TwoDimOrb(body.GM, a=(body.R + alt), e=0.0),
                             TwoDimOrb(body.GM, a=(body.R + alt_p), e=0.0))

        for i in range(nb_samp):

            sc = Spacecraft(self.x_samp[i, 0], self.x_samp[i, 1], g=body.g)
            deorbit_burn = ImpulsiveBurn(sc, ht.dva)
            nlp = TwoDimDescTwoPhasesNLP(body, deorbit_burn.sc, alt, alt_switch, ht.transfer.vp, theta, (0.0, np.pi),
                                         tof, t_bounds, method, nb_seg, order, solver, ('free', 'vertical'),
                                         snopt_opts=snopt_opts)

            self.m_prop[i, 0], self.failures[i, 0] = self.solve(nlp, i)
