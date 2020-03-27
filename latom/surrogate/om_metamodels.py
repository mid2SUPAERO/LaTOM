"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, MetaModelStructuredComp
from latom.utils.pickle_utils import load, save
from latom.utils.spacecraft import Spacecraft, ImpulsiveBurn
from latom.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP, TwoDimDescConstNLP, \
    TwoDimDescVarNLP, TwoDimDescVLandNLP
from latom.guess.guess_2d import HohmannTransfer
from latom.utils.keplerian_orbit import TwoDimOrb
from latom.data.metamodels.data_mm import dirname_metamodels
from latom.plots.response_surfaces import RespSurf
from latom.analyzer.analyzer_2d import TwoDimDescTwoPhasesAnalyzer


class MetaModel:
    """`MetaModel` class sets up an OpenMDAO `Problem` object and a `MetaModelStructuredComp` subsystem to compute
    and exploit a surrogate model for different transfer trajectories.

    The model inputs are the spacecraft specific impulse `Isp` and initial thrust/weight ratio `twr` while the model
    output is the propellant fraction `m_prop`.

    Parameters
    ----------
    distributed : bool, optional
        ``True`` if the component has variables that are distributed across multiple processes, ``False`` otherwise.
        Default is ``False``
    extrapolate : bool, optional
        Sets whether extrapolation should be performed when an input is out of bounds. Default is ``False``
    method : str, optional
        Spline interpolation method to use for all outputs among ``cubic``, ``slinear``, ``lagrange2``, ``lagrange3``,
        ``akima``, ``scipy_cubic``, ``scipy_slinear``, ``scipy_quintic``. Default is ``scipy_cubic``
    training_data_gradients : bool, optional
        Sets whether gradients with respect to output training data should be computed. Default is ``True``
    vec_size : int, optional
        Number of points to evaluate at once. Default is ``1``
    rec_file : str or ``None``, optional
        Name of the file in `latom.data.metamodels` where the meta model is stored or ``None`` if a new model has to
        be built. Default is ``None``

    Attributes
    ----------
    mm : MetaModelStructuredComp
        OpenMDAO `MetaModelStructuredComp` object
    p : Problem
        OpenMDAO `Problem` object
    twr : ndarray
        List of thrust/weight ratios in the sampling grid [-]
    Isp : ndarray
        List of specific impulses in the sampling grid [s]
    m_prop : ndarray
        Propellant fraction in all training points [-]
    failures : ndarray
        Matrix of boolean to verify all the NLP solutions have properly converged
    limits : ndarray
        Sampling grid limits in terms of minimum and maximum `Isp` and `twr`
    d : dict
        Dictionary that contain all the information to reconstruct a meta model

    """

    def __init__(self, distributed=False, extrapolate=False, method='scipy_cubic', training_data_gradients=True,
                 vec_size=1, rec_file=None):
        """Initializes `MetaModel` class. """

        self.mm = MetaModelStructuredComp(distributed=distributed, extrapolate=extrapolate, method=method,
                                          training_data_gradients=training_data_gradients, vec_size=vec_size)

        self.p = Problem()
        self.p.model.add_subsystem('mm', self.mm, promotes=['Isp', 'twr'])
        self.twr = self.Isp = self.m_prop = self.failures = self.limits = self.d = None

        if rec_file is not None:  # load a stored set of data and instantiate a meta model from it
            self.load(rec_file)
            self.setup()

    @staticmethod
    def abs_path(rec_file):
        """Returns the absolute path of the file where the meta model is stored.

        Parameters
        ----------
        rec_file : str
            Name of the file in `latom.data.metamodels` where the meta model is stored

        Returns
        -------
        fid : str
            Full path where the meta model is stored

        """

        fid = '/'.join([dirname_metamodels, rec_file])

        return fid

    def load(self, rec_file):
        """Loads stored data to instantiate a meta model.

        Parameters
        ----------
        rec_file : str
            Name of the file in `latom.data.metamodels` where the meta model is stored

        """

        self.d = load(self.abs_path(rec_file))
        self.twr = self.d['twr']
        self.Isp = self.d['Isp']
        self.m_prop = self.d['m_prop']
        self.failures = self.d['failures']

    def save(self, rec_file):
        """Saves data corresponding to a meta model instance.

        Parameters
        ----------
        rec_file : str
            Name of the file in `latom.data.metamodels` where the meta model is stored

        """

        d = {'Isp': self.Isp, 'twr': self.twr, 'm_prop': self.m_prop, 'failures': self.failures}
        save(d, self.abs_path(rec_file))

    def compute_grid(self, twr_lim, isp_lim, nb_samp):
        """Computes a regular sampling grid in the input space `twr, Isp`.

        Parameters
        ----------
        twr_lim : iterable
            Thrust/weight ratio lower and upper limits [-]
        isp_lim : iterable
            Specific impulse lower and upper limits [s]
        nb_samp : iterable
            Number of samples along the `twr` and `Isp` axes

        """

        self.twr = np.linspace(twr_lim[0], twr_lim[1], nb_samp[0])
        self.Isp = np.linspace(isp_lim[0], isp_lim[1], nb_samp[1])

        self.m_prop = np.zeros(nb_samp)
        self.failures = np.zeros(nb_samp)

    def setup(self):
        """Setup the OpenMDAO `Problem` object with known data for `twr`, `Isp` and `m_prop`. """

        self.limits = np.array([[self.Isp[0], self.Isp[-1]], [self.twr[0], self.twr[-1]]])
        self.mm.add_input('twr', training_data=self.twr)
        self.mm.add_input('Isp', training_data=self.Isp)
        self.mm.add_output('m_prop', training_data=self.m_prop)

        self.p.setup()
        self.p.final_setup()

    def sampling(self, body, twr_lim, isp_lim, alt, t_bounds, method, nb_seg, order, solver, nb_samp, snopt_opts=None,
                 u_bound=None, rec_file=None, **kwargs):
        """Compute a new set of training data starting from a given sampling grid.

        Parameters
        ----------
        body : Primary
            Central attracting body
        twr_lim : iterable
            Thrust/weight ratio lower and upper limits [-]
        isp_lim : iterable
            Specific impulse lower and upper limits [s]
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
            Number of samples along the `twr` and `Isp` axes
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``None``
        rec_file : str, optional
            Name of the file in `latom.data.metamodels` where the meta model will be stored. Default is ``None``
        kwargs : dict
            Additional input arguments

        """

        self.compute_grid(twr_lim, isp_lim, nb_samp)

        count = 1

        for i in range(nb_samp[0]):  # loop over thrust/weight ratios
            for j in range(nb_samp[1]):  # loop over specific impulses

                print(f"\nMajor Iteration {j}"
                      f"\nSpecific impulse: {self.Isp[j]:.6f} s"
                      f"\nThrust/weight ratio: {self.twr[i]:.6f}\n")

                sc = Spacecraft(self.Isp[j], self.twr[i], g=body.g)

                try:
                    m, f = self.solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=snopt_opts,
                                      u_bound=u_bound, **kwargs)
                except ValueError:
                    m = None
                    f = 1.

                self.m_prop[i, j] = m
                self.failures[i, j] = f

                count += 1

        self.setup()

        if rec_file is not None:
            self.save(rec_file)

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        """Solve the NLP for the i-th `twr` and k-th `Isp` values.

        Parameters
        ----------
        body : Primary
            Central attracting body
        sc : Spacecraft
            Spacecraft object characterized by the i-th `twr` and k-th `Isp` values
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
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``None``
        kwargs : dict
            Additional input arguments

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        return None, None

    def plot(self, nb_lines=50, kind='prop', log_scale=False):
        """Plot the response surface corresponding to the loaded meta model.

        Parameters
        ----------
        nb_lines : int, optional
            Number of contour lines. Default is ``50``
        kind : str
            ``prop`` to display the propellant fraction `m_prop`, ``final`` to display the final spacecraft mass
            `1 - m_prop`
        log_scale : bool, optional
            Set to ``True`` if the `twr` values are in logarithmic scale. Default is ``False``

        """

        if kind == 'prop':
            rs = RespSurf(self.Isp, self.twr, self.m_prop.T, 'Propellant fraction', nb_lines=nb_lines,
                          log_scale=log_scale)
        elif kind == 'final':
            rs = RespSurf(self.Isp, self.twr, (1 - self.m_prop.T), 'Final/initial mass ratio', nb_lines=nb_lines,
                          log_scale=log_scale)
        else:
            raise ValueError('kind must be either prop or final')

        rs.plot()
        plt.show()


class TwoDimAscConstMetaModel(MetaModel):
    """`TwoDimAscConstMetaModel` implements `MetaModel` for a two-dimensional ascent trajectory at constant thrust. """

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        """Solve the NLP for the i-th `twr` and k-th `Isp` values.

        Parameters
        ----------
        body : Primary
            Central attracting body
        sc : Spacecraft
            Spacecraft object characterized by the i-th `twr` and k-th `Isp` values
        alt : float
            Orbit altitude [m]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int
            Number of segments in which the phase is discretized
        order : int
            Transcription order within the phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``None``

        Other Parameters
        ----------------
        theta : float
            Guessed spawn angle [rad]
        tof : float
            Guessed time of flight [s]

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        nlp = TwoDimAscConstNLP(body, sc, alt, kwargs['theta'], (-np.pi / 2, np.pi / 2), kwargs['tof'], t_bounds,
                                method, nb_seg, order, solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimAscVarMetaModel(MetaModel):
    """`TwoDimAscVarMetaModel` implements `MetaModel` for a two-dimensional ascent trajectory with variable thrust. """

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        """Solve the NLP for the i-th `twr` and k-th `Isp` values.

        Parameters
        ----------
        body : Primary
            Central attracting body
        sc : Spacecraft
            Spacecraft object characterized by the i-th `twr` and k-th `Isp` values
        alt : float
            Orbit altitude [m]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int
            Number of segments in which the phase is discretized
        order : int
            Transcription order within the phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``None``
        kwargs : dict
            Additional input arguments

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        nlp = TwoDimAscVarNLP(body, sc, alt, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg, order, solver,
                              'powered', snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimAscVToffMetaModel(MetaModel):
    """`TwoDimAscVToffMetaModel` implements `MetaModel` for a two-dimensional ascent trajectory with variable thrust
    and constrained minimum safe altitude.

    """

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        """Solve the NLP for the i-th `twr` and k-th `Isp` values.

        Parameters
        ----------
        body : Primary
            Central attracting body
        sc : Spacecraft
            Spacecraft object characterized by the i-th `twr` and k-th `Isp` values
        alt : float
            Orbit altitude [m]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int
            Number of segments in which the phase is discretized
        order : int
            Transcription order within the phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``None``

        Other Parameters
        ----------------
        alt_safe : float
            Asymptotic minimum safe altitude [m]
        slope : float
            Minimum safe altitude slope close to the launch site [-]

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        nlp = TwoDimAscVToffNLP(body, sc, alt, kwargs['alt_safe'], kwargs['slope'], (-np.pi / 2, np.pi / 2), t_bounds,
                                method, nb_seg, order, solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimDescConstMetaModel(MetaModel):
    """`TwoDimDescConstMetaModel` implements `MetaModel` for a two-dimensional descent trajectory at constant thrust.

    """

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        """Solve the NLP for the i-th `twr` and k-th `Isp` values.

        Parameters
        ----------
        body : Primary
            Central attracting body
        sc : Spacecraft
            Spacecraft object characterized by the i-th `twr` and k-th `Isp` values
        alt : float
            Orbit altitude [m]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int
            Number of segments in which the phase is discretized
        order : int
            Transcription order within the phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``None``

        Other Parameters
        ----------------
        alt_p : float
            Periapsis altitude where the final powered descent is initiated [m]
        theta : float
            Guessed spawn angle [rad]
        tof : float
            Guessed time of flight [s]

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        if 'alt_p' in kwargs:
            alt_p = kwargs['alt_p']
        else:
            alt_p = alt

        dep = TwoDimOrb(body.GM, a=(body.R + alt), e=0.0)
        arr = TwoDimOrb(body.GM, a=(body.R + alt_p), e=0.0)

        ht = HohmannTransfer(body.GM, dep, arr)
        deorbit_burn = ImpulsiveBurn(sc, ht.dva)

        nlp = TwoDimDescConstNLP(body, deorbit_burn.sc, alt_p, ht.transfer.vp, kwargs['theta'], (0.0, 1.5 * np.pi),
                                 kwargs['tof'], t_bounds, method, nb_seg, order, solver, 'powered',
                                 snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimDescVarMetaModel(MetaModel):
    """`TwoDimDescVarMetaModel` implements `MetaModel` for a two-dimensional descent trajectory with variable thrust.

    """

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        """Solve the NLP for the i-th `twr` and k-th `Isp` values.

        Parameters
        ----------
        body : Primary
            Central attracting body
        sc : Spacecraft
            Spacecraft object characterized by the i-th `twr` and k-th `Isp` values
        alt : float
            Orbit altitude [m]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int
            Number of segments in which the phase is discretized
        order : int
            Transcription order within the phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``None``
        kwargs : dict
            Additional input arguments

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        nlp = TwoDimDescVarNLP(body, sc, alt, (0.0, 3 / 2 * np.pi), t_bounds, method, nb_seg, order, solver, 'powered',
                               snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimDescVLandMetaModel(MetaModel):
    """`TwoDimDescVLandMetaModel` implements `MetaModel` for a two-dimensional descent trajectory with variable thrust
    and constrained minimum safe altitude.

    """

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        """Solve the NLP for the i-th `twr` and k-th `Isp` values.

        Parameters
        ----------
        body : Primary
            Central attracting body
        sc : Spacecraft
            Spacecraft object characterized by the i-th `twr` and k-th `Isp` values
        alt : float
            Orbit altitude [m]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int
            Number of segments in which the phase is discretized
        order : int
            Transcription order within the phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``None``

        Other Parameters
        ----------------
        alt_safe : float
            Asymptotic minimum safe altitude [m]
        slope : float
            Minimum safe altitude slope close to the launch site [-]

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        nlp = TwoDimDescVLandNLP(body, sc, alt, kwargs['alt_safe'], kwargs['slope'], (0.0, 3 / 2 * np.pi), t_bounds,
                                 method, nb_seg, order, solver, 'powered', snopt_opts=snopt_opts, u_bound=u_bound)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name + '.timeseries.states:m')[-1, -1]

        return m_prop, f


class TwoDimDescTwoPhasesMetaModel(MetaModel):

    @staticmethod
    def solve(body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, u_bound=None, **kwargs):
        """Solve the NLP for the i-th `twr` and k-th `Isp` values.

        Parameters
        ----------
        body : Primary
            Central attracting body
        sc : Spacecraft
            Spacecraft object characterized by the i-th `twr` and k-th `Isp` values
        alt : float
            Orbit altitude [m]
        t_bounds : iterable
            Time of flight lower and upper bounds expressed as fraction of `tof`
        method : str
            Transcription method used to discretize the continuous time trajectory into a finite set of nodes,
            allowed ``gauss-lobatto``, ``radau-ps`` and ``runge-kutta``
        nb_seg : int
            Number of segments in which the phase is discretized
        order : int
            Transcription order within the phase, must be odd
        solver : str
            NLP solver, must be supported by OpenMDAO
        snopt_opts : dict or None, optional
            SNOPT optional settings expressed as key-value pairs. Default is None
        u_bound : str, optional
            Specify the bounds on the radial velocity along the transfer as ``lower``, ``upper`` or ``None``.
            Default is ``None``

        Other Parameters
        ----------------
        alt_p : float
            Periapsis altitude where the powered descent is initiated [m]
        alt_switch : float
            Altitude at which the final vertical descent is triggered [m]
        theta : float
            Guessed spawn angle [rad]
        tof : float
            Guessed time of flight [s]
        fix : str
            ``alt`` if the vertical phase is triggered at a fixed altitude, ``time`` for a fixed time-to-go

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        tr = TwoDimDescTwoPhasesAnalyzer(body, sc, alt, kwargs['alt_p'], kwargs['alt_switch'], kwargs['theta'],
                                         kwargs['tof'], t_bounds, method, nb_seg, order, solver, snopt_opts=snopt_opts,
                                         fix=kwargs['fix'])

        f = tr.run_driver()
        tr.get_solutions(explicit=False, scaled=False)
        tr.nlp.cleanup()

        m_prop = 1 - tr.states[-1][-1, -1] / tr.sc.m0

        return m_prop, f
