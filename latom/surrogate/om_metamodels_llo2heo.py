"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.nlp.nlp_heo_2d import TwoDim3PhasesLLO2HEONLP
from latom.analyzer.analyzer_heo_2d import TwoDimLLO2ApoAnalyzer, TwoDimLLO2ApoContinuationAnalyzer
from latom.surrogate.om_metamodels import MetaModel
from latom.utils.pickle_utils import save
from latom.utils.spacecraft import Spacecraft
from latom.plots.response_surfaces import RespSurf


class TwoDimLLO2ApoMetaModel(MetaModel):
    """`TwoDimLLO2ApoMetaModel` implements `MetaModel` for a two-dimensional escape burn from an LLO to an intermediate
    ballistic arc to reach the specified HEO apoapsis radius.

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
            rp : float
                HEO periapsis radius [m]
            t : float
                HEO period [s]

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        tr = TwoDimLLO2ApoAnalyzer(body, sc, alt, kwargs['rp'], kwargs['t'], t_bounds, method, nb_seg, order, solver,
                                   snopt_opts=snopt_opts)

        f = tr.run_driver()
        tr.get_solutions(explicit=False, scaled=False)
        tr.nlp.cleanup()

        m_prop = 1. - tr.insertion_burn.mf/tr.sc.m0

        return m_prop, f


class TwoDimLLO2ApoContinuationMetaModel(MetaModel):
    """`TwoDimLLO2ApoContinuationMetaModel` implements `MetaModel` for a two-dimensional escape burn from an LLO to an
    intermediate ballistic arc to reach the specified HEO apoapsis radius.

    For each `Isp` value in the sampling grid, the training data are computed using a continuation method for decreasing
    `twr` values.

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
    energy : ndarray
        Spacecraft specific energy at the end of the transfer on all training points [m^2/s^2]

    """

    def __init__(self, distributed=False, extrapolate=False, method='scipy_cubic', training_data_gradients=True,
                 vec_size=1, rec_file=None):

        self.energy = None

        MetaModel.__init__(self, distributed=distributed, extrapolate=extrapolate, method=method,
                           training_data_gradients=training_data_gradients, vec_size=vec_size, rec_file=rec_file)

    def load(self, rec_file):
        """Loads stored data to instantiate a meta model.

        Parameters
        ----------
        rec_file : str
            Name of the file in `latom.data.metamodels` where the meta model is stored

        """

        MetaModel.load(self, rec_file)
        self.energy = self.d['energy']

    def save(self, rec_file):
        """Saves data corresponding to a meta model instance.

        Parameters
        ----------
        rec_file : str
            Name of the file in `latom.data.metamodels` where the meta model is stored

        """

        d = {'Isp': self.Isp, 'twr': self.twr, 'm_prop': self.m_prop, 'failures': self.failures, 'energy': self.energy}
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

        MetaModel.compute_grid(self, twr_lim, isp_lim, nb_samp)
        self.energy = np.zeros(nb_samp)

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
            rp : float
                HEO periapsis radius [m]
            t : float
                HEO period [s]
            log_scale : bool
                ``True`` if `twr` values are given in logarithmic scale as `log(twr)`

        """

        self.compute_grid(twr_lim, isp_lim, nb_samp)
        twr_flip = np.flip(self.twr)

        for j in range(nb_samp[1]):  # loop over specific impulses

            print(f"\nMajor Iteration {j}\nSpecific impulse: {self.Isp[j]:.6f} s\n")

            if kwargs['log_scale']:
                sc = Spacecraft(self.Isp[j], np.exp(twr_flip[0]), g=body.g)
            else:
                sc = Spacecraft(self.Isp[j], twr_flip[0], g=body.g)
            tr = TwoDimLLO2ApoContinuationAnalyzer(body, sc, alt, kwargs['rp'], kwargs['t'], t_bounds, twr_flip,
                                                   method, nb_seg, order, solver, snopt_opts=snopt_opts,
                                                   log_scale=kwargs['log_scale'])
            tr.run_continuation()

            self.m_prop[:, j] = np.flip(tr.m_prop_list)
            self.energy[:, j] = np.flip(tr.energy_list)

        self.setup()
        if rec_file is not None:
            self.save(rec_file)

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

        en = RespSurf(self.Isp, self.twr, self.energy, 'Specific energy [m^2/s^2]', nb_lines=nb_lines,
                      log_scale=log_scale)
        en.plot()
        MetaModel.plot(self, nb_lines=nb_lines, kind=kind, log_scale=log_scale)


class TwoDim3PhasesLLO2HEOMetaModel(MetaModel):
    """`TwoDim3PhasesLLO2HEOMetaModel` implements `MetaModel` for a two-dimensional transfer from an LLO to an HEO
    modeled as a three-phases trajectory.

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
            rp : float
                HEO periapsis radius [m]
            t : float
                HEO period [s]
            phase_name : iterable
                Names for the three phases within OpenMDAO

        Returns
        -------
        m_prop : float
            Propellant fraction [-]
        f : bool
            Failure status

        """

        nlp = TwoDim3PhasesLLO2HEONLP(body, sc, alt, kwargs['rp'], kwargs['t'], (-np.pi/2, np.pi/2), t_bounds, method,
                                      nb_seg, order, solver, kwargs['phase_name'], snopt_opts=snopt_opts)

        f = nlp.p.run_driver()
        nlp.cleanup()
        m_prop = 1. - nlp.p.get_val(nlp.phase_name[-1] + '.timeseries.states:m')[-1, -1]

        return m_prop, f
