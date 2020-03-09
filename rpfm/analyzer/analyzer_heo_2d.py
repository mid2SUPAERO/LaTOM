"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
from copy import deepcopy
from time import time

from rpfm.analyzer.analyzer_2d import TwoDimAscAnalyzer, TwoDimAnalyzer
from rpfm.nlp.nlp_heo_2d import TwoDimLLO2HEONLP, TwoDimLLO2ApoNLP, TwoDim3PhasesLLO2HEONLP, TwoDim2PhasesLLO2HEONLP
from rpfm.plots.solutions import TwoDimSolPlot, TwoDimMultiPhaseSolPlot
from rpfm.plots.timeseries import TwoDimStatesTimeSeries, TwoDimControlsTimeSeries
from rpfm.plots.continuation import MassEnergyContinuation, TwoDimTrajectoryContinuation
from rpfm.utils.keplerian_orbit import TwoDimOrb
from rpfm.utils.spacecraft import ImpulsiveBurn
from rpfm.utils.const import g0
from rpfm.utils.pickle_utils import save


class TwoDimLLO2HEOAnalyzer(TwoDimAscAnalyzer):
    """TwoDimLLO2HEOAnalyzer class defines the methods to analyze the results of a two dimensional simulation from a
    Low Lunar Orbit to an Highly Elliptical Orbit.

     The Highly Elliptical Orbit is chosen as first planar approximation of a three-dimensional Near Rectilinear Halo
     Orbit. The transfer is modelled as a single phase trajectory performed at variable thrust.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
        alt : float
    alt : float
        Value of the initial LLO altitude [m]
    rp : float
        Value for the target HEO periselene radius [m]
    t : float
        Value for the guessed trajectory time of flight [s]
    t_bounds : float
        Value for the time of flight bounds [-]
    method : str
        NLP transcription method
    nb_seg : int
        Number of segments for the transcription
    order : int
        Transcription order
    solver : str
        NLP solver
    snopt_opts : dict
        Sets some SNOPT's options. Default is ``None``
    rec_file : str
        Directory path for the solution recording file. Default is ``None``
    check_partials : bool
        Checking of partial derivatives. Default is ``False``
    u_bound : str
        Sets the bound of the radial velocity. Can be ``lower``, ``upper`` or ``None``. Default is
        ``lower``

    Attributes
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    phase_name : str
        Describes the phase name in case of multi-phase trajectories
    nlp : NLP
        Instance of `NLP` object describing the type of Non Linear Problem solver used
    tof : float
        Value of the time of flight resulting by the simulation [s]
    tof_exp : float
        Value of the time of flight of the explicit simulation [s]
    err : float
        Value of the error between the optimized simulation results and the explicit simulation results
    rm_res : float
        Value of the central body radius [-] or [m]
    states_scalers : ndarray
        Reference values of the states with which perform the scaling
    controls_scalers : ndarray
        Reference values of the controls with which perform the scaling

    """

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False, u_bound='lower'):
        """Initializes the `TwoDimLLO2HEOAnalyzer` class variables. """

        TwoDimAscAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimLLO2HEONLP(body, sc, alt, rp, t, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg, order,
                                    solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                    check_partials=check_partials, u_bound=u_bound)

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
        time. The semi-major axis and the eccentricity of the HEO are also displayed.

        """

        sol_plot = TwoDimSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 a=self.nlp.guess.ht.arrOrb.a*(self.rm_res/self.body.R), e=self.nlp.guess.ht.arrOrb.e)
        sol_plot.plot()

    def __str__(self):
        """Prints info on the `TwoDimLLO2HEOAnalyzer`.

        Returns
        -------
        s : str
            Info on `TwoDimLLO2HEOAnalyzer`

        """

        if np.isclose(self.gm_res, 1.0):
            tof = self.tof*self.body.tc/86400
        else:
            tof = self.tof/86400

        lines = ['\n{:^50s}'.format('2D Transfer trajectory from LLO to HEO:'),
                 self.nlp.guess.__str__(),
                 '\n{:^50s}'.format('Optimal transfer:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1. - self.states[-1, -1]/self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', tof, 'days'),
                 TwoDimAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s


class TwoDimLLO2ApoAnalyzer(TwoDimAscAnalyzer):
    """TwoDimLLO2HEOAnalyzer class defines the methods to analyze the results of a two dimensional simulation from a
    Low Lunar Orbit to an Highly Elliptical Orbit.

    The Highly Elliptical Orbit is chosen as first planar approximation of a three-dimensional Near Rectilinear Halo
    Orbit. The optimal control problem is solved for a single phase trajectory performed at constant thrust that
    injects the spacecraft from the Low Lunar Orbit to an intermediate transfer trajectory whose aposelene radius
    coincides with the one of the target Highly Elliptical Orbit. The final insertion manoeuvre is then modelled as
    an ideal impulsive burn due to its limited dV magnitude.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]
    rp : float
        Value for the target HEO periselene radius [m]
    t : float
        Value for the target HEO period [s]
    t_bounds : float
        Value for the time of flight bounds [-]
    method : str
        NLP transcription method
    nb_seg : int
        Number of segments for the transcription
    order : int
        Transcription order
    solver : str
        NLP solver
    snopt_opts : dict
        Sets some SNOPT's options. Default is ``None``
    rec_file : str
        Directory path for the solution recording file. Default is ``None``
    check_partials : bool
        Checking of partial derivatives. Default is ``False``

    Attributes
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    phase_name : str
        Describes the phase name in case of multi-phase trajectories
    nlp : NLP
        Instance of `NLP` object describing the type of Non Linear Problem solver used
    tof : float
        Value of the time of flight resulting by the simulation [s]
    tof_exp : float
        Value of the time of flight of the explicit simulation [s]
    err : float
        Value of the error between the optimized simulation results and the explicit simulation results
    rm_res : float
        Value of the central body radius [- or m]
    states_scalers = ndarray
        Reference values of the states with which perform the scaling
    controls_scalers : ndarray
        Reference values of the controls with which perform the scaling
    alt : float
        Value of the final orbit altitude [m]
    phase_name : str
        Name assigned to the problem phase
    transfer : HohmannTransfer
        Instance of `HohmannTransfer` computing the keplerian parameters of the transfer orbit
    insertion_burn : ImpulsiveBurn
        Instance of `ImpulsiveBurn` defining the delta v required for an impulsive burn
    dv_dep : float
        Delta v required for a manoeuvre [m/s]

    """

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False):
        """Initializes the `TwoDimLLO2ApoAnalyzer` class variables. """

        TwoDimAscAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimLLO2ApoNLP(body, sc, alt, rp, t, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg, order,
                                    solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                    check_partials=check_partials)

        self.transfer = self.insertion_burn = self.dv_dep = None
        self.guess = self.nlp.guess  # save the initial initial guess at analyzer level

    def compute_solution(self, nb=200):
        """Retrieves the optimization results and computes the spacecraft states along the ballistic arc as well as the
        final insertion burn.

        Parameters
        ----------
        nb : int, optional
            Number of points in which the spacecraft states along the ballistic arc are computed. Default is ``200``

        """

        # states and COEs at the end of the departure burn
        states_end = self.states[-1]
        a, e, h, ta = TwoDimOrb.polar2coe(self.gm_res, states_end[0], states_end[2], states_end[3])

        # coasting orbit in dimensional units
        if np.isclose(self.gm_res, 1.0):
            self.transfer = TwoDimOrb(self.body.GM, a=a*self.body.R, e=e)
        else:
            self.transfer = TwoDimOrb(self.body.GM, a=a, e=e)

        # finite dV at departure [m/s]
        self.dv_dep = ImpulsiveBurn.tsiolkovsky_dv(self.sc.m0, states_end[-1], self.sc.Isp)

        # impulsive dV at arrival [m/s]
        sc = deepcopy(self.sc)
        sc.m0 = states_end[-1]
        self.insertion_burn = ImpulsiveBurn(sc, self.guess.ht.arrOrb.va - self.transfer.va)

        # time and states along transfer orbit
        t, states = TwoDimOrb.propagate(self.gm_res, a, e, ta, np.pi, nb)

        # adjust time
        t_pow_end = self.time[-1, -1]
        t_coast_start = t[0, 0]
        t_coast_end = t[-1, 0]
        tof_coast = t_coast_end - t_coast_start

        self.tof = [self.tof, tof_coast]
        self.time = [self.time, (t - t_coast_start) + t_pow_end]

        # add mass
        m = np.vstack((states_end[-1]*np.ones((len(t) - 1, 1)), [self.insertion_burn.mf]))
        states = np.hstack((states, m))

        # stack states and controls
        self.states = [self.states, states]
        self.controls = [self.controls, np.zeros((len(t), 2))]
        self.controls[-1][-1, 0] = self.controls[0][0, 0]

        # adjust theta
        dtheta = ta - self.states[0][-1, 1]
        self.states[0][:, 1] = self.states[0][:, 1] + dtheta
        if self.states_exp is not None:
            self.states_exp[:, 1] = self.states_exp[:, 1] + dtheta

    def get_solutions(self, explicit=True, scaled=False, nb=200):
        """Access the simulation solution.

        Parameters
        ----------
        explicit : bool
            Computes also the explicit simulation. Default is ``True``
        scaled : bool
            Scales the simulation results. Default is ``False``
        nb : int
            Number of points where the coasting arch is computed

        """

        TwoDimAscAnalyzer.get_solutions(self, explicit=explicit, scaled=scaled)

        self.compute_solution(nb=nb)

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
        time. The semi-major axis and the eccentricity of the HEO are also displayed.

        """

        states_plot = TwoDimStatesTimeSeries(self.rm_res, self.time[0], self.states[0], self.time_exp, self.states_exp)

        if np.isclose(self.rm_res, 1.0):
            controls_plot = TwoDimControlsTimeSeries(self.time[0], self.controls[0], units=('-', '-'), threshold=None)
        else:
            controls_plot = TwoDimControlsTimeSeries(self.time[0], self.controls[0], threshold=None)

        sol_plot = TwoDimMultiPhaseSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                           self.states_exp, a=self.guess.ht.arrOrb.a*(self.rm_res/self.body.R),
                                           e=self.guess.ht.arrOrb.e)

        states_plot.plot()
        controls_plot.plot()
        sol_plot.plot()

    def __str__(self):
        """Prints info on the `TwoDimLLO2ApoAnalyzer`.

        Returns
        -------
        s : str
            Info on `TwoDimLLO2ApoAnalyzer`

        """

        if np.isclose(self.gm_res, 1.0):
            time_scaler = self.body.tc
        else:
            time_scaler = 1.0

        lines = ['\n{:^50s}'.format('2D Transfer trajectory from LLO to HEO:'),
                 self.guess.__str__(),
                 '\n{:^50s}'.format('Optimal coasting phase:'),
                 self.transfer.__str__(),
                 '\n{:^50s}'.format('Optimal powered phases:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:',
                                                   1 - self.insertion_burn.mf/self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', sum(self.tof)*time_scaler/86400, 'days'),
                 '\n{:^50s}'.format('Departure burn:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.guess.pow.dv_inf, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Finite dV:', self.dv_dep, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Burn time:', self.tof[0]*time_scaler, 's'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1 - self.states[0][-1, -1]/self.sc.m0, ''),
                 '\n{:^50s}'.format('Injection burn:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.insertion_burn.dv, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', self.insertion_burn.dm/self.sc.m0, ''),
                 TwoDimAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s


class TwoDimLLO2ApoContinuationAnalyzer(TwoDimLLO2ApoAnalyzer):
    """TwoDimLLO2ApoContinuationAnalyzer class defines the methods to analyze the results of a two dimensional
    simulation from a Low Lunar Orbit to an Highly Elliptical Orbit using a continuation method over several
    thrust/weight ratios.

    The Highly Elliptical Orbit is chosen as first planar approximation of a three-dimensional Near Rectilinear Halo
    Orbit. The optimal control problem is solved for a single phase trajectory performed at constant thrust that
    injects the spacecraft from the Low Lunar Orbit to an intermediate transfer trajectory whose aposelene radius
    coincides with the one of the target Highly Elliptical Orbit. The final insertion manoeuvre is then modelled as
    an ideal impulsive burn due to its limited dV magnitude.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]
    rp : float
        Value for the target HEO periselene radius [m]
    t : float
        Value for the target HEO period [s]
    t_bounds : float
        Value for the time of flight bounds [-]
    twr_list : iterable
        List of thrust/weight ratios over which the continuation of the solutions is performed [-]
    method : str
        NLP transcription method
    nb_seg : int
        Number of segments for the transcription
    order : int
        Transcription order
    solver : str
        NLP solver
    snopt_opts : dict, optional
        Sets some SNOPT's options. Default is ``None``
    check_partials : bool, optional
        Checking of partial derivatives. Default is ``False``
    log_scale : bool, optional
        ``True`` if `twr_list` is provided in logarithmic scale, ``False`` otherwise. Default is ``False``
    significant_figures : int, optional
        Significant figures to be considered while casting the list of thrust/weight ratios. Default is ``4``

    Attributes
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    phase_name : str
        Describes the phase name in case of multi-phase trajectories
    nlp : NLP
        Instance of `NLP` object describing the type of Non Linear Problem solver used
    tof : float
        Value of the time of flight resulting by the simulation [s]
    tof_exp : float
        Value of the time of flight of the explicit simulation [s]
    err : float
        Value of the error between the optimized simulation results and the explicit simulation results
    rm_res : float
        Value of the central body radius [- or m]
    states_scalers : ndarray
        Reference values of the states with which perform the scaling
    controls_scalers : ndarray
        Reference values of the controls with which perform the scaling
    alt : float
        Value of the final orbit altitude [m]
    phase_name : str
        Name assigned to the problem phase
    transfer : HohmannTransfer
        Instance of `HohmannTransfer` computing the keplerian parameters of the transfer orbit
    insertion_burn : ImpulsiveBurn
        Instance of `ImpulsiveBurn` defining the delta v required for an impulsive burn
    dv_dep : float
        Delta v required for a manoeuvre [m/s]
    twr_list : ndarray
        List of thrust/weight ratios over which the continuation is performed [-]
    m_prop_list : ndarray
        List of propellant masses corresponding to `twr_list` [kg]
    energy_list : ndarray
        List of specific energies of the spacecraft along the ballistic arc corresponding to `twr_list` [m^2/s^2]
    sol_list : dict
        Dictionary of optimal solutions corresponding to `twr_list`

    """

    def __init__(self, body, sc, alt, rp, t, t_bounds, twr_list, method, nb_seg, order, solver, snopt_opts=None,
                 check_partials=False, log_scale=False, significant_figures=4):
        """Initializes the `TwoDimLLO2ApoContinuationAnalyzer` class variables. """

        TwoDimLLO2ApoAnalyzer.__init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver,
                                       snopt_opts=snopt_opts, rec_file=None, check_partials=check_partials)

        if log_scale:
            self.log_twr_list = np.around(np.asarray(twr_list), significant_figures)
            self.twr_list = np.exp(self.log_twr_list)
        else:
            self.log_twr_list = None
            self.twr_list = np.around(np.asarray(twr_list), significant_figures)
        self.m_prop_list = np.zeros(np.shape(self.twr_list))
        self.energy_list = np.zeros(np.shape(self.twr_list))
        self.sol_list = {}

    def compute_energy_mprop(self, r, u, v, m_coast):
        """Computes the specific energy of the spacecraft on the ballistic arc and the total propellant mass including
        the final insertion burn.

        Parameters
        ----------
        r : float
            Radius at the end of the first powered phase [m]
        u : float
            Radial velocity at the end of the first powered phase [m/s]
        v : float
            Tangential velocity at the end of the first powered phase [m/s]
        m_coast : float
            Spacecraft mass on the ballistic arc [kg]

        Returns
        -------
        en : float
            Specific energy of the spacecraft on the ballistic arc [m/s]
        m_prop : float
            Total propellant mass including the final insertion burn [kg]

        """

        en = TwoDimOrb.polar2energy(self.body.GM, r, u, v)  # specific energy on ballistic arc [m^2/s^2]
        va = (2 * (en + self.body.GM / self.nlp.guess.ht.arrOrb.ra)) ** 0.5  # ballistic arc apoapsis velocity [m/s]
        dva = self.guess.ht.arrOrb.va - va  # apoapsis dv [m/s]
        m_prop = self.sc.m0 - ImpulsiveBurn.tsiolkovsky_mf(m_coast, dva, self.sc.Isp)  # total propellant mass [kg]

        return en, m_prop

    def run_continuation(self, rec_file=None):
        """Iterative solution of all optimal control problems corresponding to the different thrust/weight ratios in
        `twr_list` using a continuation method in which the solution of the problem ``i`` is set as initial guess for
        the problem ``i + 1``. The first solution is computed using the initial guess provided by `TwoDimLLO2HEOGuess`
        class.

        Parameters
        ----------
        rec_file : str or None, optional
            Directory path for the file in which the thrust/weight ratios, propellant masses, specific energies and
            solutions are recorded or ``None``. Default is ``None``

        """

        nlp = None
        params = {'ra_heo': self.guess.ht.arrOrb.ra, 'rp_llo': self.guess.ht.depOrb.rp,
                  'vp_hoh': self.guess.ht.transfer.vp, 'thetaf_pow': self.guess.pow.thetaf}

        t0_loop = time()
        for i in range(np.size(self.twr_list)):

            if i == 0:  # first solution
                nlp = self.nlp
            else:  # subsequent solutions
                self.sc.update_twr(self.twr_list[i])
                nlp = TwoDimLLO2ApoNLP(self.body, self.sc, self.alt, None, None, (-np.pi / 2, np.pi / 2), None,
                                       self.nlp.method, self.nlp.nb_seg, self.nlp.order, self.nlp.solver,
                                       self.phase_name, snopt_opts=self.nlp.snopt_opts, params=params)

            # solve NLP
            print(f"\nIteration {i}\nThrust/weight ratio: {self.sc.twr:.6f}\n")
            t0_iter = time()
            failed = nlp.p.run_driver()
            tf_iter = time()
            nlp.cleanup()
            print(f"\nTime to solve the NLP problem: {(tf_iter - t0_iter):.6f} s\n")

            if failed:
                break

            # extract and store the NLP solution
            params['tof'], params['states'], params['controls'] = self.get_discretization_phase(nlp.p, nlp.phase_name)
            if self.log_twr_list is None:
                k = str(self.twr_list[i])
            else:
                k = str(self.log_twr_list[i])
            self.sol_list[k] = self.get_solution_dictionary(nlp.p)

            # compute the specific energy of the spacecraft at the end of the powered phase and the total required
            # propellant mass including the final insertion burn
            states_end = params['states'][-1]
            en, m_prop = self.compute_energy_mprop(states_end[0], states_end[2], states_end[3], states_end[-1])

            # store the spacecraft energy and propellant mass
            self.energy_list[i] = en
            self.m_prop_list[i] = m_prop

        tf_loop = time()
        print(f"\nTotal time for continuation loop: {(tf_loop - t0_loop):.6f} s\n")

        self.nlp = nlp

        if rec_file is not None:
            d = {'twr': self.twr_list, 'm_prop': self.m_prop_list, 'energy': self.energy_list, 'sol': self.sol_list}
            save(d, rec_file)

    def plot(self):

        mass_energy_continuation = MassEnergyContinuation(self.twr_list, self.m_prop_list/self.sc.m0, self.energy_list)

        if self.log_twr_list is None:
            trajectory_continuation = TwoDimTrajectoryContinuation(self.body.R, self.body.R + self.alt, self.sol_list)
        else:
            trajectory_continuation = TwoDimTrajectoryContinuation(self.body.R, self.body.R + self.alt, self.sol_list,
                                                                   log_scale=True)

        mass_energy_continuation.plot()
        trajectory_continuation.plot()

        TwoDimLLO2ApoAnalyzer.plot(self)


class TwoDimMultiPhasesLLO2HEOAnalyzer(TwoDimAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional multi phases LLO to HEO simulation

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics

    Attributes
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    phase_name : str
        Describes the phase name in case of multi-phase trajectories
    nlp : NLP
        Instance of `NLP` object describing the type of Non Linear Problem solver used
    tof : float
        Value of the time of flight resulting by the simulation [s]
    tof_exp : float
        Value of the time of flight of the explicit simulation [s]
    err : float
        Value of the error between the optimized simulation results and the explicit simulation results
    rm_res : float
        Value of the central body radius [-] or [m]
    states_scalers : ndarray
        Reference values of the states with which perform the scaling
    controls_scalers : ndarray
        Reference values of the controls with which perform the scaling
    transfer : HohmannTransfer
        Instance of `HohmannTransfer` computing the keplerian parameters of the transfer orbit
    dv : float
        Delta v required for a manoeuvre [m/s]

    """

    def __init__(self, body, sc):
        """Initializes the `TwoDimMultiPhasesLLO2HEOAnalyzer` class variables. """

        TwoDimAnalyzer.__init__(self, body, sc)

        self.transfer = None
        self.dv = []

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
            time. The semi-major axis and the eccentricity of the HEO are also displayed.
        """

        coe_inj = TwoDimOrb.polar2coe(self.gm_res, self.states[-1][-1, 0], self.states[-1][-1, 2],
                                      self.states[-1][-1, 3])

        dtheta = coe_inj[-1] - self.states[-1][-1, 1]

        sol_plot = TwoDimMultiPhaseSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                           self.states_exp, a=self.nlp.guess.ht.arrOrb.a*(self.rm_res/self.body.R),
                                           e=self.nlp.guess.ht.arrOrb.e, dtheta=dtheta)
        sol_plot.plot()


class TwoDim2PhasesLLO2HEOAnalyzer(TwoDimMultiPhasesLLO2HEOAnalyzer):

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False):

        TwoDimMultiPhasesLLO2HEOAnalyzer.__init__(self, body, sc)

        self.phase_name = ('dep', 'arr')
        self.nlp = TwoDim2PhasesLLO2HEONLP(body, sc, alt, rp, t, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg, order,
                                           solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                           check_partials=check_partials)

    def get_time_series(self, p, scaled=False):

        tof1, t1, s1, c1 = self.get_time_series_phase(p, self.nlp.phase_name[0], scaled=scaled)
        tof2, t2, s2, c2 = self.get_time_series_phase(p, self.nlp.phase_name[1], scaled=scaled)

        return [tof1, tof2], [t1, t2], [s1, s2], [c1, c2]

    def compute_coasting_arc(self, nb=200):

        # COEs as (a, e, h, ta) at the end of the 1st powered phase and at the beginning of the 2nd one
        coe1 = TwoDimOrb.polar2coe(self.gm_res, self.states[0][-1, 0], self.states[0][-1, 2], self.states[0][-1, 3])
        coe2 = TwoDimOrb.polar2coe(self.gm_res, self.states[1][0, 0], self.states[1][0, 2], self.states[1][0, 3])

        if np.allclose(coe1[:3], coe2[:3], rtol=1e-4, atol=1e-6):

            t, states = TwoDimOrb.propagate(self.gm_res, coe1[0], coe1[1], coe1[-1], coe2[-1], nb)
            tof = t[-1, 0] - t[0, 0]

            # adjust time
            self.tof = [self.tof[0], tof, self.tof[1]]
            self.time = [self.time[0], t + self.tof[0], self.time[1] + self.tof[0] + tof]

            # adjust theta
            states[:, 1] = states[:, 1] - coe1[-1] + self.states[0][-1, 1]
            states = np.hstack((states, self.states[0][-1, -1] * np.ones((len(t), 1))))
            self.states[1][:, 1] = self.states[1][:, 1] + states[-1, 1]

            # adjust states and controls
            self.states = [self.states[0], states, self.states[1]]
            self.controls = [self.controls[0], np.zeros((len(t), 2)), self.controls[1]]

        else:
            raise ValueError('a, e, h are not constant throughout the coasting phase')

        return coe1, coe2

    def get_solutions(self, explicit=True, scaled=False, nb=200):

        TwoDimAnalyzer.get_solutions(self, explicit=explicit, scaled=scaled)

        self.compute_coasting_arc(nb=nb)

    def __str__(self):

        pass


class TwoDim3PhasesLLO2HEOAnalyzer(TwoDimMultiPhasesLLO2HEOAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional 3 phases LLO to HEO simulation.

     It gives the results of an optimal trajectory from a Low Lunar Orbit of chosen altitude to an High Elliptical
     Orbit which can represent a transposition of an Halo orbit in 2D. The trajectory is modeled as the succession
     of three different phases: a first powered phase for departure, a coasting arch and a final powered phase for
     the injection.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
        alt : float
    alt : float
        Value of the initial LLO altitude [m]
    rp : float
        Value for the target HEO periselene radius [m]
    t : float
        Value for the guessed trajectory time of flight [s]
    t_bounds : float
        Value for the time of flight bounds [-]
    method : str
        NLP transcription method
    nb_seg : int
        Number of segments for the transcription
    order : int
        Transcription order
    solver : str
        NLP solver
    snopt_opts : dict
        Sets some SNOPT's options. Default is ``None``
    rec_file : str
        Directory path for the solution recording file. Default is ``None``
    check_partials : bool
        Checking of partial derivatives. Default is ``False``

    Attributes
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    phase_name : str
        Describes the phase name in case of multi-phase trajectories. Can be ``dep``, ``coast`` or ``arr``.
    nlp : NLP
        Instance of `NLP` object describing the type of Non Linear Problem solver used
    tof : float
        Value of the time of flight resulting by the simulation [s]
    tof_exp : float
        Value of the time of flight of the explicit simulation [s]
    err : float
        Value of the error between the optimized simulation results and the explicit simulation results
    rm_res : float
        Value of the central body radius [-] or [m]
    states_scalers : ndarray
        Reference values of the states with which perform the scaling
    controls_scalers : ndarray
        Reference values of the controls with which perform the scaling

    """

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False):
        """Initializes the `TwoDim3PhasesLLO2HEOAnalyzer` class variables. """

        TwoDimMultiPhasesLLO2HEOAnalyzer.__init__(self, body, sc)

        self.phase_name = ('dep', 'coast', 'arr')

        self.nlp = TwoDim3PhasesLLO2HEONLP(body, sc, alt, rp, t, (-np.pi/2, np.pi/2), t_bounds, method, nb_seg, order,
                                           solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                           check_partials=check_partials)

    def get_time_series(self, p, scaled=False):
        """Access the time series of the problem.

        Parameters
        ----------
        p : Problem
            Instance of `Problem` class
        scaled : bool
            Scales the simulation results

        Returns
        -------
        tof : float
            Time of flight resulting from the optimized simulation phase [-] or [s]
        t : ndarray
            Time of flight time series for the optimized simulation phase [-] or [s]
        states : ndarray
            States time series for the optimized simulation phase
        controls : ndarray
            Controls time series for the optimized simulation phase

        """
        tof = []
        t = []
        states = []
        controls = []

        for i in range(3):
            tofi, ti, si, ci = self.get_time_series_phase(p, self.nlp.phase_name[i], scaled=scaled)

            tof.append(tofi)
            t.append(ti)
            states.append(si)
            controls.append(ci)

        return tof, t, states, controls

    def get_solutions(self, explicit=True, scaled=False):
        """Access the simulation solution.

        Parameters
        ----------
        explicit : bool
            Computes also the explicit simulation. Default is ``True``
        scaled : bool
            Scales the simulation results. Default is ``False``

        """

        TwoDimAnalyzer.get_solutions(self, explicit=explicit, scaled=scaled)

        coe = TwoDimOrb.polar2coe(self.gm_res, self.states[0][-1, 0], self.states[0][-1, 2], self.states[0][-1, 3])

        self.transfer = TwoDimOrb(self.body.GM, a=coe[0]*self.body.R/self.rm_res, e=coe[1])
        for i in [0, 2]:
            self.dv.append(self.sc.Isp*g0*np.log(self.states[i][0, -1]/self.states[i][-1, -1]))

    def __str__(self):
        """Prints info on the `TwoDim3PhasesLLO2HEOAnalyzer`.

        Returns
        -------
        s : str
            Info on `TwoDim3PhasesLLO2HEOAnalyzer`

        """

        if np.isclose(self.gm_res, 1.0):
            time_scaler = self.body.tc
        else:
            time_scaler = 1.0

        lines = ['\n{:^50s}'.format('2D Transfer trajectory from LLO to HEO:'),
                 self.nlp.guess.__str__(),
                 '\n{:^50s}'.format('Coasting orbit:'),
                 self.transfer.__str__(),
                 '\n{:^50s}'.format('Optimal transfer:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1 - self.states[-1][-1, -1]/self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', sum(self.tof)*time_scaler/86400, 'days'),
                 '\n{:^50s}'.format('Departure burn:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.nlp.guess.pow1.dv_inf, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Finite dV:', self.dv[0], 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Burn time:', self.tof[0]*time_scaler, 's'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', 1 - self.states[0][-1, -1]/self.sc.m0, ''),
                 '\n{:^50s}'.format('Injection burn:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Impulsive dV:', self.nlp.guess.pow2.dv_inf, 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Finite dV:', self.dv[-1], 'm/s'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Burn time:', self.tof[-1]*time_scaler, 's'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:',
                                                 1 - self.states[-1][-1, -1]/self.states[-1][0, -1], ''),
                 TwoDimAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s
