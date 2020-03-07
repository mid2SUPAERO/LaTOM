"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from rpfm.analyzer.analyzer import Analyzer
from rpfm.nlp.nlp_2d import TwoDimAscConstNLP, TwoDimAscVarNLP, TwoDimAscVToffNLP, TwoDimDescConstNLP, \
    TwoDimDescTwoPhasesNLP, TwoDimDescVarNLP, TwoDimDescVLandNLP
from rpfm.plots.solutions import TwoDimSolPlot, TwoDimTwoPhasesSolPlot
from rpfm.utils.const import states_2d
from rpfm.guess.guess_2d import HohmannTransfer, ImpulsiveBurn
from rpfm.utils.keplerian_orbit import TwoDimOrb


class TwoDimAnalyzer(Analyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional simulation.

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

    """

    def __init__(self, body, sc):
        """Initializes the `TwoDimAnalyzer` class variables. """

        Analyzer.__init__(self, body, sc)

        # reference values:
        # radius: equatorial radius of the central attracting body (Moon)
        # velocity: velocity on a circular orbit at zero altitude
        # thrust magnitude: initial spacecraft weight on the Moon surface
        self.states_scalers = np.array([self.body.R, 1.0, self.body.vc, self.body.vc, 1.0])
        self.controls_scalers = np.array([self.body.g*self.sc.m0, 1.0])

    def get_time_series_phase(self, p, phase_name, scaled=False):
        """Access the time series of one of the problem phases.

        Parameters
        ----------
        p : Problem
            Instance of `Problem` class
        phase_name : str
            Name defined for the problem phase
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

        tof = float(p.get_val(phase_name + '.t_duration'))  # non dimensional time of flight [-]
        t = p.get_val(phase_name + '.timeseries.time')  # non dimensional time [-]

        states = np.empty((np.size(t), 0))

        for k in states_2d:
            s = p.get_val(phase_name + '.timeseries.states:' + k)  # non dimensional states [-]
            states = np.append(states, s, axis=1)

        alpha = p.get_val(phase_name + '.timeseries.controls:alpha')  # thrust direction [rad]

        try:
            thrust = p.get_val(phase_name + '.timeseries.controls:thrust')  # non dimensional thrust [-]
        except KeyError:
            thrust = p.get_val(phase_name + '.design_parameters:thrust')
            thrust = thrust*np.ones((len(alpha), 1))  # non dimensional thrust [-]

        controls = np.hstack((thrust, alpha))

        if not scaled:
            tof = tof*self.body.tc  # dimensional time of flight [s]
            t = t*self.body.tc  # dimensional time [s]
            states = states*self.states_scalers  # dimensional states [m, rad, m/s, m/s, kg]
            controls = controls*self.controls_scalers  # dimensional controls [N, rad]

        return tof, t, states, controls

    def get_tof_states_alpha(self, p, phase_name, scaled=False):

        tof = float(p.get_val(phase_name + '.t_duration'))  # non dimensional time of flight [-]

        # non dimensional states [-]
        states = p.get_val(phase_name + '.states:' + states_2d[0])
        for k in states_2d[1:]:
            s = p.get_val(phase_name + '.states:' + k)
            states = np.append(states, s, axis=1)

        # thrust direction [rad]
        alpha = p.get_val(phase_name + '.controls:alpha')

        if not scaled:
            tof = tof*self.body.tc  # dimensional time of flight [s]
            states = states*self.states_scalers  # dimensional states [m, rad, m/s, m/s, kg]

        return tof, states, alpha

    def __str__(self):
        """Prints info on the TwoDimAnalyzer.

        Returns
        -------
        s : str
            Info lines

        """

        lines = [self.sc.__str__(), self.nlp.__str__()]

        if self.err is not None:

            try:
                lines_err = ['\n{:^50s}'.format('Error:'),
                             '\n{:<25s}{:>20.12f}{:>5s}'.format('Radius:', self.err[0] / 1e3, 'km'),
                             '{:<25s}{:>20.12f}{:>5s}'.format('Angle:', self.err[1] * np.pi / 180, 'deg'),
                             '{:<25s}{:>20.12f}{:>5s}'.format('Radial velocity:', self.err[2] / 1e3, 'km/s'),
                             '{:<25s}{:>20.12f}{:>5s}'.format('Tangential velocity:', self.err[3] / 1e3, 'km/s'),
                             '{:<25s}{:>20.12f}{:>5s}'.format('Mass:', self.err[4], 'kg')]

            except TypeError:
                lines_err = ['\n{:^50s}'.format('Error:'),
                             '\n{:<25s}{:>50s}{:>5s}'.format('Radius:', str(self.err[:, 0]/1e3), 'km'),
                             '{:<25s}{:>50s}{:>5s}'.format('Angle:', str(self.err[:, 1]*np.pi/180), 'deg'),
                             '{:<25s}{:>50s}{:>5s}'.format('Radial velocity:', str(self.err[:, 2]/1e3), 'km/s'),
                             '{:<25s}{:>50s}{:>5s}'.format('Tangential velocity:', str(self.err[:, 3]/1e3), 'km/s'),
                             '{:<25s}{:>50s}{:>5s}'.format('Mass:', str(self.err[:, 4]), 'kg')]

            lines.extend(lines_err)

        s = '\n'.join(lines)

        return s


class TwoDimSinglePhaseAnalyzer(TwoDimAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional single phase simulation.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]

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

    """

    def __init__(self, body, sc, alt):
        """Initializes the `TwoDimSinglePhaseAnalyzer` class attributes. """

        TwoDimAnalyzer.__init__(self, body, sc)

        self.alt = alt
        self.phase_name = 'powered'

    def get_time_series(self, p, scaled=False):
        """Access the time series of one of the problem phases.

        Parameters
        ----------
        p : Problem
            Instance of `Problem` class
        scaled : bool
            Scales the simulation results

        Returns
        -------
        tof : float
            Time of flight resulting from the optimized simulation phase [s]
        t : ndarray
            Time of flight time series for the optimized simulation phase [s]
        states : ndarray
            States time series for the optimized simulation phase
        controls : ndarray
            Controls time series for the optimized simulation phase

        """

        tof, t, states, controls = self.get_time_series_phase(p, self.nlp.phase_name, scaled=scaled)

        return tof, t, states, controls

    def __str__(self):
        """Prints infos on the `TwoDimSinglePhaseAnalyser`.

        Returns
        -------
        s : str
        Info on the `TwoDimSinglePhaseAnalyser`

        """

        lines = ['{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', (1 - self.states[-1, -1] / self.sc.m0), ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', self.tof, 's'),
                 TwoDimAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s


class TwoDimAscAnalyzer(TwoDimSinglePhaseAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional ascent simulation.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]

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

    """

    def __str__(self):
        """Prints info on the `TwoDimAscAnalyzer`.

        Returns
        -------
        s : str
        Info on `TwoDimAscAnalyzer`

        """
        lines = ['\n{:^50s}'.format('2D Ascent Trajectory:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Final orbit altitude:', self.alt / 1e3, 'km'),
                 TwoDimSinglePhaseAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s


class TwoDimAscConstAnalyzer(TwoDimAscAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional ascent with constant thrust
    simulation.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]
    theta : float
        Value for the guessed angle spawn during the trajectory [rad]
    tof : float
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
        Value of the central body radius [- or m]
    states_scalers : ndarray
        Reference values of the states with which perform the scaling
    controls_scalers : ndarray
        Reference values of the controls with which perform the scaling
    alt : float
        Value of the final orbit altitude [m]
    phase_name : str
        Name assigned to the problem phase

    """

    def __init__(self, body, sc, alt, theta, tof, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound='lower'):
        """Initializes the `TwoDimAscConstAnalyzer` class variables. """

        TwoDimAscAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimAscConstNLP(body, sc, alt, theta, (-np.pi / 2, np.pi / 2), tof, t_bounds, method, nb_seg,
                                     order,
                                     solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                     check_partials=check_partials, u_bound=u_bound)

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
        time.

        """

        sol_plot = TwoDimSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                 self.states_exp, threshold=None)

        sol_plot.plot()


class TwoDimAscVarAnalyzer(TwoDimAscAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional ascent with variable thrust
    simulation.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]
    theta : float
        Value for the guessed angle spawn during the trajectory [rad]
    tof : float
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
        Value of the central body radius [- or m]
    states_scalers : ndarray
        Reference values of the states with which perform the scaling
    controls_scalers : ndarray
        Reference values of the controls with which perform the scaling
    alt : float
        Value of the final orbit altitude [m]
    phase_name : str
        Name assigned to the problem phase

    """

    def __init__(self, body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False, u_bound='lower'):
        """Initializes the `TwoDimAscVarAnalyzer` class variables. """

        TwoDimAscAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimAscVarNLP(body, sc, alt, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg, order, solver,
                                   self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                   check_partials=check_partials, u_bound=u_bound)

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
        time.

        """

        sol_plot = TwoDimSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                 self.states_exp)

        sol_plot.plot()


class TwoDimAscVToffAnalyzer(TwoDimAscVarAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional ascent with vertical take off
    simulation.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]
    alt_safe : float
        Value of the minimun safe altitude to avoid geographical constraints [m]
    slope : float
        Value of the slope of the constraint on minimum safe altitude [-]
    theta : float
        Value for the guessed angle spawn during the trajectory [rad]
    tof : float
        Value for the guessed trajectory time of flight [-]
    t_bounds : float
        Value for the time of flight bounds [s]
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
        Value of the central body radius [- or m]
    states_scalers : ndarray
        Reference values of the states with which perform the scaling
    controls_scalers : ndarray
        Reference values of the controls with which perform the scaling
    alt : float
        Value of the final orbit altitude [m]
    phase_name : str
        Name assigned to the problem phase
    alt_safe : float
        Value of the minimum safe altitude to avoid geographical constraints [m]
    slope : float
        Value of the slope of the constraint on minimum safe altitude [-]
    r_safe : float
        Value of the minimum orbit radius to be compliant with the constraints [m]

    """

    def __init__(self, body, sc, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound='lower'):
        """Initializes the `TwoDimAscVarAnalyzer` class variables. """

        TwoDimAscAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimAscVToffNLP(body, sc, alt, alt_safe, slope, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg,
                                     order, solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                     check_partials=check_partials, u_bound=u_bound)

        self.alt_safe = alt_safe
        self.slope = slope
        self.r_safe = None

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

        self.r_safe = self.nlp.p.get_val(self.nlp.phase_name + '.timeseries.r_safe')

        if not scaled:
            self.r_safe = self.r_safe*self.body.R

    def __str__(self):
        """Prints info on the `TwoDimAscVToffAnalyzer`.

        Returns
        -------
        s : str
            Info on `TwoDimAscVToffAnalyzer`

        """

        lines = ['\n{:^50s}'.format('2D Ascent Trajectory with Safe Altitude:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Final orbit altitude:', self.alt / 1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Safe altitude:', self.alt_safe / 1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Slope:', self.slope, ''),
                 TwoDimSinglePhaseAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
        time.

        """

        sol_plot = TwoDimSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                 self.states_exp, r_safe=self.r_safe)

        sol_plot.plot()


class TwoDimDescAnalyzer(TwoDimSinglePhaseAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional descent simulation.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]

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

    """
    def __str__(self):
        """Prints info on the `TwoDimDescAnalyzer`.

        Returns
        -------
        s : str
            Info on `TwoDimDescAnalyzer`

        """

        lines = ['\n{:^50s}'.format('2D Descent Trajectory:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Initial orbit altitude:', self.alt / 1e3, 'km'),
                 TwoDimSinglePhaseAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s


class TwoDimDescConstAnalyzer(TwoDimDescAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional descent simulation.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]
    alt_p : float
        Value of the orbit periselene altitude [m]
    theta : float
        Value for the guessed angle spawn during the trajectory [rad]
    tof : float
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
        ``upper``

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
    alt_p : float
        Value of the final orbit altitude [m]

    """

    def __init__(self, body, sc, alt, alt_p, theta, tof, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound='upper'):
        """Initializes the `TwoDimDescConstAnalyzer` class variables. """
        TwoDimDescAnalyzer.__init__(self, body, sc, alt)

        self.alt_p = alt_p

        dep = TwoDimOrb(body.GM, a=(body.R + alt), e=0.0)
        arr = TwoDimOrb(body.GM, a=(body.R + alt_p), e=0.0)

        self.ht = HohmannTransfer(body.GM, dep, arr)
        self.deorbit_burn = ImpulsiveBurn(sc, self.ht.dva)

        self.nlp = TwoDimDescConstNLP(body, self.deorbit_burn.sc, alt_p, self.ht.transfer.vp, theta, (0.0, 1.5*np.pi),
                                      tof, t_bounds, method, nb_seg, order, solver, self.phase_name,
                                      snopt_opts=snopt_opts, rec_file=rec_file, check_partials=check_partials,
                                      u_bound=u_bound)

    def __str__(self):
        """Prints info on the `TwoDimDescConstAnalyzer`.

        Returns
        -------
        s : str
            Info on `TwoDimDescConstAnalyzer`

        """

        lines = ['\n{:^50s}'.format('2D Descent Trajectory at constant thrust:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Parking orbit altitude:', self.alt / 1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Periapsis altitude:', self.alt_p / 1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Deorbit burn fraction:', self.deorbit_burn.dm / self.sc.m0, ''),
                 TwoDimSinglePhaseAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
        time.

        """

        sol_plot = TwoDimSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                 self.states_exp, threshold=None, kind='descent')

        sol_plot.plot()


class TwoDimDescVarAnalyzer(TwoDimDescAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional descent simulation.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]
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
        ``upper``

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

    """

    def __init__(self, body, sc, alt, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False, u_bound='upper'):
        """Initializes the `TwoDimDescVarAnalyzer` class variables. """

        TwoDimDescAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimDescVarNLP(body, sc, alt, (0.0, 1.5 * np.pi), t_bounds, method, nb_seg, order, solver,
                                    self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                    check_partials=check_partials, u_bound=u_bound)

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
        time.

        """

        sol_plot = TwoDimSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                 self.states_exp, kind='descent')

        sol_plot.plot()


class TwoDimDescVLandAnalyzer(TwoDimDescVarAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional descent with vertical landing
    simulation.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]
    alt_safe : float
        Value of the minimum safe altitude to avoid geographical constraints [m]
    slope : float
        Value of the slope of the constraint on minimum safe altitude [-]
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
        ``upper``

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
    alt_safe : float
        Value of the minimum safe altitude to avoid geographical constraints [m]
    slope : float
        Value of the slope of the constraint on minimum safe altitude [-]
    r_safe : float
        Value of the minimum orbit radius to be compliant with the constraints [m]

    """

    def __init__(self, body, sc, alt, alt_safe, slope, t_bounds, method, nb_seg, order, solver, snopt_opts=None,
                 rec_file=None, check_partials=False, u_bound='upper'):
        """Initializes the `TwoDimDescVLandAnalyzer` class variables. """

        TwoDimDescAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimDescVLandNLP(body, sc, alt, alt_safe, slope, (0.0, 1.5 * np.pi), t_bounds, method, nb_seg,
                                      order, solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                      check_partials=check_partials, u_bound=u_bound)

        self.alt_safe = alt_safe
        self.slope = slope
        self.r_safe = None

    def get_time_series(self, p, scaled=False):
        """Access the time series of the simulation.

        Parameters
        ----------
        p : Problem
            Instance of `Problem` class
        scaled : bool
            Scales the simulation results

        Returns
        -------
        tof : float
            Time of flight resulting from the optimized simulation [s]
        t : ndarray
            Time of flight time series for the optimized simulation [s]
        states : ndarray
            States time series for the optimized simulation
        controls : ndarray
            Controls time series for the optimized simulation

        """

        tof, t, states, controls = TwoDimDescVarAnalyzer.get_time_series(self, p, scaled=scaled)

        states[:, 1] = states[:, 1] - states[0, 1]

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

        self.r_safe = self.nlp.p.get_val(self.nlp.phase_name + '.timeseries.r_safe')

        if not scaled:
            self.r_safe = self.r_safe*self.body.R

    def __str__(self):
        """Prints info on the `TwoDimDescVLandAnalyzer`.

        Returns
        -------
        s : str
            Info on `TwoDimAscVToffAnalyzer`

        """

        lines = ['\n{:^50s}'.format('2D Descent Trajectory with Safe Altitude:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Initial orbit altitude:', self.alt / 1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Safe altitude:', self.alt_safe / 1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Slope:', self.slope, ''),
                 TwoDimSinglePhaseAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
        time.

        """

        sol_plot = TwoDimSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                 self.states_exp, r_safe=self.r_safe, kind='descent')

        sol_plot.plot()


class TwoDimDescTwoPhasesAnalyzer(TwoDimAnalyzer):
    """Analyzer class defines the methods to analyze the results of a two dimensional descent composed by two phases.

    Parameters
    ----------
    body : Primary
        Instance of `Primary` class describing the central attracting body
    sc : Spacecraft
        Instance of `Spacecraft` class describing the spacecraft characteristics
    alt : float
        Value of the final orbit altitude [m]
    alt_switch : float
        Value of the minimum safe altitude to avoid geographical constraints [m]
    theta : float
        Value for the guessed angle spawn during the trajectory [rad]
    tof : float
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
    fix  : str
        Chooses to switch from the optimized phase to the vertical one at a specific altitude or time. Can be
        ``alt`` or ``time``. Default is ``alt``

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
    alt_p: float
        Value of the minimum safe altitude to avoid geographical constraints [m]
    alt_switch : float
        Value of the minimum safe altitude to avoid geographical constraints [m]
    ht : `Guess_2d`
        Instance of `Guess_2d` class to define an Hohmann transfer trajectory
    deorbit_burn : `Guess_2d`
        Instance of `Guess_2d` class to define an an impulsive burn

    """

    def __init__(self, body, sc, alt, alt_p, alt_switch, theta, tof, t_bounds, method, nb_seg, order, solver,
                 snopt_opts=None, rec_file=None, check_partials=False, fix='alt'):
        """Initializes the `TwoDimDescTwoPhasesAnalyzer` class variables. """

        TwoDimAnalyzer.__init__(self, body, sc)

        self.alt = alt
        self.alt_p = alt_p
        self.alt_switch = alt_switch

        self.phase_name = ('free', 'vertical')

        dep = TwoDimOrb(body.GM, a=(body.R + alt), e=0.0)
        arr = TwoDimOrb(body.GM, a=(body.R + alt_p), e=0.0)

        self.ht = HohmannTransfer(body.GM, dep, arr)
        self.deorbit_burn = ImpulsiveBurn(sc, self.ht.dva)

        self.nlp = TwoDimDescTwoPhasesNLP(body, self.deorbit_burn.sc, alt_p, alt_switch, self.ht.transfer.vp, theta,
                                          (0.0, np.pi), tof, t_bounds, method, nb_seg, order, solver, self.phase_name,
                                          snopt_opts=snopt_opts, rec_file=rec_file, check_partials=check_partials,
                                          fix=fix)

    def get_time_series(self, p, scaled=False):
        """Access the time series of the simulation.

        Parameters
        ----------
        p : Problem
            Instance of `Problem` class
        scaled : bool
            Scales the simulation results

        Returns
        -------
        tof : float
            Time of flight resulting from the optimized simulation [s]
        t : ndarray
            Time of flight time series for the optimized simulation [s]
        states : ndarray
            States time series for the optimized simulation
        controls : ndarray
            Controls time series for the optimized simulation

        """

        # attitude free
        tof_free = float(p.get_val(self.nlp.phase_name[0] + '.t_duration')) * self.body.tc
        t_free = p.get_val(self.nlp.phase_name[0] + '.timeseries.time') * self.body.tc

        states_free = np.empty((np.size(t_free), 0))

        for k in states_2d:
            s = p.get_val(self.nlp.phase_name[0] + '.timeseries.states:' + k)
            states_free = np.append(states_free, s, axis=1)

        states_free = states_free * self.states_scalers

        alpha_free = p.get_val(self.nlp.phase_name[0] + '.timeseries.controls:alpha')
        controls_free = np.hstack((self.sc.T_max * np.ones((np.size(t_free), 1)), alpha_free))

        # vertical
        tof_vertical = float(p.get_val(self.nlp.phase_name[1] + '.t_duration')) * self.body.tc
        t_vertical = p.get_val(self.nlp.phase_name[1] + '.timeseries.time') * self.body.tc

        r_vertical = p.get_val(self.nlp.phase_name[1] + '.timeseries.states:r') * self.body.R
        theta_vertical = states_free[-1, 1] * np.ones((np.size(t_vertical), 1))
        u_vertical = p.get_val(self.nlp.phase_name[1] + '.timeseries.states:u') * self.body.vc
        m_vertical = p.get_val(self.nlp.phase_name[1] + '.timeseries.states:m')

        states_vertical = np.hstack((r_vertical, theta_vertical, u_vertical, np.zeros((np.size(t_vertical), 1)),
                                     m_vertical))

        controls_vertical = np.hstack((self.sc.T_max * np.ones((np.size(t_vertical), 1)),
                                       np.pi / 2 * np.ones((np.size(t_vertical), 1))))

        tof = [tof_free, tof_vertical]
        t = [t_free, t_vertical]
        states = [states_free, states_vertical]
        controls = [controls_free, controls_vertical]

        return tof, t, states, controls

    def __str__(self):
        """Prints info on the `TwoDimDescTwoPhasesAnalyzer`.

        Returns
        -------
        s : str
            Info on `TwoDimDescTwoPhasesAnalyzer`

        """

        lines = ['\n{:^50s}'.format('2D Descent Trajectory with vertical touch-down:'),
                 '\n{:<25s}{:>20.6f}{:>5s}'.format('Parking orbit altitude:', self.alt / 1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Periapsis altitude:', self.alt_p / 1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Switch altitude:', (self.states[0][-1, 0] - self.body.R)/1e3, 'km'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Deorbit burn fraction:', self.deorbit_burn.dm / self.sc.m0, ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Propellant fraction:', (1 - self.states[-1][-1, -1]/self.sc.m0), ''),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Time of flight:', sum(self.tof), 's'),
                 '{:<25s}{:>20.6f}{:>5s}'.format('Switch time:', self.tof[0], 's'),
                 TwoDimAnalyzer.__str__(self)]

        s = '\n'.join(lines)

        return s

    def plot(self):
        """Plots the states and controls resulting from the simulation and the ones from the explicit computation in
        time.

        """

        sol_plot = TwoDimTwoPhasesSolPlot(self.rm_res, self.time, self.states, self.controls,
                                          time_exp=self.time_exp, states_exp=self.states_exp, kind='descent')

        sol_plot.plot()
