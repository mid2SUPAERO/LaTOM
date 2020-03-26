"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from latom.plots.timeseries import TwoDimStatesTimeSeries, TwoDimControlsTimeSeries
from latom.plots.trajectories import TwoDimAltProfile, TwoDimSurface2LLO, TwoDimLLO2NRHO


class TwoDimSolPlot:
    """Plot the two-dimensional simulation's states and controls in time and in the xy plane.

    Parameters
    ----------
    r : float
        Equatorial radius of central attracting body [m] or [-]
    time : ndarray
        Time vector for implicit NLP solution [s] or [-]
    states : ndarray
        States time series for implicit NLP solution as `[r, theta, u, v, m]`
    controls : ndarray
        Controls time series for implicit NLP solution as `[thrust, alpha]`
    time_exp : ndarray or ``None``, optional
        Time vector for explicit simulation [s] o [-] or ``None``. Default is ``None``
    states_exp : ndarray or ``None``, optional
        States time series for explicit simulation as `[r, theta, u, v, m]` or ``None``. Default is ``None``
    r_safe : ndarray or ``None``, optional
        Time series for minimum safe altitude [m] or [-] or ``None``. Default is ``None``
    threshold : float or ``None``, optional
        Threshold value to determine the on/off control structure or ``None``. Default is ``1e-6``
    kind : str, optional
        Defines the kind of trajectory. The possible values are `ascent` or `descent`. Default is `ascent`
    a : float or ``None``, optional
        HEO orbit's semi-major axis [m] or [-] or ``None`` for surface to LLO transfers. Default is ``None``
    e : float or ``None``, optional
        HEO orbit's eccentricity [-] or ``None`` for surface to LLO transfers. Default is ``None``

    Attributes
    ----------
    R : float
        Equatorial radius of central attracting body [m] or [-]
    time : ndarray
        Time vector for implicit NLP solution [s] or [-]
    states : ndarray
        States time series for implicit NLP solution as `[r, theta, u, v, m]`
    controls : ndarray
        Controls time series for implicit NLP solution as `[thrust, alpha]`
    time_exp : ndarray or ``None``
        Time vector for explicit simulation [s] o [-] or ``None``
    states_exp : ndarray or ``None``
        States time series for explicit simulation as `[r, theta, u, v, m]` or ``None``
    r_safe : ndarray or ``None``
        Time series for minimum safe altitude [m] or [-] or ``None``
    threshold : float or ``None``
        Threshold value to determine the on/off control structure or ``None``
    kind : str
        Defines the kind of trajectory. The possible values are `ascent` or `descent`
    states_plot : TwoDimStatesTimeSeries
        Instance of `TwoDimStatesTimeSeries` class to display the states variables as function of time
    alt_plot : TwoDimAltProfile
        Instance of `TwoDimAltProfile` class to display the altitude over spawn angle
    controls_plot : TwoDimControlsTimeSeries
        Instance of `TwoDimControlsTimeSeries` class to display the controls variables as function of time
    trajectory_plot : TwoDimLLO2NRHO or TwoDimSurface2LLO
        Instance of `TwoDimLLO2NRHO` or `TwoDimSurface2LLO` class to create a LLO to NRHO trajectory plot or a Surface
        to Moon trajectory plot

    """

    def __init__(self, r, time, states, controls, time_exp=None, states_exp=None, r_safe=None,
                 threshold=1e-6, kind='ascent', a=None, e=None):
        """Initializes `TwoDimSolPlot` class. """

        self.R = deepcopy(r)

        self.time = deepcopy(time)
        self.states = deepcopy(states)
        self.controls = deepcopy(controls)

        self.time_exp = deepcopy(time_exp)
        self.states_exp = deepcopy(states_exp)

        self.r_safe = r_safe
        self.threshold = threshold

        if threshold is None:
            thrust = None
        else:
            thrust = self.controls[:, 0]

        if kind in ['ascent', 'descent']:
            self.kind = kind
        else:
            raise ValueError('kind must be either ascent or descent')

        self.states_plot = TwoDimStatesTimeSeries(self.R, self.time, self.states, self.time_exp, self.states_exp,
                                                  thrust=thrust, threshold=threshold, r_safe=self.r_safe)
        self.alt_plot = TwoDimAltProfile(self.R, self.states, self.states_exp, thrust=thrust, threshold=threshold,
                                         r_safe=self.r_safe)

        if not np.isclose(r, 1.0):
            self.controls_plot = TwoDimControlsTimeSeries(self.time, self.controls, threshold=threshold)
        else:
            self.controls_plot = TwoDimControlsTimeSeries(self.time, self.controls, threshold=threshold,
                                                          units=('-', '-'))

        if (a is not None) and (e is not None):
            self.trajectory_plot = TwoDimLLO2NRHO(self.R, a, e, self.states, self.kind)
        else:
            self.trajectory_plot = TwoDimSurface2LLO(self.R, self.states, self.kind)

    def plot(self):
        """Plots the two-dimensional simulation's states and controls in time and in the xy plane. """

        self.states_plot.plot()
        self.controls_plot.plot()
        self.alt_plot.plot()
        self.trajectory_plot.plot()

        plt.show()


class TwoDimMultiPhaseSolPlot(TwoDimSolPlot):
    """Plots the two-dimensional multi phase simulation's states and controls in time and in the xy plane.

    Parameters
    ----------
    r : float
        Equatorial radius of central attracting body [m] or [-]
    time : ndarray
        Time vector for implicit NLP solution [s] or [-]
    states : ndarray
        States time series for implicit NLP solution as `[r, theta, u, v, m]`
    controls : ndarray
        Controls time series for implicit NLP solution as `[thrust, alpha]`
    time_exp : ndarray or ``None``, optional
        Time vector for explicit simulation [s] o [-] or ``None``. Default is ``None``
    states_exp : ndarray or ``None``, optional
        States time series for explicit simulation as `[r, theta, u, v, m]` or ``None``. Default is ``None``
    r_safe : ndarray or ``None``, optional
        Time series for minimum safe altitude [m] or [-] or ``None``. Default is ``None``
    threshold : float or ``None``, optional
        Threshold value to determine the on/off control structure or ``None``. Default is ``1e-6``
    kind : str, optional
        Defines the kind of trajectory. The possible values are `ascent` or `descent`. Default is `ascent`
    a : float or ``None``, optional
        HEO orbit's semi-major axis [m] or [-] or ``None`` for surface to LLO transfers. Default is ``None``
    e : float or ``None``, optional
        HEO orbit's eccentricity [-] or ``None`` for surface to LLO transfers. Default is ``None``
    dtheta : float or ``None``, optional
        Angle to translate the position angle time series [rad]

    """

    def __init__(self, r, time, states, controls, time_exp=None, states_exp=None, r_safe=None, threshold=1e-6,
                 kind='ascent', a=None, e=None, dtheta=None):
        """Initializes `TwoDimMultiPhaseSolPlot` class. """

        time = np.vstack(time)
        states = np.vstack(states)
        controls = np.vstack(controls)

        if (time_exp is not None) and (states_exp is not None):
            time_exp = np.vstack(time_exp)
            states_exp = np.vstack(states_exp)

        if dtheta is not None:
            states[:, 1] = states[:, 1] + dtheta
            if states_exp is not None:
                states_exp[:, 1] = states_exp[:, 1] + dtheta

        TwoDimSolPlot.__init__(self, r, time, states, controls, time_exp=time_exp, states_exp=states_exp,
                               r_safe=r_safe, threshold=threshold, kind=kind, a=a, e=e)


class TwoDimDescTwoPhasesSolPlot:
    """Plot the two-dimensional two-phases descent simulation's states and controls in time and in the xy plane.

    Parameters
    ----------
    r : float
        Equatorial radius of central attracting body [m] or [-]
    time : ndarray
        Time vector for implicit NLP solution [s] or [-]
    states : ndarray
        States time series for implicit NLP solution as `[r, theta, u, v, m]`
    controls : ndarray
        Controls time series for implicit NLP solution as `[thrust, alpha]`
    time_exp : ndarray or ``None``, optional
        Time vector for explicit simulation [s] o [-] or ``None``. Default is ``None``
    states_exp : ndarray or ``None``, optional
        States time series for explicit simulation as `[r, theta, u, v, m]` or ``None``. Default is ``None``

    Attributes
    ----------
    R : float
        Equatorial radius of central attracting body [m] or [-]
    time : ndarray
        Time vector for implicit NLP solution [s] or [-]
    states : ndarray
        States time series for implicit NLP solution as `[r, theta, u, v, m]`
    controls : ndarray
        Controls time series for implicit NLP solution as `[thrust, alpha]`
    time_exp : ndarray or ``None``
        Time vector for explicit simulation [s] o [-] or ``None``
    states_exp : ndarray or ``None``
        States time series for explicit simulation as `[r, theta, u, v, m]` or ``None``
    states_plot : TwoDimStatesTimeSeries
        Instance of `TwoDimStatesTimeSeries` class to display the states variables as function of time
    alt_plot : TwoDimAltProfile
        Instance of `TwoDimAltProfile` class to display the altitude over spawn angle
    controls_plot : TwoDimControlsTimeSeries
        Instance of `TwoDimControlsTimeSeries` class to display the controls variables as function of time
    trajectory_plot : TwoDimSurface2LLO
        Instance of `TwoDimSurface2LLO` class to create a Surface to Moon trajectory plot

    """

    def __init__(self, r, time, states, controls, time_exp=None, states_exp=None):
        """Initializes `TwoDimDescTwoPhasesSolPlot` class. """

        self.R = deepcopy(r)

        self.time = np.vstack(time)
        self.states = np.vstack(states)
        self.controls = np.vstack(controls)

        if time_exp is not None:
            self.time_exp = np.vstack(time_exp)
            self.states_exp = np.vstack(states_exp)
        else:
            self.time_exp = self.states_exp = None

        n0 = np.size(time[0])
        n1 = np.size(time[1])

        thrust = np.vstack((np.zeros((n0, 1)), np.reshape(controls[1][:, 0], (n1, 1))))

        self.states_plot = TwoDimStatesTimeSeries(self.R, self.time, self.states, self.time_exp, self.states_exp,
                                                  thrust=thrust, labels=('vertical', 'attitude-free'))
        self.controls_plot = TwoDimControlsTimeSeries(self.time, self.controls, threshold=None)
        self.alt_plot = TwoDimAltProfile(self.R, self.states, self.states_exp, thrust=thrust,
                                         labels=('vertical', 'attitude-free'))
        self.trajectory_plot = TwoDimSurface2LLO(self.R, self.states, 'descent')

    def plot(self):
        """Plot the two-dimensional two-phases descent simulation's states and controls in time and in the xy plane. """

        self.states_plot.plot()
        self.controls_plot.plot()
        self.alt_plot.plot()
        self.trajectory_plot.plot()

        plt.show()
