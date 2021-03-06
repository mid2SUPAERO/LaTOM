"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt


class TwoDimAltProfile:
    """Plot the two-dimensional simulation's altitude profile over spawn angle.

    Parameters
    ----------
    r : float
        Equatorial radius of central attracting body [m] or [-]
    states : ndarray
        States time series for implicit NLP solution as `[r, theta, u, v, m]`
    states_exp : ndarray or ``None``, optional
        States time series for explicit simulation as `[r, theta, u, v, m]` or ``None``. Default is ``None``
    thrust : ndarray or ``None``, optional
        Thrust magnitude time series or ``None``. Default is ``None``
    threshold : float or ``None``, optional
        Threshold value to determine the on/off control structure or ``None``. Default is ``1e-6``
    r_safe : ndarray or ``None``, optional
        Time series for minimum safe altitude [m] or [-] or ``None``. Default is ``None``
    labels : iterable, optional
        Labels for the different phases. Default is `('powered', 'coast')`

    Attributes
    ----------
    R : float
        Equatorial radius of central attracting body [m] or [-]
    scaler : float
        Value to scale the distances
    units : str
        Measurement unit for distances
    r : ndarray
        Position time series for implicit NLP solution [m] or [-]
    theta : ndarray
        Angle time series for implicit NLP solution [rad]
    r_pow : ndarray
        Position time series for implicit NLP solution corresponding to powered phases [m] or [-]
    theta_pow : ndarray
        Angle time series for implicit NLP solution corresponding to powered phases [m] or [-]
    r_coast : ndarray
        Position time series for implicit NLP solution corresponding to coasting phases [m] or [-]
    theta_coast : ndarray
        Angle time series for implicit NLP solution corresponding to coasting phases [m] or [-]
    r_exp : ndarray
        Position time series for explicit simulation [m] or [-]
    theta_exp : ndarray
        Angle time series for explicit simulation [m] or [-]
    r_safe : ndarray or ``None``
        Time series for minimum safe altitude [m] or [-] or ``None``
    labels : iterable, optional
        Labels for the different phases

    """

    def __init__(self, r, states, states_exp=None, thrust=None, threshold=1e-6, r_safe=None,
                 labels=('powered', 'coast')):
        """Initializes `TwoDimAltProfile` class. """

        self.R = r

        if not np.isclose(r, 1.0):
            self.scaler = 1e3
            self.units = 'km'
        else:
            self.scaler = 1.0
            self.units = '-'

        self.r = states[:, 0]
        self.theta = states[:, 1]

        if thrust is not None:  # variable thrust

            states_pow = states[(thrust >= threshold).flatten(), :]
            states_coast = states[(thrust < threshold).flatten(), :]

            self.r_pow = states_pow[:, 0]
            self.theta_pow = states_pow[:, 1]
            self.r_coast = states_coast[:, 0]
            self.theta_coast = states_coast[:, 1]

        if states_exp is not None:

            self.r_exp = states_exp[:, 0]
            self.theta_exp = states_exp[:, 1]

        self.r_safe = r_safe
        self.labels = labels

    def plot(self):
        """Plot the two-dimensional simulation's altitude profile over spawn angle. """

        fig, ax = plt.subplots(1, 1, constrained_layout=True)

        if self.r_safe is not None:

            ax.plot(self.theta*180/np.pi, (self.r_safe - self.R)/self.scaler, color='k', label='safe altitude')

        if hasattr(self, 'r_exp'):  # explicit simulation

            ax.plot(self.theta_exp*180/np.pi, (self.r_exp - self.R)/self.scaler, color='g', label='explicit')

        if hasattr(self, 'r_pow'):  # implicit solution with variable thrust

            ax.plot(self.theta_coast * 180 / np.pi, (self.r_coast - self.R)/self.scaler, 'o', color='b',
                    label=self.labels[1])
            ax.plot(self.theta_pow*180/np.pi, (self.r_pow - self.R)/self.scaler, 'o', color='r', label=self.labels[0])

        else:  # implicit solution with constant thrust

            ax.plot(self.theta*180/np.pi, (self.r - self.R)/self.scaler, 'o', color='b', label='implicit')

        ax.set_ylabel(''.join(['h (', self.units, ')']))
        ax.set_xlabel('theta (deg)')
        ax.set_title('Altitude profile')
        ax.grid()
        ax.legend(loc='best')


class TwoDimTrajectory:
    """Plots the two-dimensional trajectories in the xy plane.

    Parameters
    ----------
    r_moon : float
       Moon radius [m] or [-]
    r_llo : float
       Initial Low Lunar Orbit radius [m] o [-]
    states : dict
       Dictionary that maps states values obtained from the simulation
    kind : str, optional
        Defines the kind of trajectory. It can be 'ascent' or 'descent'. Default is `ascent`
    nb : float, optional
       Number of points in which the Moon surface and the initial orbits are discretized. Default is ``2000``

    Attributes
    ----------
    scaler : float
       scaler for lengths
    units : str
       Unit of measurement for lengths
    x_moon : ndarray
       x coordinates for the Moon surface [km] or [-]
    y_moon : ndarray
       y coordinates for the Moon surface [km] or [-]
    x_llo : ndarray
       x coordinates for the initial orbit [km] or [-]
    y_llo : ndarray
       y coordinates for the initial orbit [km] or [-]
    x : dict
       x coordinates for the ascent trajectories [km] or [-]
    y : dict
       y coordinates for the ascent trajectories [km] or [-]
    kind : str
        Defines the kind of trajectory. It can be 'ascent' or 'descent'

    """

    def __init__(self, r_moon, r_llo, states, kind='ascent', nb=2000):
        """Initializes `TwoDimTrajectory` class. """

        self.scaler, self.units = self.get_scalers(r_moon)

        self.x_moon, self.y_moon = self.polar2cartesian(r_moon, self.scaler, nb=nb)  # Moon surface in xy plane
        self.x_llo, self.y_llo = self.polar2cartesian(r_llo, self.scaler, nb=nb)  # LLO in xy plane
        self.x, self.y = self.polar2cartesian(states[:, 0], scaler=self.scaler, angle=states[:, 1])  # Trajectory

        self.kind = kind

    @staticmethod
    def get_scalers(r):
        """Defines the scaling parameter for lengths and corresponding measurement unit.

        Parameters
        ----------
        r : float
            Moon radius in dimensional or non-dimensional units [m] or [-]

        Returns
        -------
        scaler : float
            Scaler value [-]
        units : str
            Measurement unit

        """

        if not np.isclose(r, 1.0):
            scaler = 1e3
            units = 'km'
        else:
            scaler = 1.0
            units = '-'

        return scaler, units

    @staticmethod
    def polar2cartesian(r, scaler=1., **kwargs):
        """Transforms the polar coordinates into cartesian coordinates.

        Parameters
        ----------
        r : ndarray
            Position series [km] or [-]
        scaler : float
            Scaler value [-]

        Other Parameters
        ----------------
        nb : int
            Number of equally spaced points between ``0`` and ``2*pi`` in which the `x` and `y` coordinates are computed
        angle : ndarray
            Angles for which the `x` and `y` coordinates are computed

        Returns
        -------
        x : ndarray
            X coordinates series [km] or [-]
        y : ndarray
            X coordinates series [km] or [-]

        """

        if 'nb' in kwargs:
            angle = np.linspace(0.0, 2 * np.pi, kwargs['nb'])
        elif 'angle' in kwargs:
            angle = kwargs['angle']
        else:
            raise ValueError('nb or angle array must be provided')

        x = r*np.cos(angle)/scaler
        y = r*np.sin(angle)/scaler

        return x, y

    @staticmethod
    def set_axes_decorators(ax, title, units):
        """Sets the plot axes decorators.

        Parameters
        ----------
        ax : Axes
            Instance of `Axes` class
        title : str
            Title of the plot
        units : str
            Measurement units

        """
        ax.set_aspect('equal')
        ax.grid()

        ax.tick_params(axis='x', rotation=60)

        ax.set_xlabel(''.join(['x (', units, ')']))
        ax.set_ylabel(''.join(['y (', units, ')']))
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1, 1), loc=2)

    def plot(self):
        """Plots the two-dimensional trajectories in the xy plane. """

        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(self.x_moon, self.y_moon, label='Moon surface')

        if hasattr(self, 'x_nrho') and hasattr(self, 'y_nrho'):
            ax.plot(self.x_llo, self.y_llo, label='departure orbit')
            ax.plot(self.x_nrho, self.y_nrho, label='target orbit')
        else:
            ax.plot(self.x_llo, self.y_llo, label='target orbit')

        label = ' '.join([self.kind, 'trajectory'])
        ax.plot(self.x, self.y, label=label)

        self.set_axes_decorators(ax, ' '.join(['Optimal', label]), self.units)


class TwoDimSurface2LLO(TwoDimTrajectory):
    """Plots the two-dimensional trajectories from the Moon surface to a Low Lunar Orbit and vice versa.

    Parameters
    ----------
    r_moon : float
       Moon radius [m] or [-]
    states : dict
       Dictionary that maps states values obtained from the simulation
    kind : str, optional
        Defines the kind of trajectory. It can be 'ascent' or 'descent'. Default is `ascent`
    nb : float, optional
       Number of points in which the Moon surface and the initial orbits are discretized. Default is ``2000``

    """

    def __init__(self, r_moon, states, kind='ascent', nb=2000):
        """Initializes `TwoDimSurface2LLO` class. """

        if kind == 'ascent':
            r_llo = states[-1, 0]
        elif kind == 'descent':
            r_llo = states[0, 0]
        else:
            raise ValueError('kind must be either ascent or descent')

        TwoDimTrajectory.__init__(self, r_moon, r_llo, states, kind=kind, nb=nb)


class TwoDimLLO2NRHO(TwoDimTrajectory):
    """Plots the two-dimensional trajectories from a Low Lunar Orbit to a Near rectilinear Halo Orbit.

    Parameters
    ----------
    r_moon : float
       Moon radius [m] or [-]
    a_nrho : float
        Semi-major axis of the NRHO [m] or [-]
    e_nrho : float
        Eccentricity of the NRHO [-]
    states : dict
       Dictionary that maps states values obtained from the simulation
    kind : str, optional
        Defines the kind of trajectory. It can be 'ascent' or 'descent'. Default is `ascent`
    nb : float, optional
       Number of points in which the Moon surface and the initial orbits are discretized. Default is ``2000``

    Attributes
    ----------
    x_nrho : ndarray
        X coordinate series for the NRHO [km] or [-]
    y_nrho : ndarray
            Y coordinate series for the NRHO [km] or [-]

    """
    def __init__(self, r_moon, a_nrho, e_nrho, states, kind='ascent', nb=2000):
        """Initializes `TwoDimLLO2NRHO` class. """

        # kind
        if kind == 'ascent':
            r_llo = states[0, 0]
        elif kind == 'descent':
            r_llo = states[-1, 0]
        else:
            raise ValueError('kind must be either ascent or descent')

        TwoDimTrajectory.__init__(self, r_moon, r_llo, states, kind=kind, nb=nb)

        # NRHO
        angle = np.linspace(0.0, 2 * np.pi, nb)
        r_nrho = a_nrho*(1 - e_nrho**2)/(1 + e_nrho*np.cos(angle))
        self.x_nrho = r_nrho/self.scaler*np.cos(angle)
        self.y_nrho = r_nrho/self.scaler*np.sin(angle)
