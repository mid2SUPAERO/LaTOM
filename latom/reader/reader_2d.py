"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.reader.reader import Reader
from latom.utils.const import states_2d
from latom.plots.solutions import TwoDimSolPlot


class TwoDimReader(Reader):
    """`TwoDimReader` class loads and displays a stored simulation corresponding to a single-phase, two-dimensional
    transfer trajectory using an OpenMDAO `CaseReader` class instance.

    Parameters
    ----------
    kind : iterable
        List of three parameters to define the characteristics of the solution to be loaded in the form
        ``ascent/descent``, ``const/variable`` and ``True/False`` where the last corresponds to the presence of a path
        constraint to impose a minimum safe altitude on the simulated transfer
    body : Primary
        Central attracting body
    db : str
        Full path of the database where the solution is stored
    case_id : str, optional
        Case identifier, ``initial`` to load the first iteration, ``final`` to load the final solution.
        Default is ``final``
    db_exp : str or ``None``, optional
        Full path of the database where the explicit simulation is stored or ``None``. Default is ``None``

    Attributes
    ----------
    kind : iterable
        List of three parameters to define the characteristics of the solution to be loaded in the form
        ``ascent/descent``, ``const/variable`` and ``True/False`` where the last corresponds to the presence of a path
        constraint to impose a minimum safe altitude on the simulated transfer
    body : Primary
        Central attracting body
    phase_name : str
        Name of the simulated phase within the OpenMDAO `Problem` class instance
    states_scalers : ndarray
        Scaling parameters for distances, angles, velocities and mass
    tof : float
        Time of flight for the implicit solution [s]
    time : ndarray
        Time vector for the implicit solution [s]
    states : ndarray
        States time series for the implicit solution as `[r, theta, u, v, m]`
    controls : ndarray
        Controls variables time series for the implicit solution as `[thrust, alpha]`
    r_safe : ndarray or ``None``
        Minimum altitude constraint time series for the implicit solution [m] or ``None``
    tof_exp : float or ``None``
        Time of flight for the explicit simulation [s] or ``None``
    time_exp : ndarray or ``None``
        Time vector for the explicit simulation [s] or ``None``
    states_exp : ndarray or ``None``
        States time series for the explicit simulation as `[r, theta, u, v, m]` or ``None``
    controls_exp : ndarray or ``None``
        Controls variables time series for the explicit simulation as `[thrust, alpha]` or ``None``
    r_safe_exp : ndarray or ``None``
        Minimum altitude constraint time series for the explicit simulation [m] or ``None``

    """

    def __init__(self, kind, body, db, case_id='final', db_exp=None):

        Reader.__init__(self, db, case_id, db_exp=db_exp)

        self.kind = kind  # ('ascent/descent', 'const/variable', 'True/False')
        self.body = body
        self.phase_name = 'traj.powered.timeseries'
        self.states_scalers = np.array([self.body.R, 1.0, self.body.vc, self.body.vc, 1.0])

        # implicit NLP solution
        self.tof, self.time, self.states, self.controls, self.r_safe =\
            self.get_time_series(self.case, kind)

        # explicit simulation
        if db_exp is not None:
            self.tof_exp, self.time_exp, self.states_exp, self.controls_exp, self.r_safe_exp =\
                self.get_time_series(self.case_exp, kind)
        else:
            self.tof_exp = self.time_exp = self.states_exp = self.controls_exp = self.r_safe_exp = None

    def get_time_series(self, case, kind):
        """Retrieve the time of flight, time vector and states, controls and minimum safe altitude time series for
        the specified `case` and `kind` of transfer.

        Parameters
        ----------
        case : Case
            OpenMDAO `Case` object
        kind : iterable
            List of three parameters to define the characteristics of the solution to be loaded in the form
            ``ascent/descent``, ``const/variable`` and ``True/False`` where the last corresponds to the presence of a
            path constraint to impose a minimum safe altitude on the simulated transfer

        Returns
        -------
        tof : float
            Time of flight for the given `Case` [s]
        time : ndarray
            Time vector for the given `Case` [s]
        states : ndarray
            States time series for the given `Case` as `[r, theta, u, v, m]`
        controls : ndarray
            Controls variables time series for the given `Case` as `[thrust, alpha]`
        r_safe : ndarray or ``None``
            Minimum altitude constraint time series for the given `Case` [m] or ``None``

        """

        # dimensional time vector and time of flight
        time = case.outputs.get(self.phase_name + '.time')*self.body.tc
        tof = time[-1] - time[0]

        # non-dimensional states
        states = np.empty((np.size(time), 0))

        for k in states_2d:
            s = case.outputs.get(self.phase_name + '.states:' + k)
            states = np.append(states, s, axis=1)

        # dimensional states
        states = states*self.states_scalers

        # non-dimensional controls
        alpha = case.outputs.get(self.phase_name + '.controls:alpha')

        if kind[1] == 'const':
            thrust = case.outputs.get(self.phase_name + '.design_parameters:thrust')
        elif kind[1] == 'variable':
            thrust = case.outputs.get(self.phase_name + '.controls:thrust')
        else:
            raise ValueError('the second element of kind must be const or variable')

        # dimensional controls
        controls = np.hstack((thrust*self.body.g*states[0, -1], alpha))

        # dimensional minimum safe altitude
        if kind[2]:
            r_safe = case.outputs.get(self.phase_name + '.r_safe')*self.body.R
        else:
            r_safe = None

        return tof, time, states, controls, r_safe

    def plot(self):
        """Plot the optimal transfer trajectory corresponding to the loaded `Case`. """

        if self.kind[1] == 'const':
            threshold = None
        else:
            threshold = 1e-6

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 self.r_safe, threshold, self.kind[0])
        sol_plot.plot()
