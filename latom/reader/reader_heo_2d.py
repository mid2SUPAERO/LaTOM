"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np

from latom.reader.reader import Reader
from latom.plots.solutions import TwoDimMultiPhaseSolPlot
from latom.utils.keplerian_orbit import TwoDimOrb
from latom.utils.const import states_2d


class TwoDim3PhasesLLO2HEOReader(Reader):
    """`TwoDim3PhasesLLO2HEOReader` class loads and displays a stored simulation corresponding to a three-phases
    transfer trajectory from LLO to HEO using an OpenMDAO `CaseReader` class instance.

    Parameters
    ----------
    body : Primary
        Central attracting body
    db : str
        Full path of the database where the solution is stored
    case_id : str, optional
        Case identifier, ``initial`` to load the first iteration, ``final`` to load the final solution.
        Default is ``final``
    db_exp : str or ``None``, optional
        Full path of the database where the explicit simulation is stored or ``None``. Default is ``None``
    scaled : bool, optional
        ``False`` to retrieve the solution in dimensional units, ``True`` otherwise. Default is ``False``

    Attributes
    ----------
    body : Primary
        Central attracting body
    gm_res : float
        Central attracting body standard gravitational parameter [m^3/s^2] or [-]
    rm_res : float
        Central attracting body equatorial radius [m] or [-]
    states_scalers : ndarray
        Scaling parameters for distances, angles, velocities and mass
    phase_name : list
        List of phases names within the OpenMDAO `Problem` object
    tof : float
        Time of flight for the implicit solution [s]
    time : list
        Time vector for the implicit solution [s]
    states : list
        States time series for the implicit solution as `[r, theta, u, v, m]`
    controls : list
        Controls variables time series for the implicit solution as `[thrust, alpha]`
    tof_exp : float or ``None``
        Time of flight for the explicit simulation [s] or ``None``
    time_exp : list or ``None``
        Time vector for the explicit simulation [s] or ``None``
    states_exp : list or ``None``
        States time series for the explicit simulation as `[r, theta, u, v, m]` or ``None``
    controls_exp : list or ``None``
        Controls variables time series for the explicit simulation as `[thrust, alpha]` or ``None``
    coe_inj : iterable
        Classical orbital elements at injection as `(a, e, h, ta)` with `a` semi-major axis, `e` eccentricity, `h`
        specific angular momentum vector and `ta` true anomaly

    """

    def __init__(self, body, db, case_id='final', db_exp=None, scaled=False):

        Reader.__init__(self, db, case_id=case_id, db_exp=db_exp)

        self.body = body

        if scaled:
            self.gm_res = 1.0
            self.rm_res = 1.0
            self.states_scalers = None
        else:
            self.gm_res = body.GM
            self.rm_res = body.R
            self.states_scalers = np.array([body.R, 1.0, body.vc, body.vc, 1.0])

        self.phase_name = []

        for s in ['dep', 'coast', 'arr']:
            ph_name = '.'.join(['traj', s, 'timeseries'])
            self.phase_name.append(ph_name)

        self.tof, self.time, self.states, self.controls = self.get_time_series(self.case)

        if db_exp is not None:
            self.tof_exp, self.time_exp, self.states_exp, self.controls_exp = self.get_time_series(self.case_exp)
        else:
            self.tof_exp = self.time_exp = self.states_exp = self.controls_exp = None

        self.coe_inj = TwoDimOrb.polar2coe(self.gm_res, self.states[-1][-1, 0], self.states[-1][-1, 2],
                                           self.states[-1][-1, 3])

    def get_time_series(self, case):
        """Retrieve the time of flight, time vector and states, controls and minimum safe altitude time series for
        the specified `case` and `kind` of transfer.

        Parameters
        ----------
        case : Case
            OpenMDAO `Case` object

        Returns
        -------
        tof : float
            Time of flight for the given `Case` [s]
        time : list
            Time vector for the given `Case` [s]
        states : list
            States time series for the given `Case` as `[r, theta, u, v, m]`
        controls : list
            Controls variables time series for the given `Case` as `[thrust, alpha]`

        """

        # initialization
        time = []
        tof = []
        states = []
        controls = []

        # loop over three phases
        for i in range(3):

            # non-dimensional time vector
            t = case.outputs.get(self.phase_name[i] + '.time')

            # non-dimensional states
            s = np.empty((np.size(t), 0))

            for k in states_2d:
                sk = case.outputs.get(self.phase_name[i] + '.states:' + k)
                s = np.append(s, sk, axis=1)

            # non-dimensional controls
            alpha = case.outputs.get(self.phase_name[i] + '.controls:alpha')
            thrust = case.outputs.get(self.phase_name[i] + '.design_parameters:thrust')

            # dimensional states and controls
            if self.states_scalers is not None:
                t = t*self.body.tc
                s = s*self.states_scalers
                thrust = thrust*self.body.g*s[0, -1]

            c = np.hstack((thrust, alpha))

            time.append(t)
            tof.append(t[-1] - t[0])
            states.append(s)
            controls.append(c)

        return tof, time, states, controls

    def plot(self):
        """Plot the optimal transfer trajectory corresponding to the loaded `Case`. """

        dtheta = self.coe_inj[-1] - self.states[-1][-1, 1]

        sol_plot = TwoDimMultiPhaseSolPlot(self.rm_res, self.time, self.states, self.controls, self.time_exp,
                                           self.states_exp, a=self.coe_inj[0], e=self.coe_inj[1], dtheta=dtheta)
        sol_plot.plot()
