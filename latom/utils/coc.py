"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np


def rot1(t):
    """Elementary rotation matrix around X axis.

    Parameters
    ----------
    t : float
        Rotation angle [rad]

    Returns
    -------
    r : ndarray
        3x3 rotation matrix around X axis

    """

    r = np.array([[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]])

    return r


def rot3(t):
    """Elementary rotation matrix around Z axis.

    Parameters
    ----------
    t : float
        Rotation angle [rad]

    Returns
    -------
    r : ndarray
        3x3 rotation matrix around Z axis

    """

    r = np.array([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]])

    return r


def eq2per(raan, i, w):
    """Rotation matrix from inertial, body-centred equatorial reference frame to perifocal reference frame.

    Parameters
    ----------
    raan : float
        Right Ascension of the Ascending Node [rad]
    i : float
        Inclination [rad]
    w : float
        Argument of Periapsis [rad]

    Returns
    -------
    q : ndarray
        3x3 rotation matrix from equatorial to perifocal reference frames

    """

    q = rot3(w) @ rot1(i) @ rot3(raan)

    return q


def per2eq(raan, i, w):
    """Rotation matrix from perifocal reference frame to inertial, body-centred equatorial reference frame.

    Parameters
    ----------
    raan : float
        Right Ascension of the Ascending Node [rad]
    i : float
        Inclination [rad]
    w : float
        Argument of Periapsis [rad]

    Returns
    -------
    q : ndarray
        3x3 rotation matrix from perifocal to equatorial reference frames

    """

    q = np.matrix.transpose(eq2per(raan, i, w))

    return q


def coe2sv_vec(a, e, i, raan, w, ta, gm):
    """Change of Coordinates from Classical Orbital Elements to State Vector in body-centred inertial reference frame.

    Parameters
    ----------
    a : float
        Semi-major axis [m]
    e : float
        Eccentricity [-]
    i : float
        Inclination [rad]
    raan : float
        Right Ascension of the Ascending Node [rad]
    w : float
        Argument of Periapsis [rad]
    ta : ndarray
        True anomalies [rad]
    gm : float
        Central body standard gravitational parameter [m^3/s^2]

    Returns
    -------
    r_vec : ndarray
        Position vector [m]
    v_vec : ndarray
        Velocity vector [m/s]

    """

    h = (gm * a * (1 - e ** 2)) ** 0.5  # specific angular momentum magnitude [km^2/s]
    r = (h ** 2 / gm) / (1 + e * np.cos(ta))  # distance from the central body [km]

    xp = r * np.cos(ta)  # position along x axis in perifocal reference frame [km]
    yp = r * np.sin(ta)  # position along y axis in perifocal reference frame [km]

    vxp = - (gm / h) * np.sin(ta)  # velocity along x axis in perifocal reference frame [km/s]
    vyp = (gm / h) * (e + np.cos(ta))  # velocity along y axis in perifocal reference frame [km/s]

    # reshape the position and velocity vectors in perifocal reference frame
    n = len(xp)

    xp = np.reshape(xp, (n, 1))
    yp = np.reshape(yp, (n, 1))

    vxp = np.reshape(vxp, (n, 1))
    vyp = np.reshape(vyp, (n, 1))

    r_vec, v_vec = per2eq_vec(xp, yp, vxp, vyp, raan, i, w)  # conversion from perifocal to equatorial reference frame

    return r_vec, v_vec


def polar2per_vec(r, theta, u, v):
    """Transformation from polar coordinates to perifocal reference frame.

    Parameters
    ----------
    r : ndarray
        Radius time series [m]
    theta : ndarray
        Angle time series [rad]
    u : ndarray
        Radial velocity time series [m/s]
    v : ndarray
        Tangential velocity time series [m/s]

    Returns
    -------
    xp : ndarray
        Position along x axis [m]
    yp : ndarray
        Position along y axis [m]
    vxp : ndarray
        Velocity component along x axis [m/s]
    vyp : ndarray
        Velocity component along y axis [m/s]

    """

    xp = r * np.cos(theta)
    yp = r * np.sin(theta)

    vxp = u * np.cos(theta) - v * np.sin(theta)
    vyp = u * np.sin(theta) + v * np.cos(theta)

    return xp, yp, vxp, vyp


def per2eq_vec(xp, yp, vxp, vyp, raan, i, w):
    """Transformation from perifocal reference frame to inertial, body-centred equatorial reference frame.

    Parameters
    ----------
    xp : ndarray
        Position along x axis [m]
    yp : ndarray
        Position along y axis [m]
    vxp : ndarray
        Velocity component along x axis [m/s]
    vyp : ndarray
        Velocity component along y axis [m/s]
    raan : float
        Right Ascension of the Ascending Node [rad]
    i : float
        Inclination [rad]
    w : float
        Argument of Periapsis [rad]

    Returns
    -------
    r_vec : ndarray
        Position vector [m]
    v_vec : ndarray
        Velocity vector [m/s]

    """

    # compute sin and cos of right ascension of the ascending node, inclination and argument of periapsis
    c_raan = np.cos(raan)
    s_raan = np.sin(raan)

    ci = np.cos(i)
    si = np.sin(i)

    cw = np.cos(w)
    sw = np.sin(w)

    # compute coefficient of the rotation matrix
    q11 = c_raan * cw - s_raan * ci * sw
    q12 = -c_raan * sw - s_raan * ci * cw
    q21 = s_raan * cw + c_raan * ci * sw
    q22 = c_raan * ci * cw - s_raan * sw
    q31 = si * sw
    q32 = cw * si

    # compute the position and velocity vectors in equatorial reference frame
    x = q11 * xp + q12 * yp
    y = q21 * xp + q22 * yp
    z = q31 * xp + q32 * yp

    vx = q11 * vxp + q12 * vyp
    vy = q21 * vxp + q22 * vyp
    vz = q31 * vxp + q32 * vyp

    # stack together the x, y, z components of the position and velocity vectors
    r_vec = np.hstack((x, y, z))
    v_vec = np.hstack((vx, vy, vz))

    return r_vec, v_vec


def polar2eq_vec(r, theta, u, v, raan, i, w):
    """Transformation from polar coordinates to inertial, body-centred equatorial reference frame.

    Parameters
    ----------
    r : nd array
        Radius time series [m]
    theta : ndarray
        Angle time series [rad]
    u : nd array
        Radial velocity time series [m/s]
    v : ndarray
        Tangential velocity time series [m/s]
    raan : float
        Right Ascension of the Ascending Node [rad]
    i : float
        Inclination [rad]
    w : float
        Argument of Periapsis [rad]

    Returns
    -------
    r_vec : ndarray
        Position vector [m]
    v_vec : ndarray
        Velocity vector [m/s]

    """

    xp, yp, vxp, vyp = polar2per_vec(r, theta, u, v)
    r_vec, v_vec = per2eq_vec(xp, yp, vxp, vyp, raan, i, w)

    return r_vec, v_vec


def lat_long2cartesian(lat, long, r):
    """Transformation from latitude, longitude to cartesian coordinates.

    Parameters
    ----------
    lat : float
        Latitude [deg]
    long : float
        Longitude [deg]
    r : float
        Planet radius [m]

    """

    lat = lat*np.pi/180
    long = long*np.pi/180

    x = r*np.cos(lat)*np.cos(long)
    y = r*np.cos(lat)*np.sin(long)
    z = r*np.sin(lat)

    return x, y, z


if __name__ == '__main__':

    lat = 90.
    long = 0.

    r = 1.

    x, y, z = lat_long2cartesian(lat, long, r)

    for i in x, y, z:
        print(i)