"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy


class RespSurf:
    """ Plots the response surface resulting from the surrogate models computations

    Parameters
    ----------
    isp : ndarray
        List of isp values [s]
    twr : ndarray
        List of thrust/weight ratios [-]
    param : ndarray
        List of parameters
    title : str
        Title of the plot
    nb_lines : int
        Numbers of lines of response surfaces
    log_scale : bool
        Defines if the twr scale is logaritmic

    Attributes
    ----------
    isp : ndarray
        List of isp values [s]
    twr : ndarray
        List of thrust/weight ratios [-]
    param : ndarray
        List of parameters
    title : str
        Title of the plot
    nb_lines : int
        Numbers of lines of response surfaces
    """

    def __init__(self, isp, twr, param, title, nb_lines=50, log_scale=False):
        """Initializes `RespSurf` class. """
        if log_scale:
            twr = np.exp(twr)
        [self.twr, self.isp] = np.meshgrid(twr, isp)
        self.param = deepcopy(param)
        self.title = title
        self.nb_lines = nb_lines

    def plot(self):
        """  Plots the response surface resulting from the surrogate models computations """

        fig, ax = plt.subplots()
        cs = ax.contour(self.twr, self.isp, self.param, self.nb_lines)
        ax.clabel(cs)
        ax.set_xlabel('Thrust/initial weight ratio')
        ax.set_ylabel('Isp (s)')
        ax.set_title(self.title)
