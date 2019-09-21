"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy


class RespSurf:

    def __init__(self, x, m, tof):

        self.Isp = np.unique(x[:, 0])
        self.twr = np.unique(x[:, 1])
        self.m = deepcopy(m)
        self.tof = deepcopy(tof)

    def plot(self):

        [twr, isp] = np.meshgrid(self.twr, self.Isp, indexing='ij')

        fig, ax = plt.subplots()
        cs = ax.contour(twr, isp, self.m, 100)
        ax.clabel(cs)
        ax.set_xlabel('Thrust/initial weight ratio')
        ax.set_ylabel('Isp (s)')
        ax.set_title('Final/initial mass ratio')

        fig, ax = plt.subplots()
        cs = ax.contour(twr, isp, self.tof, 100)
        ax.clabel(cs)
        ax.set_xlabel('Thrust/initial weight ratio')
        ax.set_ylabel('Isp (s)')
        ax.set_title('Time of flight')
