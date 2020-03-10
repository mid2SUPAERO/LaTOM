"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy


class RespSurf:

    def __init__(self, isp, twr, param, title, nb_lines=50, log_scale=False):

        if log_scale:
            twr = np.exp(twr)
        [self.twr, self.isp] = np.meshgrid(twr, isp)
        self.param = deepcopy(param)
        self.title = title
        self.nb_lines = nb_lines

    def plot(self):

        fig, ax = plt.subplots()
        cs = ax.contour(self.twr, self.isp, self.param, self.nb_lines)
        ax.clabel(cs)
        ax.set_xlabel('Thrust/initial weight ratio')
        ax.set_ylabel('Isp (s)')
        ax.set_title(self.title)
