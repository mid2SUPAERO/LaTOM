"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

import numpy as np
from latom.utils.keplerian_orbit import KepOrb, TwoDimOrb
from latom.utils.coc import per2eq_vec
from latom.guess.guess_2d import TwoDimDescGuess, TwoDimAscGuess, TwoDimLLOGuess


class ThreeDimAscGuess(TwoDimLLOGuess):

    def __init__(self, gm, r, alt, sc, raan, i, w):

        self.TwoDimAscGuess = TwoDimAscGuess(gm, r, alt, sc)

        self.R, self.V = per2eq_vec(self.TwoDimAscGuess.states[:0], self.TwoDimAscGuess.states[:1],
                                    self.TwoDimAscGuess.states[:2], self.TwoDimAscGuess.states[:3], raan, i, w)
        self.u = self.V / np.linalg.norm(self.V, axis=1, keepdims=True)

        'controls'
        self.controls = np.hstack(self.TwoDimAscGuess.controls[0], self.u)


class ThreeDimDescGuess(TwoDimLLOGuess):

    def __init__(self, gm, r, alt, sc, raan, i, w):

        self.TwoDimDescGuess = TwoDimDescGuess(gm, r, alt, sc)

        self.R, self.V = per2eq_vec(self.TwoDimDescGuess.states[:0], self.TwoDimDescGuess.states[:1],
                                self.TwoDimDescGuess.states[:2], self.TwoDimDescGuess.states[:3], raan, i, w)
        self.u = self.V / np.linalg.norm(self.V, axis=1, keepdims=True)

        'controls'
        self.controls = np.hstack(self.TwoDimDescGuess.controls[0], self.u)




