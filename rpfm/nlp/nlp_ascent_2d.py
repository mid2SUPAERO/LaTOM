"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.nlp.nlp import NLP
from rpfm.utils.primary import Moon
from rpfm.utils.const import g0
from rpfm.odes.odes_2d import ODE2dVarThrust


class TwoDimAscVarNLP(NLP):

    def __init__(self, sc, method, nb_seg, order, solver, snopt_opts=None, rec_file=None):

        NLP.__init__(self, method, nb_seg, order, solver, snopt_opts=snopt_opts, rec_file=rec_file)

        self.sc = sc
        self.moon = Moon()

        ode_kwargs = {'Isp': self.sc.Isp, 'g0': g0}
        self.set_trajectory(ODE2dVarThrust, ode_kwargs, 'powered')


if __name__ == '__main__':

    from rpfm.utils.spacecraft import Spacecraft

    sc = Spacecraft(450, 2)
    tr = TwoDimAscVarNLP(sc, 'gauss-lobatto', 100, 3, 'IPOPT')
