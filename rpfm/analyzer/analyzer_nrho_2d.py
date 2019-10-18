import numpy as np

from rpfm.analyzer.analyzer_2d import TwoDimAscAnalyzer
from rpfm.nlp.nlp_nrho_2d import TwoDimAscVarNRHO
from rpfm.plots.solutions import TwoDimSolPlot


class TwoDimAscAnalyzerNRHO(TwoDimAscAnalyzer):

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False, u_bound='lower'):

        TwoDimAscAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimAscVarNRHO(body, sc, alt, rp, t, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg, order,
                                    solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                    check_partials=check_partials, u_bound=u_bound)

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 a=self.nlp.guess.ep.a_nrho, e=self.nlp.guess.ep.e_nrho)
        sol_plot.plot()


if __name__ == '__main__':

    from rpfm.utils.primary import Moon
    from rpfm.utils.spacecraft import Spacecraft

    moon = Moon()
    sc = Spacecraft(450., 2.)

    tr = TwoDimAscAnalyzerNRHO(moon, sc, 100e3, 3150e3, 6.5655, None, 'gauss-lobatto', 400, 3, 'SNOPT')

    tr.run_driver()
    tr.nlp.exp_sim()

    tr.get_solutions()

    print(tr)

    tr.plot()
