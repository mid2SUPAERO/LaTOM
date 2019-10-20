import numpy as np

from rpfm.analyzer.analyzer_2d import TwoDimAscAnalyzer
from rpfm.nlp.nlp_heo_2d import TwoDimLLO2HEONLP
from rpfm.plots.solutions import TwoDimSolPlot


class TwoDimAscAnalyzerNRHO(TwoDimAscAnalyzer):

    def __init__(self, body, sc, alt, rp, t, t_bounds, method, nb_seg, order, solver, snopt_opts=None, rec_file=None,
                 check_partials=False, u_bound='lower'):

        TwoDimAscAnalyzer.__init__(self, body, sc, alt)

        self.nlp = TwoDimLLO2HEONLP(body, sc, alt, rp, t, (-np.pi / 2, np.pi / 2), t_bounds, method, nb_seg, order,
                                    solver, self.phase_name, snopt_opts=snopt_opts, rec_file=rec_file,
                                    check_partials=check_partials, u_bound=u_bound)

    def plot(self):

        sol_plot = TwoDimSolPlot(self.body.R, self.time, self.states, self.controls, self.time_exp, self.states_exp,
                                 a=self.nlp.guess.ht.arrOrb.a, e=self.nlp.guess.ht.arrOrb.e)
        sol_plot.plot()


if __name__ == '__main__':

    from rpfm.utils.primary import Moon
    from rpfm.utils.spacecraft import Spacecraft

    moon = Moon()
    sat = Spacecraft(450., 2.)

    tr = TwoDimAscAnalyzerNRHO(moon, sat, 100e3, 3150e3, 6.5655*86400, None, 'gauss-lobatto', 200, 3, 'SNOPT')

    # tr.run_driver()
    # tr.nlp.exp_sim()

    tr.get_solutions(explicit=False)

    print(tr)

    tr.plot()
