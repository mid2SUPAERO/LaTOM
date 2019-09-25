"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.pickle_utils import load

path = '/home/alberto/Nextcloud/HOmeBOX/Documents/surrogate/'
fid = path + 'sm_var.pkl'

sm = load(fid)
# sm.plot()

# train_method must be one between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC
sm.train('RMTC')
sm.evaluate(nb_eval=2500)
sm.plot()
