"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.pickle_utils import load

path = '/home/alberto/Nextcloud/HOmeBOX/Documents/surrogate/'
fid = path + 'sm_ese.pkl'
train = False
nb_eval = 900

sm = load(fid)

if train:
    sm.train('QP')  # train_method must be one between IDW, KPLS, KPLSK, KRG, LS, QP, RBF, RMTB, RMTC
    sm.evaluate(nb_eval=nb_eval)

sm.plot()
