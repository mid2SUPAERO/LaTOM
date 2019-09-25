"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.pickle_utils import load

path = '/home/alberto/Nextcloud/HOmeBOX/Documents/surrogate/'
fid = path + 'sm_var_snopt.pkl'

sm = load(fid)
sm.plot()
