"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.reader.reader_heo_2d import TwoDim3PhasesLLO2HEOReader
from rpfm.utils.primary import Moon


moon = Moon()

path = '/home/alberto/Documents/RpResults/'
db = path + 's7/llo2heo_ipopt1200.pkl'
db_exp = path + 's7/llo2heo_ipopt1200_exp.pkl'

cr = TwoDim3PhasesLLO2HEOReader(moon, db, db_exp=db_exp)

cr.plot()
