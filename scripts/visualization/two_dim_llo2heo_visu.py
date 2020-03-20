"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from latom.reader.reader_heo_2d import TwoDim3PhasesLLO2HEOReader
from latom.utils.primary import Moon
from latom.data.llo2heo3phases.data_llo2heo3ph import dirname_llo2heo3ph


moon = Moon()

db = dirname_llo2heo3ph + '/llo2heo_ipopt1200.pkl'

cr = TwoDim3PhasesLLO2HEOReader(moon, db, db_exp=None)

cr.plot()
