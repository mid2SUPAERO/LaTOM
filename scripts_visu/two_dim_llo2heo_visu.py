"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.reader.reader_heo_2d import TwoDim3PhasesLLO2HEOReader
from rpfm.utils.primary import Moon


moon = Moon()
db = 'llo2heo.pkl'

cr = TwoDim3PhasesLLO2HEOReader(moon, db)

cr.plot()
