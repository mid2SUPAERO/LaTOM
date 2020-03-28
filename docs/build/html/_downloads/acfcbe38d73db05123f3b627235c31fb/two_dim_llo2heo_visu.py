"""
Two-dimensional, three-phases LLO to HEO transfer visualization
===============================================================

This example loads and display an optimal transfer trajectory from LLO to HEO composed by three subsequent phases.

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from latom.reader.reader_heo_2d import TwoDim3PhasesLLO2HEOReader
from latom.utils.primary import Moon
from latom.data.transfers.data_transfers import dirname_tr

fid = 'llo2heo_ipopt1200.sql'  # file ID
rec_file = '/'.join([dirname_tr, fid])  # absolute path to file ID

moon = Moon()  # central attracting body
cr = TwoDim3PhasesLLO2HEOReader(moon, rec_file, db_exp=None)  # load stored data
cr.plot()  # display solution
