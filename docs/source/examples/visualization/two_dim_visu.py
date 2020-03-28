"""
Two-dimensional Moon to LLO transfer visualization
==================================================

This example loads and display an optimal ascent or descent trajectory from the Moon surface to a given LLO.

@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from latom.utils.primary import Moon
from latom.reader.reader_2d import TwoDimReader
from latom.data.transfers.data_transfers import dirname_tr

kind = 'descent'  # kind of transfer between 'ascent' or 'descent'
thrust = 'variable'  # 'constant' or 'variable' thrust magnitude
safe_alt = True  # constrained minimum safe altitude or not

# file IDs
if kind == 'ascent':
    fid = 'asc_vtoff_imp.sql'
    fid_exp = 'asc_vtoff_exp.sql'
elif kind == 'descent':
    fid = 'desc_vland_imp.sql'
    fid_exp = 'desc_vland_exp.sql'
else:
    raise ValueError('kind must be either ascent or descent')

# absolute path to file IDs
rec_file = '/'.join([dirname_tr, fid])
rec_file_exp = '/'.join([dirname_tr, fid_exp])

moon = Moon()  # central attracting body
r = TwoDimReader((kind, thrust, safe_alt), moon, rec_file, db_exp=rec_file_exp)  # load stored solution
r.plot()  # display transfer trajectory
