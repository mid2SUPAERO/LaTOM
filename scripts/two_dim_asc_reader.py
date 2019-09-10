"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from rpfm.utils.primary import Moon
from rpfm.reader.reader_2d import TwoDimReader

moon = Moon()

rec_file = '/home/alberto/Downloads/rec.sql'
rec_file_exp = '/home/alberto/Downloads/rec_exp.sql'

kind = ('ascent', 'variable', True)

r = TwoDimReader(kind, moon, rec_file, db_exp=rec_file_exp)
r.plot()
