"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

Defines the methods to save and load files using the pickle library
"""

import pickle


def save(obj, filename):

    with open(filename, 'wb') as fid:
        pickle.dump(obj, fid)


def load(filename):

    with open(filename, 'rb') as fid:

        obj = pickle.load(fid)

        return obj
