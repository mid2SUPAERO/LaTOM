"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

Defines the methods to save and load files using the pickle library
"""

import pickle


def save(obj, filename):
    """Save an object.

    Parameters
    ----------
    obj : object
        Object to be serialized
    filename : str
        Full path to file where the object is serialized

    """

    with open(filename, 'wb') as fid:
        pickle.dump(obj, fid)


def load(filename):
    """Load an object.

    Parameters
    ----------
    filename : str
        Full path to file where the object is serialized

    Returns
    -------
    obj : object
        Object to be serialized

    """

    with open(filename, 'rb') as fid:

        obj = pickle.load(fid)

        return obj
