"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from openmdao.api import CaseReader


class Reader:
    """`Reader` class loads and displays stored simulations using an OpenMDAO `CaseReader` class instance.

    Parameters
    ----------
    db : str
        Full path of the database where the solution is stored
    case_id : str, optional
        Case identifier, ``initial`` to load the first iteration, ``final`` to load the final solution.
        Default is ``final``
    db_exp : str or ``None``, optional
        Full path of the database where the explicit simulation is stored or ``None``. Default is ``None``

    Attributes
    ----------
    case_reader : CaseReader
        OpenMDAO `CaseReader` class instance for the implicit solution
    case_id : str
        Case identifier, ``initial`` to load the first iteration, ``final`` to load the final solution.
    case : Case
        OpenMDAO `Case` class instance identified by `case_id` within the `case_reader` object
    case_reader_exp : CaseReader or ``None``
        OpenMDAO `CaseReader` class instance for the explicit simulation or ``None``
    case : Case or ``None``
        OpenMDAO `Case` class instance identified by `case_id` within the `case_reader_exp` object or ``None``

    """

    def __init__(self, db, case_id='final', db_exp=None):
        """Init Reader class. """

        self.case_reader = CaseReader(db)

        if case_id in ['initial', 'final']:
            self.case_id = case_id
        else:
            raise ValueError("Case must be either 'initial' or 'final'")

        self.case = self.case_reader.get_case(self.case_id)

        # explicit simulation
        if db_exp is not None:
            self.case_reader_exp = CaseReader(db_exp)
            self.case_exp = self.case_reader_exp.get_case(-1)
        else:
            self.case_reader_exp = self.case_exp = None
