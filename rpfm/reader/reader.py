"""
@authors: Alberto FOSSA' Giuliana Elena MICELI

"""

from openmdao.api import CaseReader


class Reader:

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
