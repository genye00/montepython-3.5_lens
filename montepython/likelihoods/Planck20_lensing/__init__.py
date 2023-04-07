from montepython.likelihood_class import Likelihood


class Planck20_lensing(Likelihood):
    def __init__(self, path, data, command_line):
        super().__init__(path, data, command_line)
