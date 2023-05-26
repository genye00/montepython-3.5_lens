import os
from montepython.likelihood_class import Likelihood_prior


class tau_prior(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        tau = (data.mcmc_parameters["tau_reio"]["current"] * data.mcmc_parameters["tau_reio"]["scale"])
        tau0 = self.tau
        if tau > tau0:
            sigma = self.sigmap
        else:
            sigma = self.sigmam
        loglkl = -0.5 * ((tau - tau0)/sigma)**2
        return loglkl