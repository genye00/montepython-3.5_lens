import os
from montepython.likelihood_class import Likelihood_prior


class Planck_prior(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):
        lkl = 0
        for key, dat in self.priors.items():
            if key in self.use_nuisance:
                val = data.mcmc_parameters[key]['current'] * data.mcmc_parameters[key]['scale']
                lkl -= 0.5 * ((val-dat[0])/dat[1])**2
        if self.SZ_prior:    
            A_sz =  data.mcmc_parameters['A_sz']['current'] * data.mcmc_parameters['A_sz']['scale']
            ksz_norm = data.mcmc_parameters['ksz_norm']['current'] * data.mcmc_parameters['ksz_norm']['scale']
            # Combine the two into one new nuisance-like variable
            val = ksz_norm + 1.6 * A_sz
            dat = self.priors['SZ']
            lkl -= 0.5 * ((val-dat[0])/dat[1])**2
        return lkl
