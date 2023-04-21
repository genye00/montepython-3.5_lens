from montepython.likelihood_class import Likelihood_prior


class Planck_SZ_prior(Likelihood_prior):

    def loglkl(self, cosmo, data):
        A_sz =  data.mcmc_parameters['A_sz']['current'] * data.mcmc_parameters['A_sz']['scale']
        ksz_norm = data.mcmc_parameters['ksz_norm']['current'] * data.mcmc_parameters['ksz_norm']['scale']
        # Combine the two into one new nuisance-like variable
        val = ksz_norm + 1.6 * A_sz
        return -0.5 * ((val - self.SZ_center)/self.SZ_sigma)**2
