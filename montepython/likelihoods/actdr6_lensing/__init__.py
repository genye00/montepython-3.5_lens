from montepython.likelihood_class import Likelihood

import act_dr6_lenslike as alike

class actdr6_lensing(Likelihood):
    def __init__(self, path, data, command_line):
        super().__init__(path, data, command_line)

        self.data_dict = alike.load_data(self.variant,lens_only=self.lens_only)

        self.need_cosmo_arguments(data, {'lensing': 'yes', 'output': 'tCl lCl pCl', 'l_max_scalars': self.lmax})

    def loglkl(self, cosmo, data):
        # get cl, not dl, in muK^2 units
        cls = self.get_cl(cosmo)

        # convert clpp to clkk = L^2*(L+1)^2/(2pi)*clpp
        ell_fac = (cls['ell']*(cls['ell'] + 1))**2*0.25
        cls['pp'] *= ell_fac

        return alike.generic_lnlike(self.data_dict,cls['ell'],cls['pp'],cls['ell'],cls['tt'],cls['ee'],cls['te'],cls['bb'])

    
