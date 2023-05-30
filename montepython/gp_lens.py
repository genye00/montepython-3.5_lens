import numpy as np

class gp_gen():
    def __init__(self, hyperpar, l_node):
        self.hyperpar = np.array(hyperpar) # [sigma^2, l^2]
        def _kernel(x1,x2):
            sigma2 = self.hyperpar[0]
            l2 = self.hyperpar[1]
            x1, x2 = np.meshgrid(x2, x1)
            return sigma2*np.exp(-0.5*(x1-x2)**2/l2)
        self.lmax = 4000
        self.dat_x = np.log(np.array(l_node))
        self.itp_x = np.log(np.arange(2, self.lmax+1)) # L<2 is of no use
        k_id = _kernel(self.itp_x, self.dat_x)
        kn_dd_inv = np.linalg.inv(_kernel(self.dat_x, self.dat_x) + np.diag(1e-8*np.ones(len(self.dat_x))))
        k_ii = _kernel(self.itp_x, self.itp_x)
        k_di = _kernel(self.dat_x, self.itp_x)
        covm = k_ii - k_id@kn_dd_inv@k_di
        (u, s, vh) = np.linalg.svd(covm)
        self.gp_conm = (u * np.sqrt(s)).T
        self.gp_meanm = k_id@kn_dd_inv
        self.renew_cache()
    
    def renew_cache(self):
        self.rng_pool = np.random.default_rng().standard_normal(size=len(self.itp_x)*1024)
        self.cache_index = 0

    def get_func(self, dat_y):
        if len(dat_y) != len(self.dat_x): raise Exception("dat_y is NOT of the same length as dat_x")
        if self.cache_index == 1024: self.renew_cache()
        dat_y = np.array(dat_y)
        y_mean = np.mean(dat_y)
        idx = self.cache_index*len(self.itp_x)
        rng_seed = self.rng_pool[idx:idx+len(self.itp_x)]
        a_lens = np.zeros(self.lmax+1)
        a_lens[2:] = y_mean + self.gp_meanm@(dat_y-y_mean) + rng_seed@self.gp_conm
        a_lens = np.exp(a_lens)
        self.cache_index += 1
        return a_lens

    