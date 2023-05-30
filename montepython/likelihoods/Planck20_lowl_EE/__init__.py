from montepython.likelihood_class import Likelihood
from . import tools
from .bins import Bins
import os
import astropy.io.fits as fits
import numpy as np
import io_mp

class Planck20_lowl_EE(Likelihood):
    def __init__(self, path, data, command_line):
        super().__init__(path, data, command_line)

        if not os.path.exists(self.data_folder):
            raise io_mp.LikelihoodError("The 'data_folder' directory does not exist. Check the given path [%s].",self.data_folder,)

        # Binning (fixed binning)
        self.bins = tools.get_binning()
        print(f"lmax = {self.bins.lmax}")

        # Data (ell,ee,bb,eb)
        print("Reading cross-spectrum")
        filepath = os.path.join(self.data_folder, self.cl_file)
        dat = tools.read_dl(filepath)
        self.cldata = self.bins.bin_spectra(dat)

        # Fiducial spectrum (ell,ee,bb,eb)
        print("Reading model")
        filepath = os.path.join(self.data_folder, self.fiducial_file)
        dat = tools.read_dl(filepath)
        self.clfid = self.bins.bin_spectra(dat)

        # covmat (ee,bb,eb)
        print("Reading covariance")
        filepath = os.path.join(self.data_folder, self.cl_cov_file)
        clcov = fits.getdata(filepath)
        if self.mode == "lowlEB":
            cbcov = tools.bin_covEB(clcov, self.bins)
        elif self.mode == "lowlE":
            cbcov = tools.bin_covEE(clcov, self.bins)
        elif self.mode == "lowlB":
            cbcov = tools.bin_covBB(clcov, self.bins)
        clvar = np.diag(cbcov).reshape(-1, self.bins.nbins)

        if self.mode == "lowlEB":
            rcond = getattr(self, "rcond", 1e-9)
            self.invclcov = np.linalg.pinv(cbcov, rcond)
        else:
            self.invclcov = np.linalg.inv(cbcov)

        # Hartlap et al. 2008
        if self.hartlap_factor:
            if self.Nsim != 0:
                self.invclcov *= (self.Nsim - len(cbcov) - 2) / (self.Nsim - 1)

        if self.marginalised_over_covariance:
            if self.Nsim <= 1:
                raise io_mp.LikelihoodError("Need the number of MC simulations used to compute the covariance in order to marginalise over (Nsim>1).")

        # compute offsets
        print("Compute offsets")
        fsky = getattr(self, "fsky", 0.52)
        self.cloff = tools.compute_offsets(self.bins.lbin, clvar, self.clfid, fsky=fsky)
        self.cloff[2:] = 0.0  # force NO offsets EB

        self.need_cosmo_arguments(data, {'l_max_scalars': self.bins.lmax, 'output': 'pCl lCl', 'lensing': 'yes'})
        
        print("Initialized!")

    def _compute_chi2_2fields(self, cl):
        """
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        """
        # get model in Cl, muK^2
        clth = np.array(
            [self.bins.bin_spectra(cl[mode]) for mode in ["ee", "bb", "eb"] if mode in cl]
        )

        nell = self.cldata.shape[1]
        x = np.zeros(self.cldata.shape)
        for ell in range(nell):
            O = tools.vec2mat(self.cloff[:, ell])
            D = tools.vec2mat(self.cldata[:, ell]) + O
            M = tools.vec2mat(clth[:, ell]) + O
            F = tools.vec2mat(self.clfid[:, ell]) + O

            # compute P = C_model^{-1/2}.C_data.C_model^{-1/2}
            w, V = np.linalg.eigh(M)
            #            if prod( sign(w)) <= 0:
            #                print( "WARNING: negative eigenvalue for l=%d" %l)
            L = V @ np.diag(1.0 / np.sqrt(w)) @ V.transpose()
            P = L.transpose() @ D @ L

            # apply HL transformation
            w, V = np.linalg.eigh(P)
            g = np.sign(w) * tools.ghl(np.abs(w))
            G = V @ np.diag(g) @ V.transpose()

            # cholesky fiducial
            w, V = np.linalg.eigh(F)
            L = V @ np.diag(np.sqrt(w)) @ V.transpose()

            # compute C_fid^1/2 * G * C_fid^1/2
            X = L.transpose() @ G @ L
            x[:, ell] = tools.mat2vec(X)

        # compute chi2
        x = x.flatten()
        if self.marginalised_over_covariance:
            chi2 = self.Nsim * np.log(1 + (x @ self.invclcov @ x) / (self.Nsim - 1))
        else:
            chi2 = x @ self.invclcov @ x

        # print(f"chi2/ndof = {chi2}/{len(x)}")
        return chi2

    def _compute_chi2_1field(self, cl):
        """
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        """
        # model in Cl, muK^2
        m = 0 if self.mode == "lowlE" else 1
        clth = self.bins.bin_spectra(cl["ee" if self.mode == "lowlE" else "bb"])

        x = (self.cldata[m] + self.cloff[m]) / (clth + self.cloff[m])
        g = np.sign(x) * tools.ghl(np.abs(x))

        X = (np.sqrt(self.clfid[m] + self.cloff[m])) * g * (np.sqrt(self.clfid[m] + self.cloff[m]))

        if self.marginalised_over_covariance:
            # marginalised over S = Ceff
            chi2 = self.Nsim * np.log(1 + (X @ self.invclcov @ X) / (self.Nsim - 1))
        else:
            chi2 = X @ self.invclcov @ X

        # print(f"chi2/ndof = {chi2}/{len(X)}")
        return chi2
    
    def loglkl(self, cosmo, data):
        cl = self.get_cl(cosmo)
        if self.mode == "lowlEB":
            chi2 = self._compute_chi2_2fields(cl)
        elif self.mode in ["lowlE", "lowlB"]:
            chi2 = self._compute_chi2_1field(cl)

        return -0.5 * chi2
