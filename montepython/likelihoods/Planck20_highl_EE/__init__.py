from montepython.likelihood_class import Likelihood
import glob
import logging
import os
import re
from itertools import combinations
from typing import Optional

import astropy.io.fits as fits
import numpy as np

from . import foregrounds as fg
from . import tools

import io_mp

# list of available foreground models
fg_list = {
    "ps": fg.ps,
    "dust": fg.dust_model,
    "ksz": fg.ksz_model,
    "ps_radio": fg.ps_radio,
    "ps_dusty": fg.ps_dusty,
    "cib": fg.cib_model,
    "tsz": fg.tsz_model,
    "szxcib": fg.szxcib_model,
}


class Planck20_highl_EE(Likelihood):
    def __init__(self, path, data, command_line):
        super().__init__(path, data, command_line)

        if not os.path.exists(self.data_folder):
            raise io_mp.LikelihoodError("The 'data_folder' directory does not exist. Check the given path [%s].",self.data_folder,)
        
        self.frequencies = [100, 100, 143, 143, 217, 217]
        self._mapnames = ["100A", "100B", "143A", "143B", "217A", "217B"]
        self._nmap = len(self.frequencies)
        self._nfreq = len(np.unique(self.frequencies))
        self._nxfreq = self._nfreq * (self._nfreq + 1) // 2
        self._nxspec = self._nmap * (self._nmap - 1) // 2
        self._xspec2xfreq = self._xspec2xfreq()
        print(f"frequencies = {self.frequencies}")

        self._is_mode["ET"] = self._is_mode["TE"]

        # Multipole ranges
        filename = os.path.join(self.data_folder, self.multipoles_range_file)
        self._lmins, self._lmaxs = self._set_multipole_ranges(filename)
        self.lmax = np.max([max(l) for l in self._lmaxs])

        # Data
        basename = os.path.join(self.data_folder, self.xspectra_basename)
        self._dldata = self._read_dl_xspectra(basename, field=1)

        # Weights
        dlsig = self._read_dl_xspectra(basename, field=2)
        dlsig[dlsig == 0] = np.inf
        self._dlweight = 1.0 / dlsig**2

        # Inverted Covariance matrix
        filename = os.path.join(self.data_folder, self.covariance_matrix_file)
        self._invkll = self._read_invcovmatrix(filename)

        # Foregrounds
        self.fgs = []  # list of foregrounds per mode [TT,EE,TE,ET]
        # Init foregrounds TT
        fgsTT = []
        if self._is_mode["TT"]:
            for name in self.foregrounds["TT"].keys():
                if name not in fg_list.keys():
                    raise io_mp.LikelihoodError(f"Unkown foreground model '{name}'!")
                print(f"Adding '{name}' foreground for TT")
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies, mode="TT")
                if isinstance(self.foregrounds["TT"][name], str):
                    kwargs["filename"] = os.path.join(
                        self.data_folder, self.foregrounds["TT"][name]
                    )
                fgsTT.append(fg_list[name](**kwargs))
        self.fgs.append(fgsTT)

        # Init foregrounds EE
        fgsEE = []
        if self._is_mode["EE"]:
            for name in self.foregrounds["EE"].keys():
                if name not in fg_list.keys():
                    raise io_mp.LikelihoodError(f"Unkown foreground model '{name}'!")
                print(f"Adding '{name}' foreground for EE")
                filename = os.path.join(self.data_folder, self.foregrounds["EE"].get(name))
                fgsEE.append(
                    fg_list[name](self.lmax, self.frequencies, mode="EE", filename=filename)
                )
        self.fgs.append(fgsEE)

        # Init foregrounds TE
        fgsTE = []
        fgsET = []
        if self._is_mode["TE"]:
            for name in self.foregrounds["TE"].keys():
                if name not in fg_list.keys():
                    raise io_mp.LikelihoodError(f"Unkown foreground model '{name}'!")
                print(f"Adding '{name}' foreground for TE")
                filename = os.path.join(self.data_folder, self.foregrounds["TE"].get(name))
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies, filename=filename)
                fgsTE.append(fg_list[name](mode="TE", **kwargs))
                fgsET.append(fg_list[name](mode="ET", **kwargs))
        self.fgs.append(fgsTE)
        self.fgs.append(fgsET)

        self.need_cosmo_arguments(data, {'l_max_scalars': self.lmax, 'output': 'tCl pCl lCl', 'lensing': 'yes'})
        
        print("Initialized!")
        
    def _xspec2xfreq(self):
        list_fqs = []
        for f1 in range(self._nfreq):
            for f2 in range(f1, self._nfreq):
                list_fqs.append((f1, f2))

        freqs = list(np.unique(self.frequencies))
        spec2freq = []
        for m1 in range(self._nmap):
            for m2 in range(m1 + 1, self._nmap):
                f1 = freqs.index(self.frequencies[m1])
                f2 = freqs.index(self.frequencies[m2])
                spec2freq.append(list_fqs.index((f1, f2)))

        return spec2freq

    def _set_multipole_ranges(self, filename):
        """
        Return the (lmin,lmax) for each cross-spectra for each mode (TT, EE, TE, ET)
        array(nmode,nxspec)
        """
        print("Define multipole ranges")
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        lmins = []
        lmaxs = []
        for hdu in [0, 1, 3, 3]:  # file HDU [TT,EE,BB,TE]
            tags = ["TT", "EE", "BB", "TE", "TB", "EB"]
            data = fits.getdata(filename, hdu + 1)
            lmins.append(np.array(data.field(0), int))
            lmaxs.append(np.array(data.field(1), int))
            if self._is_mode[tags[hdu]]:
                print(f"{tags[hdu]}")
                print(f"lmin: {np.array(data.field(0), int)}")
                print(f"lmax: {np.array(data.field(1), int)}")

        return lmins, lmaxs

    def _read_dl_xspectra(self, basename, field=1):
        """
        Read xspectra from Xpol [Dl in K^2]
        Output: Dl in muK^2
        """
        print("Reading cross-spectra {}".format("errors" if field == 2 else ""))

        dldata = []
        for m1, m2 in combinations(range(self._nmap), 2):
            tmpcl = []
            for mode, hdu in {"TT": 1, "EE": 2, "TE": 4, "ET": 4}.items():
                filename = f"{basename}_{m1}_{m2}.fits"
                if mode == "ET":
                    filename = f"{basename}_{m2}_{m1}.fits"
                if not os.path.exists(filename):
                    raise ValueError(f"File missing {filename}")
                data = fits.getdata(filename, hdu)
                ell = np.array(data.field(0), int)
                datacl = np.zeros(np.max(ell) + 1)
                datacl[ell] = data.field(field) * 1e12
                tmpcl.append(datacl[: self.lmax + 1])

            dldata.append(tmpcl)

        return np.transpose(np.array(dldata), (1, 0, 2))

    def _read_invcovmatrix(self, filename):
        """
        Read xspectra inverse covmatrix from Xpol [Dl in K^-4]
        Output: invkll [Dl in muK^-4]
        """
        print(f"Covariance matrix file: {filename}")
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        #        data = fits.getdata(filename).field(0)
        data = fits.getdata(filename)
        nel = int(np.sqrt(data.size))
        data = data.reshape((nel, nel)) / 1e24  # muK^-4

        nell = self._get_matrix_size()
        if nel != nell:
            raise ValueError(f"Incoherent covariance matrix (read:{nel}, expected:{nell})")

        return data

    def _get_matrix_size(self):
        """
        Compute covariance matrix size given activated mode
        Return: number of multipole
        """
        nell = 0

        # TT,EE,TEET
        for im, m in enumerate(["TT", "EE", "TE"]):
            if self._is_mode[m]:
                nells = self._lmaxs[im] - self._lmins[im] + 1
                nell += np.sum([nells[self._xspec2xfreq.index(k)] for k in range(self._nxfreq)])

        return nell

    def _select_spectra(self, cl, mode=0):
        """
        Cut spectra given Multipole Ranges and flatten
        Return: list
        """
        acl = np.asarray(cl)
        xl = []
        for xf in range(self._nxfreq):
            lmin = self._lmins[mode][self._xspec2xfreq.index(xf)]
            lmax = self._lmaxs[mode][self._xspec2xfreq.index(xf)]
            xl += list(acl[xf, lmin : lmax + 1])
        return xl

    def _xspectra_to_xfreq(self, cl, weight, normed=True):
        """
        Average cross-spectra per cross-frequency
        """
        xcl = np.zeros((self._nxfreq, self.lmax + 1))
        xw8 = np.zeros((self._nxfreq, self.lmax + 1))
        for xs in range(self._nxspec):
            xcl[self._xspec2xfreq[xs]] += weight[xs] * cl[xs]
            xw8[self._xspec2xfreq[xs]] += weight[xs]

        xw8[xw8 == 0] = np.inf
        if normed:
            return xcl / xw8
        else:
            return xcl, xw8

    def _compute_residuals(self, pars, dlth, mode=0):
        # Nuisances
        cal = []
        for m1, m2 in combinations(range(self._nmap), 2):
            cal.append(
                pars["A_planck"] ** 2
                * (1.0 + pars[f"cal{self._mapnames[m1]}"] + pars[f"cal{self._mapnames[m2]}"])
            )

        # Data
        dldata = self._dldata[mode]

        # Model
        dlmodel = [dlth[mode]] * self._nxspec
        for fg in self.fgs[mode]:
            dlmodel += fg.compute_dl(pars)

        # Compute Rl = Dl - Dlth
        Rspec = np.array([dldata[xs] - cal[xs] * dlmodel[xs] for xs in range(self._nxspec)])

        return Rspec
    
    def compute_chi2(self, dl, pars):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        dl: array or arr2d
              CMB power spectrum (Dl in muK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """

        # cl_boltz from Boltzmann (Cl in muK^2)
        lth = np.arange(self.lmax + 1)
        dlth = np.asarray(dl)[:, lth][[0, 1, 3, 3]]  # select TT,EE,TE,TE

        # Create Data Vector
        Xl = []
        if self._is_mode["TT"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(pars, dlth, mode=0)
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight[0])
            # select multipole range
            Xl += self._select_spectra(Rl, mode=0)

        if self._is_mode["EE"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(pars, dlth, mode=1)
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight[1])
            # select multipole range
            Xl += self._select_spectra(Rl, mode=1)

        if self._is_mode["TE"] or self._is_mode["ET"]:
            Rl = 0
            Wl = 0
            # compute residuals Rl = Dl - Dlth
            if self._is_mode["TE"]:
                Rspec = self._compute_residuals(pars, dlth, mode=2)
                RlTE, WlTE = self._xspectra_to_xfreq(Rspec, self._dlweight[2], normed=False)
                Rl = Rl + RlTE
                Wl = Wl + WlTE
            if self._is_mode["ET"]:
                Rspec = self._compute_residuals(pars, dlth, mode=3)
                RlET, WlET = self._xspectra_to_xfreq(Rspec, self._dlweight[3], normed=False)
                Rl = Rl + RlET
                Wl = Wl + WlET
            # select multipole range
            Xl += self._select_spectra(Rl / Wl, mode=2)

        Xl = np.array(Xl)
        chi2 = Xl @ self._invkll @ Xl

        # self.log.debug(f"chi2/ndof = {chi2}/{len(Xl)}")
        return chi2

    def loglkl(self, cosmo, data):
        dl = self.get_cl(cosmo)
        ells_factor = ((dl["ell"] + 1) * dl["ell"] / (2 * np.pi))[2:]
        for cl in dl:
            if cl not in ['pp', 'ell']:
                dl[cl][2:] *= ells_factor
            if cl == 'pp':
                dl['pp'][2:] *= ells_factor * ells_factor * (2 * np.pi)
        # cl_boltz from Boltzmann (Cl in muK^2)
        lth = np.arange(self.lmax + 1)
        dlth = np.zeros((4, self.lmax + 1))
        dlth[0] = dl["tt"][lth]
        dlth[1] = dl["ee"][lth]
        dlth[3] = dl["te"][lth]

        nuisance_pars = {}
        for par in self.use_nuisance:
            nuisance_pars[par] = data.mcmc_parameters[par]['current'] * data.mcmc_parameters[par]['scale']
        if 'Ad100T' in nuisance_pars:
            nuisance_pars['Ad100T'] = nuisance_pars['Ad100']
        if 'Ad143T' in nuisance_pars:
            nuisance_pars['Ad143T'] = nuisance_pars['Ad143']
        if 'Ad217T' in nuisance_pars:
            nuisance_pars['Ad217T'] = nuisance_pars['Ad217']
        if 'Ad100P' in nuisance_pars:
            nuisance_pars['Ad100P'] = nuisance_pars['Ad100']
        if 'Ad143P' in nuisance_pars:
            nuisance_pars['Ad143P'] = nuisance_pars['Ad143']
        if 'Ad217P' in nuisance_pars:
            nuisance_pars['Ad217P'] = nuisance_pars['Ad217']
        
        chi2 = self.compute_chi2(dlth, nuisance_pars)

        for par in self.use_nuisance:
            if par in self.nuisance_priors:
                mean = self.nuisance_priors[par][0]
                sgm = self.nuisance_priors[par][1]
                chi2 += ((nuisance_pars[par] - mean)/sgm)**2

        return -0.5*chi2
