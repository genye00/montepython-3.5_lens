"""
.. module:: likelihood_class
   :synopsis: Definition of the major likelihoods
.. moduleauthor:: Julien Lesgourgues <lesgourg@cern.ch>
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

Contains the definition of the base likelihood class :class:`Likelihood`, with
basic functions, as well as more specific likelihood classes that may be reused
to implement new ones.

"""
from __future__ import print_function
import os
import numpy as np
import math
import warnings
import re
import scipy.constants as const
import scipy.integrate
import scipy.interpolate
import scipy.misc

from getdist import IniFile, ParamNames
from typing import List
from scipy.linalg import sqrtm

import io_mp
from io_mp import dictitems,dictvalues,dictkeys


class Likelihood(object):
    """
    General class that all likelihoods will inherit from.

    """

    def __init__(self, path, data, command_line):
        """
        It copies the content of self.path from the initialization routine of
        the :class:`Data <data.Data>` class, and defines a handful of useful
        methods, that every likelihood might need.

        If the nuisance parameters required to compute this likelihood are not
        defined (either fixed or varying), the code will stop.

        Parameters
        ----------
        data : class
            Initialized instance of :class:`Data <data.Data>`
        command_line : NameSpace
            NameSpace containing the command line arguments

        """

        self.name = self.__class__.__name__
        self.folder = os.path.abspath(os.path.join(
            data.path['MontePython'], 'likelihoods', self.name))
        if not data.log_flag:
            path = os.path.join(command_line.folder, 'log.param')

        # Define some default fields
        self.data_directory = ''

        # Store all the default fields stored, for the method read_file.
        self.default_values = ['data_directory']

        # Recover the values potentially read in the input.param file.
        if hasattr(data, self.name):
            attributes = [e for e in dir(getattr(data,self.name)) if e.find('__') == -1]
            for elem in attributes:
                setattr(self, elem, getattr(getattr(data,self.name), elem))

        # Read values from the data file
        self.read_from_file(path, data, command_line)

        # Default state
        self.need_update = True

        # Check if the nuisance parameters are defined
        error_flag = False
        try:
            for nuisance in self.use_nuisance:
                if nuisance not in data.get_mcmc_parameters(['nuisance']):
                    error_flag = True
                    warnings.warn(
                        nuisance + " must be defined, either fixed or " +
                        "varying, for %s likelihood" % self.name)
            self.nuisance = self.use_nuisance
        except AttributeError:
            self.use_nuisance = []
            self.nuisance = []

        # If at least one is missing, raise an exception.
        if error_flag:
            raise io_mp.LikelihoodError(
                "Check your nuisance parameter list for your set of experiments")

        # Append to the log.param the value used (WARNING: so far no comparison
        # is done to ensure that the experiments share the same parameters)
        if data.log_flag:
            io_mp.log_likelihood_parameters(self, command_line)

    def loglkl(self, cosmo, data):
        """
        Placeholder to remind that this function needs to be defined for a
        new likelihood.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError(
            'Must implement method loglkl() in your likelihood')

    def read_from_file(self, path, data, command_line):
        """
        Extract the information from the log.param concerning this likelihood.

        If the log.param is used, check that at least one item for each
        likelihood is recovered. Otherwise, it means the log.param does not
        contain information on the likelihood. This happens when the first run
        fails early, before calling the likelihoods, and the program did not
        log the information. This check might not be completely secure, but it
        is better than nothing.

        .. warning::

            This checks relies on the fact that a likelihood should always have
            at least **one** line of code written in the likelihood.data file.
            This should be always true, but in case a run fails with the error
            message described below, think about it.

        .. warning::

            As of version 2.0.2, you can specify likelihood options in the
            parameter file. They have complete priority over the ones specified
            in the `likelihood.data` file, and it will be reflected in the
            `log.param` file.

        """

        # Counting how many lines are read.
        counter = 0

        self.path = path
        self.dictionary = {}
        if os.path.isfile(path):
            data_file = open(path, 'r')
            for line in data_file:
                if line.find('#') == -1:
                    if line.find(self.name+'.') != -1:
                        # Recover the name and value from the .data file
                        regexp = re.match(
                            "%s.(.*)\s*=\s*(.*)" % self.name, line)
                        name, value = (
                            elem.strip() for elem in regexp.groups())
                        # If this name was already defined in the parameter
                        # file, be sure to take this value instead. Beware,
                        # there are a few parameters which are always
                        # predefined, such as data_directory, which should be
                        # ignored in this check.
                        is_ignored = False
                        if name not in self.default_values:
                            try:
                                value = getattr(self, name)
                                is_ignored = True
                            except AttributeError:
                                pass
                        if not is_ignored:
                            exec('self.'+name+' = '+value)
                        value = getattr(self, name)
                        counter += 1
                        self.dictionary[name] = value
            data_file.seek(0)
            data_file.close()

        # Checking that at least one line was read, exiting otherwise
        if counter == 0:
            raise io_mp.ConfigurationError(
                "No information on %s likelihood " % self.name +
                "was found in the %s file.\n" % path +
                "This can result from a failed initialization of a previous " +
                "run. To solve this, you can do a \n " +
                "]$ rm -rf %s \n " % command_line.folder +
                "Be sure there is noting in it before doing this !")

    def get_cl(self, cosmo, l_max=-1):
        """
        Return the :math:`C_{\ell}` from the cosmological code in
        :math:`\mu {\\rm K}^2`

        """
        # get C_l^XX from the cosmological code
        cl = cosmo.lensed_cl(int(l_max))

        # convert dimensionless C_l's to C_l in muK**2
        T = cosmo.T_cmb()
        for key in dictkeys(cl):
            # All quantities need to be multiplied by this factor, except the
            # phi-phi term, that is already dimensionless
            # phi cross-terms should only be multiplied with this factor once
            if key not in ['pp', 'ell', 'tp', 'ep']:
                cl[key] *= (T*1.e6)**2
            elif key in ['tp', 'ep']:
                cl[key] *= (T*1.e6)

        return cl

    def get_unlensed_cl(self, cosmo, l_max=-1):
        """
        Return the :math:`C_{\ell}` from the cosmological code in
        :math:`\mu {\\rm K}^2`

        """
        # get C_l^XX from the cosmological code
        cl = cosmo.raw_cl(l_max)

        # convert dimensionless C_l's to C_l in muK**2
        T = cosmo.T_cmb()
        for key in dictkeys(cl):
            # All quantities need to be multiplied by this factor, except the
            # phi-phi term, that is already dimensionless
            # phi cross-terms should only be multiplied with this factor once
            if key not in ['pp', 'ell', 'tp', 'ep']:
                cl[key] *= (T*1.e6)**2
            elif key in ['tp', 'ep']:
                cl[key] *= (T*1.e6)

        return cl

    def need_cosmo_arguments(self, data, dictionary):
        """
        Ensure that the arguments of dictionary are defined to the correct
        value in the cosmological code

        .. warning::

            So far there is no way to enforce a parameter where `smaller is
            better`. A bigger value will always overried any smaller one
            (`cl_max`, etc...)

        Parameters
        ----------
        data : dict
            Initialized instance of :class:`data`
        dictionary : dict
            Desired precision for some cosmological parameters

        """

        for key, value in dictitems(dictionary):
            array_flag = False
            num_flag = True
            try:
                data.cosmo_arguments[key]
                try:
                    float(data.cosmo_arguments[key])
                except ValueError:
                    num_flag = False
                    # ML+NS+DCH --> Fixed the adding of new string parameters generating spurious copies and/or concatenating without spaces
                    splitstring = [item for string in value.split(" ") for item in string.split(",")]
                    for addvalue in splitstring:
                        if data.cosmo_arguments[key].find(addvalue)==-1:
                            data.cosmo_arguments[key] += ' '+addvalue+''
                except TypeError:
                    array_flag = True

            except KeyError:
                try:
                    float(value)
                    data.cosmo_arguments[key] = 0
                except ValueError:
                    num_flag = False
                    data.cosmo_arguments[key] = ''+value+''
                    #print(data.cosmo_arguments[key])
                except TypeError:
                    array_flag = True
            
            if num_flag is True:
                if array_flag is False:
                    if float(data.cosmo_arguments[key]) < value:
                        data.cosmo_arguments[key] = value
                else:
                    data.cosmo_arguments[key] = '%.2g' % value[0]
                    for i in range(1, len(value)):
                        data.cosmo_arguments[key] += ',%.2g' % (value[i])

    def read_contamination_spectra(self, data):

        for nuisance in self.use_nuisance:
            # read spectrum contamination (so far, assumes only temperature
            # contamination; will be trivial to generalize to polarization when
            # such templates will become relevant)
            setattr(self, "%s_contamination" % nuisance,
                    np.zeros(self.l_max+1, 'float64'))
            try:
                File = open(os.path.join(
                    self.data_directory, getattr(self, "%s_file" % nuisance)),
                    'r')
                for line in File:
                    l = int(float(line.split()[0]))
                    if ((l >= 2) and (l <= self.l_max)):
                        exec("self.%s_contamination[l]=float(line.split()[1])/(l*(l+1.)/2./math.pi)" % nuisance)
            except:
                print('Warning: you did not pass a file name containing ')
                print('a contamination spectrum regulated by the nuisance ')
                print('parameter '+nuisance)

            # read renormalization factor
            # if it is not there, assume it is one, i.e. do not renormalize
            try:
                # do the following operation:
                # self.nuisance_contamination *= float(self.nuisance_scale)
                setattr(self, "%s_contamination" % nuisance,
                        getattr(self, "%s_contamination" % nuisance) *
                        float(getattr(self, "%s_scale" % nuisance)))
            except AttributeError:
                pass

            # read central value of nuisance parameter
            # if it is not there, assume one by default
            try:
                getattr(self, "%s_prior_center" % nuisance)
            except AttributeError:
                setattr(self, "%s_prior_center" % nuisance, 1.)

            # read variance of nuisance parameter
            # if it is not there, assume flat prior (encoded through
            # variance=0)
            try:
                getattr(self, "%s_prior_variance" % nuisance)
            except:
                setattr(self, "%s_prior_variance" % nuisance, 0.)

    def add_contamination_spectra(self, cl, data):

        # Recover the current value of the nuisance parameter.
        for nuisance in self.use_nuisance:
            nuisance_value = float(
                data.mcmc_parameters[nuisance]['current'] *
                data.mcmc_parameters[nuisance]['scale'])

            # add contamination spectra multiplied by nuisance parameters
            for l in range(2, self.l_max):
                exec("cl['tt'][l] += nuisance_value*self.%s_contamination[l]" % nuisance)

        return cl

    def add_nuisance_prior(self, lkl, data):

        # Recover the current value of the nuisance parameter.
        for nuisance in self.use_nuisance:
            nuisance_value = float(
                data.mcmc_parameters[nuisance]['current'] *
                data.mcmc_parameters[nuisance]['scale'])

            # add prior on nuisance parameters
            if getattr(self, "%s_prior_variance" % nuisance) > 0:
                # convenience variables
                prior_center = getattr(self, "%s_prior_center" % nuisance)
                prior_variance = getattr(self, "%s_prior_variance" % nuisance)
                lkl += -0.5*((nuisance_value-prior_center)/prior_variance)**2

        return lkl

    def computeLikelihood(self, ctx):
        """
        Interface with CosmoHammer

        Parameters
        ----------
        ctx : Context
                Contains several dictionaries storing data and cosmological
                information

        """
        # Recover both instances from the context
        cosmo = ctx.get("cosmo")
        data = ctx.get("data")

        loglkl = self.loglkl(cosmo, data)

        return loglkl


###################################
#
# END OF GENERIC LIKELIHOOD CLASS
#
###################################



###################################
# PRIOR TYPE LIKELIHOOD
# --> H0,...
###################################
class Likelihood_prior(Likelihood):

    def loglkl(self):
        raise NotImplementedError('Must implement method loglkl() in your likelihood')


###################################
# NEWDAT TYPE LIKELIHOOD
# --> spt,boomerang,etc.
###################################
class Likelihood_newdat(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        # open .newdat file
        newdatfile = open(
            os.path.join(self.data_directory, self.file), 'r')

        # find beginning of window functions file names
        window_name = newdatfile.readline().strip('\n').replace(' ', '')

        # initialize list of fist and last band for each type
        band_num = np.zeros(6, 'int')
        band_min = np.zeros(6, 'int')
        band_max = np.zeros(6, 'int')

        # read number of bands for each of the six types TT, EE, BB, EB, TE, TB
        line = newdatfile.readline()
        for i in range(6):
            band_num[i] = int(line.split()[i])

        # read string equal to 'BAND_SELECTION' or not
        line = str(newdatfile.readline()).strip('\n').replace(' ', '')

        # if yes, read 6 lines containing 'min, max'
        if (line == 'BAND_SELECTION'):
            for i in range(6):
                line = newdatfile.readline()
                band_min[i] = int(line.split()[0])
                band_max[i] = int(line.split()[1])

        # if no, set min to 1 and max to band_num (=use all bands)
        else:
            band_min = [1 for i in range(6)]
            band_max = band_num

        # read line defining calibration uncertainty
        # contains: flag (=0 or 1), calib, calib_uncertainty
        line = newdatfile.readline()
        calib = float(line.split()[1])
        if (int(line.split()[0]) == 0):
            self.calib_uncertainty = 0
        else:
            self.calib_uncertainty = float(line.split()[2])

        # read line defining beam uncertainty
        # contains: flag (=0, 1 or 2), beam_width, beam_sigma
        line = newdatfile.readline()
        beam_type = int(line.split()[0])
        if (beam_type > 0):
            self.has_beam_uncertainty = True
        else:
            self.has_beam_uncertainty = False
        beam_width = float(line.split()[1])
        beam_sigma = float(line.split()[2])

        # read flag (= 0, 1 or 2) for lognormal distributions and xfactors
        line = newdatfile.readline()
        likelihood_type = int(line.split()[0])
        if (likelihood_type > 0):
            self.has_xfactors = True
        else:
            self.has_xfactors = False

        # declare array of quantitites describing each point of measurement
        # size yet unknown, it will be found later and stored as
        # self.num_points
        self.obs = np.array([], 'float64')
        self.var = np.array([], 'float64')
        self.beam_error = np.array([], 'float64')
        self.has_xfactor = np.array([], 'bool')
        self.xfactor = np.array([], 'float64')

        # temporary array to know which bands are actually used
        used_index = np.array([], 'int')

        index = -1

        # scan the lines describing each point of measurement
        for cltype in range(6):
            if (int(band_num[cltype]) != 0):
                # read name (but do not use it)
                newdatfile.readline()
                for band in range(int(band_num[cltype])):
                    # read one line corresponding to one measurement
                    line = newdatfile.readline()
                    index += 1

                    # if we wish to actually use this measurement
                    if ((band >= band_min[cltype]-1) and
                            (band <= band_max[cltype]-1)):

                        used_index = np.append(used_index, index)

                        self.obs = np.append(
                            self.obs, float(line.split()[1])*calib**2)

                        self.var = np.append(
                            self.var,
                            (0.5*(float(line.split()[2]) +
                                  float(line.split()[3]))*calib**2)**2)

                        self.xfactor = np.append(
                            self.xfactor, float(line.split()[4])*calib**2)

                        if ((likelihood_type == 0) or
                                ((likelihood_type == 2) and
                                (int(line.split()[7]) == 0))):
                            self.has_xfactor = np.append(
                                self.has_xfactor, [False])
                        if ((likelihood_type == 1) or
                                ((likelihood_type == 2) and
                                (int(line.split()[7]) == 1))):
                            self.has_xfactor = np.append(
                                self.has_xfactor, [True])

                        if (beam_type == 0):
                            self.beam_error = np.append(self.beam_error, 0.)
                        if (beam_type == 1):
                            l_mid = float(line.split()[5]) +\
                                0.5*(float(line.split()[5]) +
                                     float(line.split()[6]))
                            self.beam_error = np.append(
                                self.beam_error,
                                abs(math.exp(
                                    -l_mid*(l_mid+1)*1.526e-8*2.*beam_sigma *
                                    beam_width)-1.))
                        if (beam_type == 2):
                            if (likelihood_type == 2):
                                self.beam_error = np.append(
                                    self.beam_error, float(line.split()[8]))
                            else:
                                self.beam_error = np.append(
                                    self.beam_error, float(line.split()[7]))

                # now, skip and unused part of the file (with sub-correlation
                # matrices)
                for band in range(int(band_num[cltype])):
                    newdatfile.readline()

        # number of points that we will actually use
        self.num_points = np.shape(self.obs)[0]

        # total number of points, including unused ones
        full_num_points = index+1

        # read full correlation matrix
        full_covmat = np.zeros((full_num_points, full_num_points), 'float64')
        for point in range(full_num_points):
            full_covmat[point] = newdatfile.readline().split()

        # extract smaller correlation matrix for points actually used
        covmat = np.zeros((self.num_points, self.num_points), 'float64')
        for point in range(self.num_points):
            covmat[point] = full_covmat[used_index[point], used_index]

        # recalibrate this correlation matrix
        covmat *= calib**4

        # redefine the correlation matrix, the observed points and their
        # variance in case of lognormal likelihood
        if (self.has_xfactors):

            for i in range(self.num_points):

                for j in range(self.num_points):
                    if (self.has_xfactor[i]):
                        covmat[i, j] /= (self.obs[i]+self.xfactor[i])
                    if (self.has_xfactor[j]):
                        covmat[i, j] /= (self.obs[j]+self.xfactor[j])

            for i in range(self.num_points):
                if (self.has_xfactor[i]):
                    self.var[i] /= (self.obs[i]+self.xfactor[i])**2
                    self.obs[i] = math.log(self.obs[i]+self.xfactor[i])

        # invert correlation matrix
        self.inv_covmat = np.linalg.inv(covmat)

        # read window function files a first time, only for finding the
        # smallest and largest l's for each point
        self.win_min = np.zeros(self.num_points, 'int')
        self.win_max = np.zeros(self.num_points, 'int')
        for point in range(self.num_points):
            for line in open(os.path.join(
                    self.data_directory, 'windows', window_name) +
                    str(used_index[point]+1), 'r'):
                if any([float(line.split()[i]) != 0.
                        for i in range(1, len(line.split()))]):
                    if (self.win_min[point] == 0):
                        self.win_min[point] = int(line.split()[0])
                    self.win_max[point] = int(line.split()[0])

        # infer from format of window function files whether we will use
        # polarisation spectra or not
        num_col = len(line.split())
        if (num_col == 2):
            self.has_pol = False
        else:
            if (num_col == 5):
                self.has_pol = True
            else:
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "Window function files are understood if they contain " +
                    "2 columns (l TT), or 5 columns (l TT TE EE BB)." +
                    "In this case the number of columns is %d" % num_col)

        # define array of window functions
        self.window = np.zeros(
            (self.num_points, max(self.win_max)+1, num_col-1), 'float64')

        # go again through window function file, this time reading window
        # functions; that are distributed as: l TT (TE EE BB) where the last
        # columns contaim W_l/l, not W_l we mutiply by l in order to store the
        # actual W_l
        for point in range(self.num_points):
            for line in open(os.path.join(
                    self.data_directory, 'windows', window_name) +
                    str(used_index[point]+1), 'r'):
                l = int(line.split()[0])
                if (((self.has_pol is False) and (len(line.split()) != 2))
                        or ((self.has_pol is True) and
                            (len(line.split()) != 5))):
                    raise io_mp.LikelihoodError(
                        "In likelihood %s. " % self.name +
                        "for a given experiment, all window functions should" +
                        " have the same number of columns, 2 or 5. " +
                        "This is not the case here.")
                if ((l >= self.win_min[point]) and (l <= self.win_max[point])):
                    self.window[point, l, :] = [
                        float(line.split()[i])
                        for i in range(1, len(line.split()))]
                    self.window[point, l, :] *= l

        # eventually, initialise quantitites used in the marginalization over
        # nuisance parameters
        if ((self.has_xfactors) and
                ((self.calib_uncertainty > 1.e-4) or
                 (self.has_beam_uncertainty))):
            self.halfsteps = 5
            self.margeweights = np.zeros(2*self.halfsteps+1, 'float64')
            for i in range(-self.halfsteps, self.halfsteps+1):
                self.margeweights[i+self.halfsteps] = np.exp(
                    -(float(i)*3./float(self.halfsteps))**2/2)
            self.margenorm = sum(self.margeweights)

        # store maximum value of l needed by window functions
        self.l_max = max(self.win_max)

        # impose that the cosmological code computes Cl's up to maximum l
        # needed by the window function
        self.need_cosmo_arguments(data, {'l_max_scalars': self.l_max})

        # deal with nuisance parameters
        try:
            self.use_nuisance
            self.nuisance = self.use_nuisance
        except:
            self.use_nuisance = []
            self.nuisance = []
        self.read_contamination_spectra(data)

        # end of initialisation

    def loglkl(self, cosmo, data):
        # get Cl's from the cosmological code
        cl = self.get_cl(cosmo)

        # add contamination spectra multiplied by nuisance parameters
        cl = self.add_contamination_spectra(cl, data)

        # get likelihood
        lkl = self.compute_lkl(cl, cosmo, data)

        # add prior on nuisance parameters
        # lkl = self.add_nuisance_prior(lkl, data) # do nuisance as seperate lkl, Gen Ye

        return lkl

    def compute_lkl(self, cl, cosmo, data):
        # checks that Cl's have been computed up to high enough l given window
        # function range. Normally this has been imposed before, so this test
        # could even be supressed.
        if (np.shape(cl['tt'])[0]-1 < self.l_max):
            raise io_mp.LikelihoodError(
                "%s computed Cls till l=" % data.cosmological_module_name +
                "%d " % (np.shape(cl['tt'])[0]-1) +
                "while window functions need %d." % self.l_max)

        # compute theoretical bandpowers, store them in theo[points]
        theo = np.zeros(self.num_points, 'float64')

        for point in range(self.num_points):

            # find bandpowers B_l by convolving C_l's with [(l+1/2)/2pi W_l]
            for l in range(self.win_min[point], self.win_max[point]):

                theo[point] += cl['tt'][l]*self.window[point, l, 0] *\
                    (l+0.5)/2./math.pi

                if (self.has_pol):
                    theo[point] += (
                        cl['te'][l]*self.window[point, l, 1] +
                        cl['ee'][l]*self.window[point, l, 2] +
                        cl['bb'][l]*self.window[point, l, 3]) *\
                        (l+0.5)/2./math.pi

        # allocate array for differencve between observed and theoretical
        # bandpowers
        difference = np.zeros(self.num_points, 'float64')

        # depending on the presence of lognormal likelihood, calibration
        # uncertainty and beam uncertainity, use several methods for
        # marginalising over nuisance parameters:

        # first method: numerical integration over calibration uncertainty:
        if (self.has_xfactors and
                ((self.calib_uncertainty > 1.e-4) or
                 self.has_beam_uncertainty)):

            chisq_tmp = np.zeros(2*self.halfsteps+1, 'float64')
            chisqcalib = np.zeros(2*self.halfsteps+1, 'float64')
            beam_error = np.zeros(self.num_points, 'float64')

            # loop over various beam errors
            for ibeam in range(2*self.halfsteps+1):

                # beam error
                for point in range(self.num_points):
                    if (self.has_beam_uncertainty):
                        beam_error[point] = 1.+self.beam_error[point] *\
                            (ibeam-self.halfsteps)*3/float(self.halfsteps)
                    else:
                        beam_error[point] = 1.

                # loop over various calibraion errors
                for icalib in range(2*self.halfsteps+1):

                    # calibration error
                    calib_error = 1+self.calib_uncertainty*(
                        icalib-self.halfsteps)*3/float(self.halfsteps)

                    # compute difference between observed and theoretical
                    # points, after correcting the later for errors
                    for point in range(self.num_points):

                        # for lognormal likelihood, use log(B_l+X_l)
                        if (self.has_xfactor[point]):
                            difference[point] = self.obs[point] -\
                                math.log(
                                    theo[point]*beam_error[point] *
                                    calib_error+self.xfactor[point])
                        # otherwise use B_l
                        else:
                            difference[point] = self.obs[point] -\
                                theo[point]*beam_error[point]*calib_error

                    # find chisq with those corrections
                    # chisq_tmp[icalib] = np.dot(np.transpose(difference),
                    # np.dot(self.inv_covmat, difference))
                    chisq_tmp[icalib] = np.dot(
                        difference, np.dot(self.inv_covmat, difference))

                minchisq = min(chisq_tmp)

            # find chisq marginalized over calibration uncertainty (if any)
                tot = 0
                for icalib in range(2*self.halfsteps+1):
                    tot += self.margeweights[icalib]*math.exp(
                        max(-30., -(chisq_tmp[icalib]-minchisq)/2.))

                chisqcalib[ibeam] = -2*math.log(tot/self.margenorm)+minchisq

            # find chisq marginalized over beam uncertainty (if any)
            if (self.has_beam_uncertainty):

                minchisq = min(chisqcalib)

                tot = 0
                for ibeam in range(2*self.halfsteps+1):
                    tot += self.margeweights[ibeam]*math.exp(
                        max(-30., -(chisqcalib[ibeam]-minchisq)/2.))

                chisq = -2*math.log(tot/self.margenorm)+minchisq

            else:
                chisq = chisqcalib[0]

        # second method: marginalize over nuisance parameters (if any)
        # analytically
        else:

            # for lognormal likelihood, theo[point] should contain log(B_l+X_l)
            if (self.has_xfactors):
                for point in range(self.num_points):
                    if (self.has_xfactor[point]):
                        theo[point] = math.log(theo[point]+self.xfactor[point])

            # find vector of difference between observed and theoretical
            # bandpowers
            difference = self.obs-theo

            # find chisq
            chisq = np.dot(
                np.transpose(difference), np.dot(self.inv_covmat, difference))

            # correct eventually for effect of analytic marginalization over
            # nuisance parameters
            if ((self.calib_uncertainty > 1.e-4) or self.has_beam_uncertainty):

                denom = 1.
                tmpi = np.dot(self.inv_covmat, theo)
                chi2op = np.dot(np.transpose(difference), tmp)
                chi2pp = np.dot(np.transpose(theo), tmp)

                # TODO beam is not defined here !
                if (self.has_beam_uncertainty):
                    for points in range(self.num_points):
                        beam[point] = self.beam_error[point]*theo[point]
                    tmp = np.dot(self.inv_covmat, beam)
                    chi2dd = np.dot(np.transpose(beam), tmp)
                    chi2pd = np.dot(np.transpose(theo), tmp)
                    chi2od = np.dot(np.transpose(difference), tmp)

                if (self.calib_uncertainty > 1.e-4):
                    wpp = 1/(chi2pp+1/self.calib_uncertainty**2)
                    chisq = chisq-wpp*chi2op**2
                    denom = denom/wpp*self.calib_uncertainty**2
                else:
                    wpp = 0

                if (self.has_beam_uncertainty):
                    wdd = 1/(chi2dd-wpp*chi2pd**2+1)
                    chisq = chisq-wdd*(chi2od-wpp*chi2op*chi2pd)**2
                    denom = denom/wdd

                chisq += math.log(denom)

        # finally, return ln(L)=-chi2/2

        self.lkl = -0.5 * chisq
        return self.lkl


###################################
# CLIK TYPE LIKELIHOOD
# --> clik_fake_planck,clik_wmap,etc.
###################################
class Likelihood_clik(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        self.need_cosmo_arguments(
            data, {'lensing': 'yes','output': 'tCl lCl pCl'})

        try:
            import clik
        except ImportError:
            raise io_mp.MissingLibraryError(
                "You must first activate the binaries from the Clik " +
                "distribution. Please run : \n " +
                "]$ source /path/to/clik/bin/clik_profile.sh \n " +
                "and try again.")
        # for lensing, some routines change. Intializing a flag for easier
        # testing of this condition
        #if self.name == 'Planck_lensing':
        if 'lensing' in self.name and 'Planck' in self.name:
            self.lensing = True
        else:
            self.lensing = False

        try:
            if self.lensing:
                self.clik = clik.clik_lensing(self.path_clik)
                try:
                    self.l_max = max(self.clik.get_lmax())
                # following 2 lines for compatibility with lensing likelihoods of 2013 and before
                # (then, clik.get_lmax() just returns an integer for lensing likelihoods;
                # this behavior was for clik versions < 10)
                except:
                    self.l_max = self.clik.get_lmax()
            else:
                self.clik = clik.clik(self.path_clik)
                self.l_max = max(self.clik.get_lmax())
        except clik.lkl.CError:
            raise io_mp.LikelihoodError(
                "The path to the .clik file for the likelihood "
                "%s was not found where indicated:\n%s\n"
                % (self.name,self.path_clik) +
                " Note that the default path to search for it is"
                " one directory above the path['clik'] field. You"
                " can change this behaviour in all the "
                "Planck_something.data, to reflect your local configuration, "
                "or alternatively, move your .clik files to this place.")
        except KeyError:
            raise io_mp.LikelihoodError(
                "In the %s.data file, the field 'clik' of the " % self.name +
                "path dictionary is expected to be defined. Please make sure"
                " it is the case in you configuration file")

        self.need_cosmo_arguments(
            data, {'l_max_scalars': self.l_max})

        self.nuisance = list(self.clik.extra_parameter_names)

        # line added to deal with a bug in planck likelihood release: A_planck called A_Planck in plik_lite
        if (self.name == 'Planck15_highl_lite') or (self.name == 'Planck15_highl_TTTEEE_lite'):
            for i in range(len(self.nuisance)):
                if (self.nuisance[i] == 'A_Planck'):
                    self.nuisance[i] = 'A_planck'
            print("In %s, MontePython corrected nuisance parameter name A_Planck to A_planck" % self.name)

        # testing if the nuisance parameters are defined. If there is at least
        # one non defined, raise an exception.
        exit_flag = False
        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])
        for nuisance in self.nuisance:
            if nuisance not in nuisance_parameter_names:
                exit_flag = True
                print('%20s\tmust be a fixed or varying nuisance parameter' % nuisance)

        if exit_flag:
            raise io_mp.LikelihoodError(
                "The likelihood %s " % self.name +
                "expected some nuisance parameters that were not provided")

        # deal with nuisance parameters
        try:
            self.use_nuisance
        except:
            self.use_nuisance = []

        # Add in use_nuisance all the parameters that have non-flat prior
        for nuisance in self.nuisance:
            if hasattr(self, '%s_prior_center' % nuisance):
                self.use_nuisance.append(nuisance)

    def loglkl(self, cosmo, data):

        nuisance_parameter_names = data.get_mcmc_parameters(['nuisance'])

        # get Cl's from the cosmological code
        cl = self.get_cl(cosmo)

        # testing for lensing
        if self.lensing:
            try:
                length = len(self.clik.get_lmax())
                tot = np.zeros(
                    np.sum(self.clik.get_lmax()) + length +
                    len(self.clik.get_extra_parameter_names()))
            # following 3 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                length = 2
                tot = np.zeros(2*self.l_max+length + len(self.clik.get_extra_parameter_names()))
        else:
            length = len(self.clik.get_has_cl())
            tot = np.zeros(
                np.sum(self.clik.get_lmax()) + length +
                len(self.clik.get_extra_parameter_names()))

        # fill with Cl's
        index = 0
        if not self.lensing:
            for i in range(length):
                if (self.clik.get_lmax()[i] > -1):
                    for j in range(self.clik.get_lmax()[i]+1):
                        if (i == 0):
                            tot[index+j] = cl['tt'][j]
                        if (i == 1):
                            tot[index+j] = cl['ee'][j]
                        if (i == 2):
                            tot[index+j] = cl['bb'][j]
                        if (i == 3):
                            tot[index+j] = cl['te'][j]
                        if (i == 4):
                            tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                        if (i == 5):
                            tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                    index += self.clik.get_lmax()[i]+1

        else:
            try:
                for i in range(length):
                    if (self.clik.get_lmax()[i] > -1):
                        for j in range(self.clik.get_lmax()[i]+1):
                            if (i == 0):
                                tot[index+j] = cl['pp'][j]
                            if (i == 1):
                                tot[index+j] = cl['tt'][j]
                            if (i == 2):
                                tot[index+j] = cl['ee'][j]
                            if (i == 3):
                                tot[index+j] = cl['bb'][j]
                            if (i == 4):
                                tot[index+j] = cl['te'][j]
                            if (i == 5):
                                tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                            if (i == 6):
                                tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                        index += self.clik.get_lmax()[i]+1

            # following 8 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                for i in range(length):
                    for j in range(self.l_max):
                        if (i == 0):
                            tot[index+j] = cl['pp'][j]
                        if (i == 1):
                            tot[index+j] = cl['tt'][j]
                    index += self.l_max+1

        # fill with nuisance parameters
        for nuisance in self.clik.get_extra_parameter_names():

            # line added to deal with a bug in planck likelihood release: A_planck called A_Planck in plik_lite
            if (self.name == 'Planck15_highl_lite') or (self.name == 'Planck15_highl_TTTEEE_lite'):
                if nuisance == 'A_Planck':
                    nuisance = 'A_planck'

            if nuisance in nuisance_parameter_names:
                nuisance_value = data.mcmc_parameters[nuisance]['current'] *\
                    data.mcmc_parameters[nuisance]['scale']
            else:
                raise io_mp.LikelihoodError(
                    "the likelihood needs a parameter %s. " % nuisance +
                    "You must pass it through the input file " +
                    "(as a free nuisance parameter or a fixed parameter)")
            #print("found one nuisance with name",nuisance)
            tot[index] = nuisance_value
            index += 1

        # compute likelihood
        #print("lkl:",self.clik(tot))
        lkl = self.clik(tot)[0]

        # do nuisance prior seperately, by Gen Ye
        return lkl

        # add prior on nuisance parameters
        lkl = self.add_nuisance_prior(lkl, data) 

        # Option added by D.C. Hooper to deal with the joint prior on ksz_norm (A_ksz in Planck notation)
        # and A_sz (A_tsz in Planck notation), of the form ksz_norm + 1.6 * A_sz (according to eq. 23 of 1907.12875).
        # Behaviour (True/False), centre, and variance set in the .data files (default = True).

        # Check if the joint prior has been requested
        if getattr(self, 'joint_sz_prior', False):

            # Check that the joint_sz prior is only requested when A_sz and ksz_norm are present
            if not ('A_sz' in self.clik.get_extra_parameter_names() and 'ksz_norm' in self.clik.get_extra_parameter_names()):
                 raise io_mp.LikelihoodError(
                    "You requested a gaussian prior on ksz_norm + 1.6 * A_sz," +
                    "however A_sz or ksz_norm are not present in your param file.")

            # Recover the current values of the two sz nuisance parameters
            A_sz =  data.mcmc_parameters['A_sz']['current'] * data.mcmc_parameters['A_sz']['scale']
            ksz_norm = data.mcmc_parameters['ksz_norm']['current'] * data.mcmc_parameters['ksz_norm']['scale']

            # Combine the two into one new nuisance-like variable
            joint_sz = ksz_norm + 1.6 * A_sz

            # Check if the user has passed the prior center and variance on sz, otherwise abort
            if not (hasattr(self, 'joint_sz_prior_center') and hasattr(self, 'joint_sz_prior_variance')):
                raise io_mp.LikelihoodError(
                    " You requested a gaussian prior on ksz_norm + 1.6 * A_sz," +
                    " however you did not pass the center and variance." +
                    " You can pass this in the .data file.")

            # add prior on joint_sz parameter
            if not self.joint_sz_prior_variance == 0:
                lkl += -0.5*((joint_sz-self.joint_sz_prior_center)/self.joint_sz_prior_variance)**2

            # End of block for joint sz prior.

        return lkl


###################################
# MOCK CMB TYPE LIKELIHOOD
# --> mock planck, cmbpol, etc.
###################################
class Likelihood_mock_cmb(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl lCl pCl'})

        ################
        # Noise spectrum
        ################

        try:
            self.noise_from_file
        except:
            self.noise_from_file = False

        if self.noise_from_file:

            try:
                self.noise_file
            except:
                raise io_mp.LikelihoodError("For reading noise from file, you must provide noise_file")

            self.noise_T = np.zeros(self.l_max+1, 'float64')
            self.noise_P = np.zeros(self.l_max+1, 'float64')
            if self.LensingExtraction:
                self.Nldd = np.zeros(self.l_max+1, 'float64')

            if os.path.exists(os.path.join(self.data_directory, self.noise_file)):
                noise = open(os.path.join(
                    self.data_directory, self.noise_file), 'r')
                line = noise.readline()
                while line.find('#') != -1:
                    line = noise.readline()

                for l in range(self.l_min, self.l_max+1):
                    ll = int(float(line.split()[0]))
                    if l != ll:
                        # if l_min is larger than the first l in the noise file we can skip lines
                        # until we are at the correct l. Otherwise raise error
                        while l > ll:
                            try:
                                line = noise.readline()
                                ll = int(float(line.split()[0]))
                            except:
                                raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the noise file")
                        if l < ll:
                            raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the noise file")
                    # read noise for C_l in muK**2
                    self.noise_T[l] = float(line.split()[1])
                    self.noise_P[l] = float(line.split()[2])
                    if self.LensingExtraction:
                        try:
                            # read noise for C_l^dd = l(l+1) C_l^pp
                            self.Nldd[l] = float(line.split()[3])/(l*(l+1)/2./math.pi)
                        except:
                            raise io_mp.LikelihoodError("For reading lensing noise from file, you must provide one more column")
                    line = noise.readline()
            else:
                raise io_mp.LikelihoodError("Could not find file ",self.noise_file)


        else:
            # convert arcmin to radians
            self.theta_fwhm *= np.array([math.pi/60/180])
            self.sigma_T *= np.array([math.pi/60/180])
            self.sigma_P *= np.array([math.pi/60/180])

            # compute noise in muK**2
            self.noise_T = np.zeros(self.l_max+1, 'float64')
            self.noise_P = np.zeros(self.l_max+1, 'float64')

            for l in range(self.l_min, self.l_max+1):
                self.noise_T[l] = 0
                self.noise_P[l] = 0
                for channel in range(self.num_channels):
                    self.noise_T[l] += self.sigma_T[channel]**-2 *\
                                       math.exp(
                                           -l*(l+1)*self.theta_fwhm[channel]**2/8/math.log(2))
                    self.noise_P[l] += self.sigma_P[channel]**-2 *\
                                       math.exp(
                                           -l*(l+1)*self.theta_fwhm[channel]**2/8/math.log(2))
                self.noise_T[l] = 1/self.noise_T[l]
                self.noise_P[l] = 1/self.noise_P[l]


        # trick to remove any information from polarisation for l<30
        try:
            self.no_small_l_pol
        except:
            self.no_small_l_pol = False

        if self.no_small_l_pol:
            for l in range(self.l_min,30):
                # plug a noise level of 100 muK**2, equivalent to no detection at all of polarisation
                self.noise_P[l] = 100.

        # trick to remove any information from temperature above l_max_TT
        try:
            self.l_max_TT
        except:
            self.l_max_TT = False

        if self.l_max_TT:
            for l in range(self.l_max_TT+1,l_max+1):
                # plug a noise level of 100 muK**2, equivalent to no detection at all of temperature
                self.noise_T[l] = 100.

        # impose that the cosmological code computes Cl's up to maximum l
        # needed by the window function
        self.need_cosmo_arguments(data, {'l_max_scalars': self.l_max})

        # if you want to print the noise spectra:
        #test = open('noise_T_P','w')
        #for l in range(self.l_min, self.l_max+1):
        #    test.write('%d  %e  %e\n'%(l,self.noise_T[l],self.noise_P[l]))

        ###########################################################################
        # implementation of default settings for flags describing the likelihood: #
        ###########################################################################

        # - ignore B modes by default:
        try:
            self.Bmodes
        except:
            self.Bmodes = False
        # - do not use delensing by default:
        try:
            self.delensing
        except:
            self.delensing = False
        # - do not include lensing extraction by default:
        try:
            self.LensingExtraction
        except:
            self.LensingExtraction = False
        # - neglect TD correlation by default:
        try:
            self.neglect_TD
        except:
            self.neglect_TD = True
        # - use lthe lensed TT, TE, EE by default:
        try:
            self.unlensed_clTTTEEE
        except:
            self.unlensed_clTTTEEE = False
        # - do not exclude TTEE by default:
        try:
            self.ExcludeTTTEEE
            if self.ExcludeTTTEEE and not self.LensingExtraction:
                raise io_mp.LikelihoodError("Mock CMB likelihoods where TTTEEE is not used have only been "
                                            "implemented for the deflection spectrum (i.e. not for B-modes), "
                                            "but you do not seem to have lensing extraction enabled")
        except:
            self.ExcludeTTTEEE = False

	#added by Siavash Yasini
        try:
            self.OnlyTT
            if self.OnlyTT and self.ExcludeTTTEEE:
                raise io_mp.LikelihoodError("OnlyTT and ExcludeTTTEEE cannot be used simultaneously.")
        except:
            self.OnlyTT = False

        ##############################################
        # Delensing noise: implemented by  S. Clesse #
        ##############################################

        if self.delensing:

            try:
                self.delensing_file
            except:
                raise io_mp.LikelihoodError("For delensing, you must provide delensing_file")

            self.noise_delensing = np.zeros(self.l_max+1)
            if os.path.exists(os.path.join(self.data_directory, self.delensing_file)):
                delensing_file = open(os.path.join(
                    self.data_directory, self.delensing_file), 'r')
                line = delensing_file.readline()
                while line.find('#') != -1:
                    line = delensing_file.readline()

                for l in range(self.l_min, self.l_max+1):
                    ll = int(float(line.split()[0]))
                    if l != ll:
                        # if l_min is larger than the first l in the delensing file we can skip lines
                        # until we are at the correct l. Otherwise raise error
                        while l > ll:
                            try:
                                line = delensing_file.readline()
                                ll = int(float(line.split()[0]))
                            except:
                                raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                        if l < ll:
                            raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                    self.noise_delensing[ll] = float(line.split()[2])/(ll*(ll+1)/2./math.pi)
                    # change 3 to 4 in the above line for CMBxCIB delensing
                    line = delensing_file.readline()

            else:
                raise io_mp.LikelihoodError("Could not find file ",self.delensing_file)

        ###############################################################
        # Read data for TT, EE, TE, [eventually BB or phi-phi, phi-T] #
        ###############################################################

        # default:
        if not self.ExcludeTTTEEE:
            numCls = 3

        # default 0 if excluding TT EE
        else:
            numCls = 0

        # deal with BB:
        if self.Bmodes:
            self.index_B = numCls
            numCls += 1

        # deal with pp, pT (p = CMB lensing potential):
        if self.LensingExtraction:
            self.index_pp = numCls
            numCls += 1
            if not self.ExcludeTTTEEE:
                self.index_tp = numCls
                numCls += 1

            if not self.noise_from_file:
                # provide a file containing NlDD (noise for the extracted
                # deflection field spectrum) This option is temporary
                # because at some point this module will compute NlDD
                # itself, when logging the fiducial model spectrum.
                try:
                    self.temporary_Nldd_file
                except:
                    raise io_mp.LikelihoodError("For lensing extraction, you must provide a temporary_Nldd_file")

                # read the NlDD file
                self.Nldd = np.zeros(self.l_max+1, 'float64')

                if os.path.exists(os.path.join(self.data_directory, self.temporary_Nldd_file)):
                    fid_file = open(os.path.join(self.data_directory, self.temporary_Nldd_file), 'r')
                    line = fid_file.readline()
                    while line.find('#') != -1:
                        line = fid_file.readline()
                    while (line.find('\n') != -1 and len(line) == 1):
                        line = fid_file.readline()
                    for l in range(self.l_min, self.l_max+1):
                        ll = int(float(line.split()[0]))
                        if l != ll:
                            # if l_min is larger than the first l in the delensing file we can skip lines
                            # until we are at the correct l. Otherwise raise error
                            while l > ll:
                                try:
                                    line = fid_file.readline()
                                    ll = int(float(line.split()[0]))
                                except:
                                    raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                            if l < ll:
                                raise io_mp.LikelihoodError("Mismatch between required values of l in the code and in the delensing file")
                        # this lines assumes that Nldd is stored in the
                        # 4th column (can be customised)
                        self.Nldd[ll] = float(line.split()[3])/(l*(l+1.)/2./math.pi)
                        line = fid_file.readline()
                else:
                    raise io_mp.LikelihoodError("Could not find file ",self.temporary_Nldd_file)

        # deal with fiducial model:
        # If the file exists, initialize the fiducial values
        self.Cl_fid = np.zeros((numCls, self.l_max+1), 'float64')
        self.fid_values_exist = False
        if os.path.exists(os.path.join(
                self.data_directory, self.fiducial_file)):
            self.fid_values_exist = True
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'r')
            line = fid_file.readline()
            while line.find('#') != -1:
                line = fid_file.readline()
            while (line.find('\n') != -1 and len(line) == 1):
                line = fid_file.readline()
            for l in range(self.l_min, self.l_max+1):
                ll = int(line.split()[0])
                if not self.ExcludeTTTEEE:
                    self.Cl_fid[0, ll] = float(line.split()[1])
                    self.Cl_fid[1, ll] = float(line.split()[2])
                    self.Cl_fid[2, ll] = float(line.split()[3])
                # read BB:
                if self.Bmodes:
                    try:
                        self.Cl_fid[self.index_B, ll] = float(line.split()[self.index_B+1])
                    except:
                        raise io_mp.LikelihoodError(
                            "The fiducial model does not have enough columns.")
                # read DD, TD (D = deflection field):
                if self.LensingExtraction:
                    try:
                        self.Cl_fid[self.index_pp, ll] = float(line.split()[self.index_pp+1])
                        if not self.ExcludeTTTEEE:
                            self.Cl_fid[self.index_tp, ll] = float(line.split()[self.index_tp+1])
                    except:
                        raise io_mp.LikelihoodError(
                            "The fiducial model does not have enough columns.")

                line = fid_file.readline()

        # Else the file will be created in the loglkl() function.

        # Explicitly display the flags to be sure that likelihood does what you expect:
        print("Initialised likelihood_mock_cmb with following options:")
        if self.unlensed_clTTTEEE:
            print("  unlensed_clTTTEEE is True")
        else:
            print("  unlensed_clTTTEEE is False")
        if self.Bmodes:
            print("  Bmodes is True")
        else:
            print("  Bmodes is False")
        if self.delensing:
            print("  delensing is True")
        else:
            print("  delensing is False")
        if self.LensingExtraction:
            print("  LensingExtraction is True")
        else:
            print("  LensingExtraction is False")
        if self.neglect_TD:
            print("  neglect_TD is True")
        else:
            print("  neglect_TD is False")
        if self.ExcludeTTTEEE:
            print("  ExcludeTTTEEE is True")
        else:
            print("  ExcludeTTTEEE is False")
        if self.OnlyTT:
            print("  OnlyTT is True")
        else:
            print("  OnlyTT is False")
        print("")

        # end of initialisation
        return

    def loglkl(self, cosmo, data):

        # get Cl's from the cosmological code (returned in muK**2 units)

        # if we want unlensed Cl's
        if self.unlensed_clTTTEEE:
            cl = self.get_unlensed_cl(cosmo)
            # exception: for non-delensed B modes we need the lensed BB spectrum
            # (this case is usually not useful/relevant)
            if self.Bmodes and (not self.delensing):
                    cl_lensed = self.get_cl(cosmo)
                    for l in range(self.lmax+1):
                        cl[l]['bb']=cl_lensed[l]['bb']

        # if we want lensed Cl's
        else:
            cl = self.get_cl(cosmo)
            # exception: for delensed B modes we need the unlensed spectrum
            if self.Bmodes and self.delensing:
                cl_unlensed = self.get_unlensed_cl(cosmo)
                for l in range(self.lmax+1):
                        cl[l]['bb']=cl_unlensed[l]['bb']

        # get likelihood
        lkl = self.compute_lkl(cl, cosmo, data)

        return lkl

    def compute_lkl(self, cl, cosmo, data):

        # Write fiducial model spectra if needed (return an imaginary number in
        # that case)
        if self.fid_values_exist is False:
            # Store the values now.
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'w')
            fid_file.write('# Fiducial parameters')
            for key, value in dictitems(data.mcmc_parameters):
                fid_file.write(', %s = %.5g' % (
                    key, value['current']*value['scale']))
            fid_file.write('\n')
            for l in range(self.l_min, self.l_max+1):
                fid_file.write("%5d  " % l)
                if not self.ExcludeTTTEEE:
                    fid_file.write("%.8g  " % (cl['tt'][l]+self.noise_T[l]))
                    fid_file.write("%.8g  " % (cl['ee'][l]+self.noise_P[l]))
                    fid_file.write("%.8g  " % cl['te'][l])
                if self.Bmodes:
                    # next three lines added by S. Clesse for delensing
                    if self.delensing:
                        fid_file.write("%.8g  " % (cl['bb'][l]+self.noise_P[l]+self.noise_delensing[l]))
                    else:
                        fid_file.write("%.8g  " % (cl['bb'][l]+self.noise_P[l]))
                if self.LensingExtraction:
                    # we want to store clDD = l(l+1) clpp
                    # and ClTD = sqrt(l(l+1)) Cltp
                    fid_file.write("%.8g  " % (l*(l+1.)*cl['pp'][l] + self.Nldd[l]))
                    if not self.ExcludeTTTEEE:
                        fid_file.write("%.8g  " % (math.sqrt(l*(l+1.))*cl['tp'][l]))
                fid_file.write("\n")
            print('\n')
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        # compute likelihood

        chi2 = 0

        # count number of modes.
        # number of modes is different form number of spectra
        # modes = T,E,[B],[D=deflection]
        # spectra = TT,EE,TE,[BB],[DD,TD]
        # default:
        if not self.ExcludeTTTEEE:
            if self.OnlyTT:
                num_modes=1
            else:
                num_modes=2
        # default 0 if excluding TT EE
        else:
            num_modes=0
        # add B mode:
        if self.Bmodes:
            num_modes += 1
        # add D mode:
        if self.LensingExtraction:
            num_modes += 1

        Cov_obs = np.zeros((num_modes, num_modes), 'float64')
        Cov_the = np.zeros((num_modes, num_modes), 'float64')
        Cov_mix = np.zeros((num_modes, num_modes), 'float64')

        for l in range(self.l_min, self.l_max+1):

            if self.Bmodes and self.LensingExtraction:
                raise io_mp.LikelihoodError("We have implemented a version of the likelihood with B modes, a version with lensing extraction, but not yet a version with both at the same time. You can implement it.")

            # case with B modes:
            elif self.Bmodes:
                Cov_obs = np.array([
                    [self.Cl_fid[0, l], self.Cl_fid[2, l], 0],
                    [self.Cl_fid[2, l], self.Cl_fid[1, l], 0],
                    [0, 0, self.Cl_fid[3, l]]])
                # next 5 lines added by S. Clesse for delensing
                if self.delensing:
                    Cov_the = np.array([
                        [cl['tt'][l]+self.noise_T[l], cl['te'][l], 0],
                        [cl['te'][l], cl['ee'][l]+self.noise_P[l], 0],
                        [0, 0, cl['bb'][l]+self.noise_P[l]+self.noise_delensing[l]]])
                else:
                    Cov_the = np.array([
                        [cl['tt'][l]+self.noise_T[l], cl['te'][l], 0],
                        [cl['te'][l], cl['ee'][l]+self.noise_P[l], 0],
                        [0, 0, cl['bb'][l]+self.noise_P[l]]])

            # case with lensing
            # note that the likelihood is based on ClDD (deflection spectrum)
            # rather than Clpp (lensing potential spectrum)
            # But the Bolztmann code input is Clpp
            # So we make the conversion using ClDD = l*(l+1.)*Clpp
            # So we make the conversion using ClTD = sqrt(l*(l+1.))*Cltp

            # just DD, i.e. no TT or EE.
            elif self.LensingExtraction and self.ExcludeTTTEEE:
                cldd_fid = self.Cl_fid[self.index_pp, l]
                cldd = l*(l+1.)*cl['pp'][l]
                Cov_obs = np.array([[cldd_fid]])
                Cov_the = np.array([[cldd+self.Nldd[l]]])

            # Usual TTTEEE plus DD and TD
            elif self.LensingExtraction:
                cldd_fid = self.Cl_fid[self.index_pp, l]
                cldd = l*(l+1.)*cl['pp'][l]
                if self.neglect_TD:
                    cltd_fid = 0.
                    cltd = 0.
                else:
                    cltd_fid = self.Cl_fid[self.index_tp, l]
                    cltd = math.sqrt(l*(l+1.))*cl['tp'][l]

                Cov_obs = np.array([
                    [self.Cl_fid[0, l], self.Cl_fid[2, l], 0.*self.Cl_fid[self.index_tp, l]],
                    [self.Cl_fid[2, l], self.Cl_fid[1, l], 0],
                    [cltd_fid, 0, cldd_fid]])
                Cov_the = np.array([
                    [cl['tt'][l]+self.noise_T[l], cl['te'][l], 0.*math.sqrt(l*(l+1.))*cl['tp'][l]],
                    [cl['te'][l], cl['ee'][l]+self.noise_P[l], 0],
                    [cltd, 0, cldd+self.Nldd[l]]])

	    # case with TT only (Added by Siavash Yasini)
            elif self.OnlyTT:
                Cov_obs = np.array([[self.Cl_fid[0, l]]])

                Cov_the = np.array([[cl['tt'][l]+self.noise_T[l]]])


            # case without B modes nor lensing:
            else:
                Cov_obs = np.array([
                    [self.Cl_fid[0, l], self.Cl_fid[2, l]],
                    [self.Cl_fid[2, l], self.Cl_fid[1, l]]])
                Cov_the = np.array([
                    [cl['tt'][l]+self.noise_T[l], cl['te'][l]],
                    [cl['te'][l], cl['ee'][l]+self.noise_P[l]]])

            # get determinant of observational and theoretical covariance matrices
            det_obs = np.linalg.det(Cov_obs)
            det_the = np.linalg.det(Cov_the)

            # get determinant of mixed matrix (= sum of N theoretical
            # matrices with, in each of them, the nth column replaced
            # by that of the observational matrix)
            det_mix = 0.
            for i in range(num_modes):
                Cov_mix = np.copy(Cov_the)
                Cov_mix[:, i] = Cov_obs[:, i]
                det_mix += np.linalg.det(Cov_mix)

            chi2 += (2.*l+1.)*self.f_sky *\
                (det_mix/det_the + math.log(det_the/det_obs) - num_modes)

        return -chi2/2


########################################
# Spectral Distortions TYPE LIKELIHOOD
# --> mock PIXIE, FIRAS, ...
# Implemented by D.C. Hooper, M. Lucca
# and N. Schoeneberg. Implementation
# described in 1910.04619 (general) and
# 2010.07814 (foregrounds)
########################################
class Likelihood_sd(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Tell CLASS that for sure we are going to need SDs
        self.need_cosmo_arguments(
            data, {'output': 'sd','modes':'s' })

        ################
        # Noise spectrum
        ################

        """The user can either pass an external noise file (as done for FIRAS), or compute it here automatically (as done for PIXIE)."""

        """If no noise file is provided, we will need the following information from the .data file:
           detector_nu_min, detector_nu_max, detector_nu_delta, detector_bin_number, detector_delta_Ic"""


        try:
            self.noise_from_file
        except:
            self.noise_from_file = False

        # Read noise from file
        if self.noise_from_file:
            try:
                self.noise_file_name
                self.noise_file_directory
            except:
                raise io_mp.LikelihoodError("If you want to read the noise from a file (according to your 'noise_from_file' option passed to the SD likelihood, you need to also add a 'noise_file_name' and 'noise_file_directory'.")

            # Warn the user if they are doing something wrong
            if hasattr(self, 'detector_bin_number') or hasattr(self, 'detector_nu_delta') or hasattr(self, 'detector_nu_min') or hasattr(self, 'detector_nu_max'):
                warnings.warn(' Warning! You asked to read the noise from a file, but you also passed detector spcifications.')
                warnings.warn(' I will ignore the bin_number, nu_delta, nu_min, and nu_max you passed in the likelihood file, and calculate them from the noise file.')

            # This is because CLASS only needs the name, but knows the directory, while here we need to link it properly
            self.noise_file = os.path.join(self.noise_file_directory, self.noise_file_name)

            if os.path.exists(self.noise_file):
                with open(self.noise_file, 'r') as noise:
                    nu_from_file = []
                    noise_from_file = []

                    # The first line contains information needed for CLASS, but not here. Skip header and first line
                    line = noise.readline()
                    while line.find('#') != -1:
                        line = noise.readline()

                    for line in noise:
                        # Get frequency from detector settings
                        nu_from_file.append(float(line.split()[0]))
                        noise_from_file.append(float(line.split()[1]))

                # Get number of bins, nu_min, and nu_max from file
                self.detector_bin_number = len(nu_from_file)
                self.detector_nu_min = nu_from_file[0]
                self.detector_nu_max = nu_from_file[-1]

                # Pass the two arrays for the rest of the likelihood
                self.nu_range = np.array(nu_from_file, 'float64')
                self.noise_Ic = np.array(noise_from_file, 'float64')

            else:
                raise io_mp.LikelihoodError("Could not find file "+str(self.noise_file))

        # Compute noise for mission based on detector specifications
        else:

            # Compute noise (in Jy/ sr). For now, for a PIXIE_like detector we use the same noise in every bin,
            # in the future we will add more options

            if hasattr(self, 'detector_bin_number') and hasattr(self, 'detector_nu_delta'):
               # If user passed both, check that they are consistent
                bin_number = int(round((self.detector_nu_max-self.detector_nu_min)/float(self.detector_nu_delta)))
                if (self.detector_bin_number != bin_number):
                    raise io_mp.LikelihoodError("You requested %d bins, with a bin width of %d. From your min_nu, max_nu, and bin width I get %d bins Aborting." \
                                                % (self.detector_bin_number, self.detector_nu_delta, bin_number ))

            if not hasattr(self, 'detector_bin_number'):
                self.detector_bin_number = int(round((self.detector_nu_max-self.detector_nu_min)/float(self.detector_nu_delta)))

            print('Computing noise for %d bins', self.detector_bin_number)

            self.noise_Ic = np.zeros(self.detector_bin_number, 'float64')
            self.nu_range = np.zeros(self.detector_bin_number, 'float64')

            for nu_i in range(self.detector_bin_number):
                self.noise_Ic[nu_i] = self.detector_delta_Ic
                self.nu_range[nu_i] = self.detector_nu_min + self.detector_nu_delta*nu_i

        # Now we pass things to CLASS. CLASS will take either detctor name and noise file, or detector name and specifications
        if self.noise_from_file:
            self.need_cosmo_arguments(data, {'sd_detector_name':self.detector})
            self.need_cosmo_arguments(data, {'sd_detector_file':self.noise_file_name})

        else:
            self.need_cosmo_arguments(data, {'sd_detector_name':self.detector})
            self.need_cosmo_arguments(data, {'sd_detector_nu_min': self.detector_nu_min, 'sd_detector_nu_max': self.detector_nu_max})
            self.need_cosmo_arguments(data, {'sd_detector_nu_delta': self.detector_nu_delta, 'sd_detector_delta_Ic': self.detector_delta_Ic})


        # Deal with fiducial model
        # If the file exists, initialize the fiducial values
        self.Ic_fid = np.zeros(self.detector_bin_number, 'float64')
        self.fid_values_exist = False
        if os.path.exists(os.path.join(
                self.data_directory, self.fiducial_file)):
            self.fid_values_exist = True
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'r')
            line = fid_file.readline()
            while line.find('#') != -1:
                line = fid_file.readline()
            while (line.find('\n') != -1 and len(line) == 1):
                line = fid_file.readline()
            for nu_i in range(self.detector_bin_number):
                self.Ic_fid[nu_i] = float(line.split()[1])
                line = fid_file.readline()

        # Else the file will be created in the loglkl() function

        # Load foreground templates:
        self.spinning_dust_file = "spectral_templates/SpinningDustTemplate.dat"
        self.spinning_dust_lognup_data = np.log(31.)
        self.spinning_dust_lognup_0 = np.log(30.)
        self.spinning_dust_lognu_0 = np.log(22.8)
        spinning_dust_lognu_data = []
        spinning_dust_logT_brightness_data = []
        with open(os.path.join(self.data_directory, self.spinning_dust_file), 'r') as sd_file:
            line = sd_file.readline()
            while line.find('#') != -1:
                line = sd_file.readline()
            while (line.find('\n') != -1 and len(line) == 1):
                line = sd_file.readline()
            while line:
                nu = float(line.split(",")[0])
                T_brightness = float(line.split(",")[1])
                spinning_dust_lognu_data.append(np.log(nu))
                spinning_dust_logT_brightness_data.append(np.log(T_brightness))
                line = sd_file.readline()
        spinning_dust_lognu_data = np.array(spinning_dust_lognu_data)
        spinning_dust_logT_brightness_data = np.array(spinning_dust_logT_brightness_data)
        self.spinning_dust_lognumin = spinning_dust_lognu_data[0]
        self.spinning_dust_lognumax = spinning_dust_lognu_data[-1]
        self.spinning_dust_logT_brightness = scipy.interpolate.CubicSpline(spinning_dust_lognu_data,spinning_dust_logT_brightness_data)

        self.co_integrated_file = "spectral_templates/COintegratedTemplate.dat"
        co_integrated_lognu_data = []
        co_integrated_logInu_data = []
        with open(os.path.join(self.data_directory, self.co_integrated_file), 'r') as co_file:
            line = co_file.readline()
            while line.find('#') != -1:
                line = co_file.readline()
            while (line.find('\n') != -1 and len(line) == 1):
                line = co_file.readline()
            while line:
                nu = float(line.split(",")[0])
                Inu = float(line.split(",")[1])
                co_integrated_lognu_data.append(np.log(nu))
                co_integrated_logInu_data.append(np.log(Inu))
                line = co_file.readline()
        co_integrated_lognu_data = np.array(co_integrated_lognu_data)
        co_integrated_logInu_data = np.array(co_integrated_logInu_data)
        self.co_integrated_lognu_min = co_integrated_lognu_data[0]
        self.co_integrated_lognu_max = co_integrated_lognu_data[-1]
        self.co_integrated_logInu = scipy.interpolate.CubicSpline(co_integrated_lognu_data,co_integrated_logInu_data)

        # End of initialisation
        return

    def eval_spinning_dust(self, lognu,lognu_p):
        # Define local quantities and range check them (outside of these values T_brightness < 0.0001*T_brightnes_max)
        lognu_tilde = lognu+self.spinning_dust_lognup_data-lognu_p
        lognu0_tilde = self.spinning_dust_lognu_0+self.spinning_dust_lognup_data-lognu_p

        if(lognu_tilde < self.spinning_dust_lognumin or lognu0_tilde < self.spinning_dust_lognumin):
          return 0.
        if(lognu_tilde > self.spinning_dust_lognumax or lognu0_tilde > self.spinning_dust_lognumax):
          return 0.

        # Calculate T_brightness*nu^2 as a proxy for I (indeed, since we are using differences of logs, the final result IS the intensity)
        log_T_brightness_times_nu2 = self.spinning_dust_logT_brightness(lognu_tilde) - self.spinning_dust_logT_brightness(lognu0_tilde) + 2 * (lognu_tilde-lognu0_tilde)

        # What we calculated is directly the delta I(nu)/delta I(nu_ref)
        return np.exp(log_T_brightness_times_nu2)

    def eval_co_integrated(self, lognu):
        if(lognu < self.co_integrated_lognu_min or lognu > self.co_integrated_lognu_max):
          return 0.

        # Calculate logI directly
        logInu = self.co_integrated_logInu(lognu)

        return np.exp(logInu)

    def loglkl(self, cosmo, data):
        # Get SDs from CLASS (returned in Jy/sr = 10^(-26)W/m^2/sr/Hz units)
        sd = cosmo.spectral_distortion()

        # Get likelihood
        lkl = self.compute_lkl(sd, cosmo, data)

        return lkl

    def compute_lkl(self, sd, cosmo, data):
        sd_nu = sd[0]   # In GHz
        sd_amp = sd[1]  # In Jy/sr = 10^(-26)W/m^2/sr/Hz

        # Define constants
        const_h = 6.62607004e-34
        const_k = 1.38064852e-23
        const_c = 299792458

        # Define useful quantities
        nu = self.nu_range       # In GHz
        T_cmb = cosmo.T_cmb()    # In Kelvin

        # Define dimensionless frequency and overall normalization
        x = (nu*1e9*const_h)/(T_cmb*const_k)
        normalisation = 2 *(T_cmb*const_k)**3/(const_h*const_c)**2*1.e26 # In Jy/sr (About 270MJy)

        # Define distortion shapes
        alpha_mu = 0.45614425920673529                             # Should be precise enough... (1/3 * zeta(2)/zeta(3))
        g_shape = normalisation*x**4*np.exp(-x)/(1.-np.exp(-x))**2 # In Jy/sr
        mu_shape = g_shape*(alpha_mu-1./x)                         # In Jy/sr
        y_shape = g_shape*(x*(1.+np.exp(-x))/(1.-np.exp(-x))-4.)   # In Jy/sr

        # Calculate marginilizations (gaussian prior chi2, and intensity of corrections)
        chi2_prior = 0.
        Ic_corr = np.zeros((9,self.detector_bin_number)) # 1 dT + 8 others

        # The following is adding marginilizations also over other parameters
        # 0) Temperature shift
        delta_T = data.mcmc_parameters['sd_delta_T']['current'] * data.mcmc_parameters['sd_delta_T']['scale']
        # This equation is taken from Chluba+2014 [1306.5751], see also Lucca+2019 [1910.04619]
        Ic_corr[0] = delta_T*(1.+delta_T)*g_shape + delta_T**2/2.*y_shape # In Jy/sr
        # Add constraints
        chi2_prior += (delta_T-self.sd_delta_T_prior_center)**2/self.sd_delta_T_prior_sigma**2

        # 1) Thermal dust
        T_D = data.mcmc_parameters['sd_T_D']['current'] * data.mcmc_parameters['sd_T_D']['scale'] # In Kelvin
        beta_D = data.mcmc_parameters['sd_beta_D']['current'] * data.mcmc_parameters['sd_beta_D']['scale']
        A_D = data.mcmc_parameters['sd_A_D']['current'] * data.mcmc_parameters['sd_A_D']['scale'] # In Jy/sr
        # The equation is taken from Tab. 4 of Planck 2015 results X [1502.01588v2]
        nu_D_ref = 545. # In GHz
        x_D = (nu*10.**9.*const_h)/(T_D*const_k)
        x_D_ref = (nu_D_ref*10.**9.*const_h)/(T_D*const_k)
        dust_shape = ((x_D/x_D_ref)**(beta_D+3.))*(np.exp(x_D_ref)-1.)/(np.exp(x_D)-1.)
        Ic_corr[1] = A_D*dust_shape
        # Add constraints
        chi2_prior += (T_D-self.sd_T_D_prior_center)**2./self.sd_T_D_prior_sigma**2.
        chi2_prior += (beta_D-self.sd_beta_D_prior_center)**2./self.sd_beta_D_prior_sigma**2.
        if(A_D < 0.):
          return data.boundary_loglike

        # 2) Cosmic Infrared Background (CIB)
        T_C = data.mcmc_parameters['sd_T_CIB']['current'] * data.mcmc_parameters['sd_T_CIB']['scale'] # In Kelvin
        beta_C = data.mcmc_parameters['sd_beta_CIB']['current'] * data.mcmc_parameters['sd_beta_CIB']['scale']
        A_C = data.mcmc_parameters['sd_A_CIB']['current'] * data.mcmc_parameters['sd_A_CIB']['scale'] # In Jy/sr
        # The equation is taken from Tab. 1 of Abitbol+2016 [1705.01534]
        # (with the inclusion of a reference frequency, which is set as in the case of the thermal dust)
        nu_C_ref = 545. # In GHz
        x_C = (nu*10.**9.*const_h)/(T_C*const_k)
        x_C_ref = (nu_C_ref*10.**9.*const_h)/(T_C*const_k)
        cib_shape = ((x_C/x_C_ref)**(beta_C+3.))*(np.exp(x_C_ref)-1.)/(np.exp(x_C)-1.)
        Ic_corr[2] = A_C*cib_shape
        # Add constraints
        chi2_prior += (T_C-self.sd_T_C_prior_center)**2./self.sd_T_C_prior_sigma**2.
        chi2_prior += (beta_C-self.sd_beta_C_prior_center)**2./self.sd_beta_C_prior_sigma**2.
        if(A_C < 0.):
          return data.boundary_loglike

        # 3) Synchrotron radiation
        alpha_S = data.mcmc_parameters['sd_alpha_sync']['current'] * data.mcmc_parameters['sd_alpha_sync']['scale']
        omega_S = data.mcmc_parameters['sd_omega_sync']['current'] * data.mcmc_parameters['sd_omega_sync']['scale']
        A_S = data.mcmc_parameters['sd_A_sync']['current'] * data.mcmc_parameters['sd_A_sync']['scale'] # In Jy/sr
        # The equation is taken from Tab. 1 of Abitbol+2016 [1705.01534]
        nu_S_ref = 100. # In GHz
        sync_shape = (nu_S_ref/nu)**alpha_S*(1.+0.5*omega_S*np.log(nu/nu_S_ref)**2.)
        Ic_corr[3] = A_S*sync_shape
        # Add constraints
        chi2_prior += (alpha_S-self.sd_alpha_S_prior_center)**2./self.sd_alpha_S_prior_sigma**2.
        chi2_prior += (A_S-self.sd_A_S_prior_center)**2./self.sd_A_S_prior_sigma**2.
        if(A_S < 0.):
          return data.boundary_loglike

        # 4) Free free dust emission
        # (For the definition of 'EM' compare to Draine+2011 [ISBN: 978-0-691-12214-4, also see astro-ph/9710152v2],
        # and Tab 4 of Planck 2015 results X [1502.01588v2].
        # Additionally, see Mukherjee+2019 [1910.02132], where they set EM=1, and instead parametrise with the use of an amplitude)
        T_e = data.mcmc_parameters['sd_T_e']['current'] * data.mcmc_parameters['sd_T_e']['scale'] # in Kelvin
        EM = data.mcmc_parameters['sd_EM']['current'] * data.mcmc_parameters['sd_EM']['scale'] # in pc/cm^6
        # The equation is taken from Tab 4 of Planck 2015 results X [1502.01588v2], but converted from brightness temperature to intensity
        g_ff = np.log(np.exp(1.)+np.exp(5.96-np.sqrt(3.)/np.pi*np.log(nu*(T_e/10.**4.)**(-1.5))))
        tau_ff = 0.05468*EM*T_e**(-1.5)*nu**(-2.)*g_ff # 0.05468 * EM /(pc*cm^(-6)) * (T_e/Kelvin)**(-1.5)*(nu/GHz)**(-2)*g_ff
        nu_ff_ref = 545. # In GHz
        # Factor from Planck 2015 brightness temperature in Kelvin to Jy/sr
        Tb_to_Inu_factor = (2.*const_k*(nu*10**9)**2/const_c**2*1e26)
        ff_shape = Tb_to_Inu_factor*(T_e*(1.-np.exp(-tau_ff))) # In Jy/sr
        # Since EM is perfectly degerenate with the amplitude, so we can manually fix it to 1
        A_ff = 1.
        Ic_corr[4] = A_ff*ff_shape
        # Add constraints
        chi2_prior += (T_e-self.sd_T_e_prior_center)**2./self.sd_T_e_prior_sigma**2.
        chi2_prior += (EM-self.sd_EM_prior_center)**2./self.sd_EM_prior_sigma**2.
        if(A_ff < 0.):
          return data.boundary_loglike

        # 5) Spinning dust emission
        nu_p_sd = data.mcmc_parameters['sd_nu_p_spin']['current'] * data.mcmc_parameters['sd_nu_p_spin']['scale'] # in GHz
        A_spin = data.mcmc_parameters['sd_A_spin']['current'] * data.mcmc_parameters['sd_A_spin']['scale'] # in Jy/sr
        # The equation is taken from Tab. 4 of Planck 2015 results X [1502.01588v2], but converted from brightness temperature to intensity
        spin_shape = np.array([self.eval_spinning_dust(np.log(nuval),np.log(nu_p_sd)) for nuval in nu])
        Ic_corr[5] = A_spin*spin_shape
        # Add constraints
        chi2_prior += (nu_p_sd-self.sd_nu_p_sd_prior_center)**2./self.sd_nu_p_sd_prior_sigma**2.
        if(A_spin < 0.):
          return data.boundary_loglike

        # 6) Integrated CO emission
        A_CO = data.mcmc_parameters['sd_A_CO']['current'] * data.mcmc_parameters['sd_A_CO']['scale'] # Dimensionless
        # The equation is taken from Tab. 1 of Abitbol+2016 [1705.01534]
        CO_shape = np.array([self.eval_co_integrated(np.log(nuval)) for nuval in nu]) # In Jy/sr
        Ic_corr[6] = A_CO*CO_shape
        # Add constraints
        if(A_CO < 0.):
          return data.boundary_loglike

        # 7) Reionization
        y_reio = data.mcmc_parameters['sd_y_reio_nuisance']['current'] * data.mcmc_parameters['sd_y_reio_nuisance']['scale'] # Dimensionless
        # The equation is taken from Chluba+2014 [1306.5751], see also Lucca+2019 [1910.04619]
        Ic_corr[7] = y_reio*y_shape
        # Add constraints
        chi2_prior += (y_reio-self.sd_y_reio_sd_prior_center)**2./self.sd_y_reio_sd_prior_sigma**2.

        # 8) Mu parameter (with this option we can test the sensitivity to the mu parameter, intended for testing purposes only)
        #    Add this parameter to your 'use_nuisance' in the corresponding .data file, if you want to use it
        if 'sd_mu_nuisance' in data.mcmc_parameters:
          mu_nuis = data.mcmc_parameters['sd_mu_nuisance']['current'] * data.mcmc_parameters['sd_mu_nuisance']['scale']
          Ic_corr[8] = mu_nuis*mu_shape
        else:
          Ic_corr[8] = 0.

        Ic_corr_total = np.sum(Ic_corr,axis=0)

        # Write fiducial model SD, unless it already exists
        if self.fid_values_exist is False:
            # Store the values now
            fid_file = open(os.path.join(
                self.data_directory, self.fiducial_file), 'w')
            fid_file.write('# Fiducial parameters')
            for key, value in iter(data.mcmc_parameters.items()):
                fid_file.write(', %s = %.6g' % (
                    key, value['current']*value['scale']))
            fid_file.write('\n')

            for nu_i in range(self.detector_bin_number):
                # Get the nu values from CLASS. The above noise file checks should insure that this is consistent
                fid_file.write("%6d  " % sd_nu[nu_i] )
                fid_file.write("%.8g  " % (sd_amp[nu_i]+self.noise_Ic[nu_i]+Ic_corr_total[nu_i]))
                fid_file.write("\n")
            print('\n')
            warnings.warn("Writing fiducial model in %s, for %s likelihood\n" % (os.path.join(self.data_directory,self.fiducial_file), self.name))
            return 1j

        # Compute likelihood
        chi2 = 0.

        Ic_obs = np.zeros(self.detector_bin_number)
        Ic_the = np.zeros(self.detector_bin_number)

        # Simplified inverse covmat (assumes uncorrelated noise)
        invcovmat = np.eye(self.detector_bin_number)

        # Calculate data, theoretical prediction, and noise
        for nu_i in range(self.detector_bin_number):
            Ic_obs[nu_i] = self.Ic_fid[nu_i]
            Ic_the[nu_i] = sd_amp[nu_i]+self.noise_Ic[nu_i]+Ic_corr_total[nu_i]
            invcovmat[nu_i,nu_i] = 1./(self.noise_Ic[nu_i]*self.noise_Ic[nu_i])

        # New formula
        chi2 = chi2_prior + np.dot((Ic_the-Ic_obs),np.dot(invcovmat,(Ic_the-Ic_obs)))

        return -chi2/2


###################################
# MPK TYPE LIKELIHOOD
# --> sdss, wigglez, etc.
###################################
class Likelihood_mpk(Likelihood):

    def __init__(self, path, data, command_line, common=False, common_dict={}):

        Likelihood.__init__(self, path, data, command_line)

        # require P(k) from class
        self.need_cosmo_arguments(data, {'output': 'mPk'})

        if common:
            self.add_common_knowledge(common_dict)

        try:
            self.use_halofit
        except:
            self.use_halofit = False

        if self.use_halofit:
            self.need_cosmo_arguments(data, {'non linear': 'halofit'})

        # sdssDR7 by T. Brinckmann
        # Based on Reid et al. 2010 arXiv:0907.1659 - Note: arXiv version not updated
        try:
            self.use_sdssDR7
        except:
            self.use_sdssDR7 = False

        # read values of k (in h/Mpc)
        self.k_size = self.max_mpk_kbands_use-self.min_mpk_kbands_use+1
        self.mu_size = 1
        self.k = np.zeros((self.k_size), 'float64')
        self.kh = np.zeros((self.k_size), 'float64')

        datafile = open(os.path.join(self.data_directory, self.kbands_file), 'r')
        for i in range(self.num_mpk_kbands_full):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            if i+2 > self.min_mpk_kbands_use and i < self.max_mpk_kbands_use:
                self.kh[i-self.min_mpk_kbands_use+1] = float(line.split()[0])
        datafile.close()

        khmax = self.kh[-1]

        # check if need hight value of k for giggleZ
        try:
            self.use_giggleZ
        except:
            self.use_giggleZ = False

        # Try a new model, with an additional nuisance parameter. Note
        # that the flag use_giggleZPP0 being True requires use_giggleZ
        # to be True as well. Note also that it is defined globally,
        # and not for every redshift bin.
        if self.use_giggleZ:
            try:
                self.use_giggleZPP0
            except:
                self.use_giggleZPP0 = False
        else:
            self.use_giggleZPP0 = False

        # If the flag use_giggleZPP0 is set to True, the nuisance parameters
        # P0_a, P0_b, P0_c and P0_d are expected.
        if self.use_giggleZPP0:
            if 'P0_a' not in data.get_mcmc_parameters(['nuisance']):
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "P0_a is not defined in the .param file, whereas this " +
                    "nuisance parameter is required when the flag " +
                    "'use_giggleZPP0' is set to true for WiggleZ")

        if self.use_giggleZ:
            datafile = open(os.path.join(self.data_directory,self.giggleZ_fidpk_file), 'r')

            line = datafile.readline()
            k = float(line.split()[0])
            line_number = 1
            while (k < self.kh[0]):
                line = datafile.readline()
                k = float(line.split()[0])
                line_number += 1
            ifid_discard = line_number-2
            while (k < khmax):
                line = datafile.readline()
                k = float(line.split()[0])
                line_number += 1
            datafile.close()
            self.k_fid_size = line_number-ifid_discard+1
            khmax = k

        if self.use_halofit:
            khmax *= 2

        # require k_max and z_max from the cosmological module
        if self.use_sdssDR7:
            self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
            self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': 7.5*self.kmax})
        else:
            self.need_cosmo_arguments(
                data, {'P_k_max_h/Mpc': khmax, 'z_max_pk': self.redshift})

        # read information on different regions in the sky
        try:
            self.has_regions
        except:
            self.has_regions = False

        if (self.has_regions):
            self.num_regions = len(self.used_region)
            self.num_regions_used = 0
            for i in range(self.num_regions):
                if (self.used_region[i]):
                    self.num_regions_used += 1
            if (self.num_regions_used == 0):
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "Mpk: no regions begin used in this data set")
        else:
            self.num_regions = 1
            self.num_regions_used = 1
            self.used_region = [True]

        # read window functions
        self.n_size = self.max_mpk_points_use-self.min_mpk_points_use+1

        self.window = np.zeros(
            (self.num_regions, self.n_size, self.k_size), 'float64')

        datafile = open(os.path.join(self.data_directory, self.windows_file), 'r')
        for i_region in range(self.num_regions):
            for i in range(self.num_mpk_points_full):
                line = datafile.readline()
                while line.find('#') != -1:
                    line = datafile.readline()
                if (i+2 > self.min_mpk_points_use and i < self.max_mpk_points_use):
                    for j in range(self.k_size):
                        self.window[i_region, i-self.min_mpk_points_use+1, j] = float(line.split()[j+self.min_mpk_kbands_use-1])
        datafile.close()

        # read measurements
        self.P_obs = np.zeros((self.num_regions, self.n_size), 'float64')
        self.P_err = np.zeros((self.num_regions, self.n_size), 'float64')

        datafile = open(os.path.join(self.data_directory, self.measurements_file), 'r')
        for i_region in range(self.num_regions):
            for i in range(self.num_mpk_points_full):
                line = datafile.readline()
                while line.find('#') != -1:
                    line = datafile.readline()
                if (i+2 > self.min_mpk_points_use and
                    i < self.max_mpk_points_use):
                    self.P_obs[i_region, i-self.min_mpk_points_use+1] = float(line.split()[3])
                    self.P_err[i_region, i-self.min_mpk_points_use+1] = float(line.split()[4])
        datafile.close()

        # read covariance matrices
        try:
            self.covmat_file
            self.use_covmat = True
        except:
            self.use_covmat = False

        try:
            self.use_invcov
        except:
            self.use_invcov = False

        self.invcov = np.zeros(
            (self.num_regions, self.n_size, self.n_size), 'float64')

        if self.use_covmat:
            cov = np.zeros((self.n_size, self.n_size), 'float64')
            invcov_tmp = np.zeros((self.n_size, self.n_size), 'float64')

            datafile = open(os.path.join(self.data_directory, self.covmat_file), 'r')
            for i_region in range(self.num_regions):
                for i in range(self.num_mpk_points_full):
                    line = datafile.readline()
                    while line.find('#') != -1:
                        line = datafile.readline()
                    if (i+2 > self.min_mpk_points_use and i < self.max_mpk_points_use):
                        for j in range(self.num_mpk_points_full):
                            if (j+2 > self.min_mpk_points_use and j < self.max_mpk_points_use):
                                cov[i-self.min_mpk_points_use+1,j-self.min_mpk_points_use+1] = float(line.split()[j])

                if self.use_invcov:
                    invcov_tmp = cov
                else:
                    invcov_tmp = np.linalg.inv(cov)
                for i in range(self.n_size):
                    for j in range(self.n_size):
                        self.invcov[i_region, i, j] = invcov_tmp[i, j]
            datafile.close()
        else:
            for i_region in range(self.num_regions):
                for j in range(self.n_size):
                    self.invcov[i_region, j, j] = \
                        1./(self.P_err[i_region, j]**2)

        # read fiducial model
        if self.use_giggleZ:
            self.P_fid = np.zeros((self.k_fid_size), 'float64')
            self.k_fid = np.zeros((self.k_fid_size), 'float64')
            datafile = open(os.path.join(self.data_directory,self.giggleZ_fidpk_file), 'r')
            for i in range(ifid_discard):
                line = datafile.readline()
            for i in range(self.k_fid_size):
                line = datafile.readline()
                self.k_fid[i] = float(line.split()[0])
                self.P_fid[i] = float(line.split()[1])
            datafile.close()

        # read integral constraint
        if self.use_sdssDR7:
            self.zerowindowfxn = np.zeros((self.k_size), 'float64')
            datafile = open(os.path.join(self.data_directory,self.zerowindowfxn_file), 'r')
            for i in range(self.k_size):
                line = datafile.readline()
                self.zerowindowfxn[i] = float(line.split()[0])
            datafile.close()
            self.zerowindowfxnsubtractdat = np.zeros((self.n_size), 'float64')
            datafile = open(os.path.join(self.data_directory,self.zerowindowfxnsubtractdat_file), 'r')
            line = datafile.readline()
            self.zerowindowfxnsubtractdatnorm = float(line.split()[0])
            for i in range(self.n_size):
                line = datafile.readline()
                self.zerowindowfxnsubtractdat[i] = float(line.split()[0])
            datafile.close()

        # initialize array of values for the nuisance parameters a1,a2
        if self.use_sdssDR7:
            nptsa1=self.nptsa1
            nptsa2=self.nptsa2
            a1maxval=self.a1maxval
            self.a1list=np.zeros(self.nptstot)
            self.a2list=np.zeros(self.nptstot)
            da1 = a1maxval/(nptsa1//2)
            da2 = self.a2maxpos(-a1maxval) / (nptsa2//2)
            count=0
            for i in range(-nptsa1//2, nptsa1//2+1):
                for j in range(-nptsa2//2, nptsa2//2+1):
                    a1val = da1*i
                    a2val = da2*j
                    if ((a2val >= 0.0 and a2val <= self.a2maxpos(a1val) and a2val >= self.a2minfinalpos(a1val)) or \
                        (a2val <= 0.0 and a2val <= self.a2maxfinalneg(a1val) and a2val >= self.a2minneg(a1val))):
                        if (self.testa1a2(a1val,a2val) == False):
                            raise io_mp.LikelihoodError(
                                'Error in likelihood %s ' % (self.name) +
                                'Nuisance parameter values not valid: %s %s' % (a1,a2) )
                        if(count >= self.nptstot):
                            raise io_mp.LikelihoodError(
                                'Error in likelihood %s ' % (self.name) +
                                'count > nptstot failure' )
                        self.a1list[count]=a1val
                        self.a2list[count]=a2val
                        count=count+1

        return

    # functions added for nuisance parameter space checks.
    def a2maxpos(self,a1val):
        a2max = -1.0
        if (a1val <= min(self.s1/self.k1,self.s2/self.k2)):
            a2max = min(self.s1/self.k1**2 - a1val/self.k1, self.s2/self.k2**2 - a1val/self.k2)
        return a2max

    def a2min1pos(self,a1val):
        a2min1 = 0.0
        if(a1val <= 0.0):
            a2min1 = max(-self.s1/self.k1**2 - a1val/self.k1, -self.s2/self.k2**2 - a1val/self.k2, 0.0)
        return a2min1

    def a2min2pos(self,a1val):
        a2min2 = 0.0
        if(abs(a1val) >= 2.0*self.s1/self.k1 and a1val <= 0.0):
            a2min2 = a1val**2/self.s1*0.25
        return a2min2

    def a2min3pos(self,a1val):
        a2min3 = 0.0
        if(abs(a1val) >= 2.0*self.s2/self.k2 and a1val <= 0.0):
            a2min3 = a1val**2/self.s2*0.25
        return a2min3

    def a2minfinalpos(self,a1val):
        a2minpos = max(self.a2min1pos(a1val),self.a2min2pos(a1val),self.a2min3pos(a1val))
        return a2minpos

    def a2minneg(self,a1val):
        if (a1val >= max(-self.s1/self.k1,-self.s2/self.k2)):
            a2min = max(-self.s1/self.k1**2 - a1val/self.k1, -self.s2/self.k2**2 - a1val/self.k2)
        else:
            a2min = 1.0
        return a2min

    def a2max1neg(self,a1val):
        if(a1val >= 0.0):
            a2max1 = min(self.s1/self.k1**2 - a1val/self.k1, self.s2/self.k2**2 - a1val/self.k2, 0.0)
        else:
            a2max1 = 0.0
        return a2max1

    def a2max2neg(self,a1val):
        a2max2 = 0.0
        if(abs(a1val) >= 2.0*self.s1/self.k1 and a1val >= 0.0):
            a2max2 = -a1val**2/self.s1*0.25
        return a2max2

    def a2max3neg(self,a1val):
        a2max3 = 0.0
        if(abs(a1val) >= 2.0*self.s2/self.k2 and a1val >= 0.0):
            a2max3 = -a1val**2/self.s2*0.25
        return a2max3

    def a2maxfinalneg(self,a1val):
        a2maxneg = min(self.a2max1neg(a1val),self.a2max2neg(a1val),self.a2max3neg(a1val))
        return a2maxneg

    def testa1a2(self,a1val, a2val):
        testresult = True
        # check if there's an extremum; either a1val or a2val has to be negative, not both
        if (a2val==0.):
             return testresult #not in the original code, but since a2val=0 returns True this way I avoid zerodivisionerror
        kext = -a1val/2.0/a2val
        diffval = abs(a1val*kext + a2val*kext**2)
        if(kext > 0.0 and kext <= self.k1 and diffval > self.s1):
            testresult = False
        if(kext > 0.0 and kext <= self.k2 and diffval > self.s2):
            testresult = False
        if (abs(a1val*self.k1 + a2val*self.k1**2) > self.s1):
            testresult = False
        if (abs(a1val*self.k2 + a2val*self.k2**2) > self.s2):
            testresult = False
        return testresult


    def add_common_knowledge(self, common_dictionary):
        """
        Add to a class the content of a shared dictionary of attributes

        The purpose of this method is to set some attributes globally for a Pk
        likelihood, that are shared amongst all the redshift bins (in
        WiggleZ.data for instance, a few flags and numbers are defined that
        will be transfered to wigglez_a, b, c and d

        """
        for key, value in dictitems(common_dictionary):
            # First, check if the parameter exists already
            try:
                exec("self.%s" % key)
                warnings.warn(
                    "parameter %s from likelihood %s will be replaced by " +
                    "the common knowledge routine" % (key, self.name))
            except:
                if type(value) != type('foo'):
                    exec("self.%s = %s" % (key, value))
                else:
                    exec("self.%s = '%s'" % (key, value))

    # compute likelihood
    def loglkl(self, cosmo, data):

        # reduced Hubble parameter
        h = cosmo.h()

        # WiggleZ and sdssDR7 specific
        if self.use_scaling:
            # angular diameter distance at this redshift, in Mpc
            d_angular = cosmo.angular_distance(self.redshift)

            # radial distance at this redshift, in Mpc, is simply 1/H (itself
            # in Mpc^-1). Hz is an array, with only one element.
            r, Hz = cosmo.z_of_r([self.redshift])
            d_radial = 1/Hz[0]

            # scaling factor = (d_angular**2 * d_radial)^(1/3) for the
            # fiducial cosmology used in the data files of the observations
            # divided by the same quantity for the cosmology we are comparing with.
            # The fiducial values are stored in the .data files for
            # each experiment, and are truly in Mpc. Beware for a potential
            # difference with CAMB conventions here.
            scaling = pow(
                (self.d_angular_fid/d_angular)**2 *
                (self.d_radial_fid/d_radial), 1./3.)
        else:
            scaling = 1
        # get rescaled values of k in 1/Mpc
        self.k = self.kh*h*scaling

        # get P(k) at right values of k, convert it to (Mpc/h)^3 and rescale it
        P_lin = np.zeros((self.k_size), 'float64')

        # If the flag use_giggleZ is set to True, the power spectrum retrieved
        # from Class will get rescaled by the fiducial power spectrum given by
        # the GiggleZ N-body simulations CITE
        if self.use_giggleZ:
            P = np.zeros((self.k_fid_size), 'float64')
            for i in range(self.k_fid_size):
                P[i] = cosmo.pk(self.k_fid[i]*h, self.redshift)
                power = 0
                # The following create a polynome in k, which coefficients are
                # stored in the .data files of the experiments.
                for j in range(6):
                    power += self.giggleZ_fidpoly[j]*self.k_fid[i]**j
                # rescale P by fiducial model and get it in (Mpc/h)**3
                P[i] *= pow(10, power)*(h/scaling)**3/self.P_fid[i]

            if self.use_giggleZPP0:
                # Shot noise parameter addition to GiggleZ model. It should
                # recover the proper nuisance parameter, depending on the name.
                # I.e., Wigglez_A should recover P0_a, etc...
                tag = self.name[-2:]  # circle over "_a", "_b", etc...
                P0_value = data.mcmc_parameters['P0'+tag]['current'] *\
                    data.mcmc_parameters['P0'+tag]['scale']
                P_lin = np.interp(self.kh,self.k_fid,P+P0_value)
            else:
                # get P_lin by interpolation. It is still in (Mpc/h)**3
                P_lin = np.interp(self.kh, self.k_fid, P)

        elif self.use_sdssDR7:
            kh = np.geomspace(1e-3,1,num=int((math.log(1.0)-math.log(1e-3))/0.01)+1) # k in h/Mpc
            # Rescale the scaling factor by the fiducial value for h divided by the sampled value
            # h=0.701 was used for the N-body calibration simulations
            scaling = scaling * (0.701/h)
            k = kh*h # k in 1/Mpc

            # Define redshift bins and associated bao 2 sigma value [NEAR, MID, FAR]
            z = np.array([0.235, 0.342, 0.421])
            sigma2bao = np.array([86.9988, 85.1374, 84.5958])
            # Initialize arrays
            # Analytical growth factor for each redshift bin
            D_growth = np.zeros(len(z))
            # P(k) *with* wiggles, both linear and nonlinear
            Plin = np.zeros(len(k), 'float64')
            Pnl = np.zeros(len(k), 'float64')
            # P(k) *without* wiggles, both linear and nonlinear
            Psmooth = np.zeros(len(k), 'float64')
            Psmooth_nl = np.zeros(len(k), 'float64')
            # Damping function and smeared P(k)
            fdamp = np.zeros([len(k), len(z)], 'float64')
            Psmear = np.zeros([len(k), len(z)], 'float64')
            # Ratio of smoothened non-linear to linear P(k)
            nlratio = np.zeros([len(k), len(z)], 'float64')
            # Loop over each redshift bin
            for j in range(len(z)):
                # Compute growth factor at each redshift
                # This growth factor is normalized by the growth factor today
                D_growth[j] = cosmo.scale_independent_growth_factor(z[j])
                # Compute Pk *with* wiggles, both linear and nonlinear
                # Get P(k) at right values of k in Mpc**3, convert it to (Mpc/h)^3 and rescale it
                # Get values of P(k) in Mpc**3
                for i in range(len(k)):
                    Plin[i] = cosmo.pk_lin(k[i], z[j])
                    Pnl[i] = cosmo.pk(k[i], z[j])
                # Get rescaled values of P(k) in (Mpc/h)**3
                Plin *= h**3 #(h/scaling)**3
                Pnl *= h**3 #(h/scaling)**3
                # Compute Pk *without* wiggles, both linear and nonlinear
                Psmooth = self.remove_bao(kh,Plin)
                Psmooth_nl = self.remove_bao(kh,Pnl)
                # Apply Gaussian damping due to non-linearities
                fdamp[:,j] = np.exp(-0.5*sigma2bao[j]*kh**2)
                Psmear[:,j] = Plin*fdamp[:,j]+Psmooth*(1.0-fdamp[:,j])
                # Take ratio of smoothened non-linear to linear P(k)
                nlratio[:,j] = Psmooth_nl/Psmooth

            # Save fiducial model for non-linear corrections using the flat fiducial
            # Omega_m = 0.25, Omega_L = 0.75, h = 0.701
            # Re-run if changes are made to how non-linear corrections are done
            # e.g. the halofit implementation in CLASS
            # To re-run fiducial, set <experiment>.create_fid = True in .data file
            # Can leave option enabled, as it will only compute once at the start
            try:
                self.create_fid
            except:
                self.create_fid = False

            if self.create_fid == True:
                # Calculate relevant flat fiducial quantities
                fidnlratio, fidNEAR, fidMID, fidFAR = self.get_flat_fid(cosmo,data,kh,z,sigma2bao)
                try:
                    existing_fid = np.loadtxt('data/sdss_lrgDR7/sdss_lrgDR7_fiducialmodel.dat')
                    print('sdss_lrgDR7: Checking fiducial deviations for near, mid and far bins:', np.sum(existing_fid[:,1] - fidNEAR),np.sum(existing_fid[:,2] - fidMID), np.sum(existing_fid[:,3] - fidFAR))
                    if np.sum(existing_fid[:,1] - fidNEAR) + np.sum(existing_fid[:,2] - fidMID) + np.sum(existing_fid[:,3] - fidFAR) < 10**-5:
                        self.create_fid = False
                except:
                    pass
                if self.create_fid == True:
                    print('sdss_lrgDR7: Creating fiducial file with Omega_b = 0.25, Omega_L = 0.75, h = 0.701')
                    print('             Required for non-linear modeling')
                    # Save non-linear corrections from N-body sims for each redshift bin
                    arr=np.zeros((np.size(kh),7))
                    arr[:,0]=kh
                    arr[:,1]=fidNEAR
                    arr[:,2]=fidMID
                    arr[:,3]=fidFAR
                    # Save non-linear corrections from halofit for each redshift bin
                    arr[:,4:7]=fidnlratio
                    np.savetxt('data/sdss_lrgDR7/sdss_lrgDR7_fiducialmodel.dat',arr)
                    self.create_fid = False
                    print('             Fiducial created')

            # Load fiducial model
            fiducial = np.loadtxt('data/sdss_lrgDR7/sdss_lrgDR7_fiducialmodel.dat')
            fid = fiducial[:,1:4]
            fidnlratio = fiducial[:,4:7]

            # Put all factors together to obtain the P(k) for each redshift bin
            Pnear=np.interp(kh,kh,Psmear[:,0]*(nlratio[:,0]/fidnlratio[:,0])*fid[:,0]*D_growth[0]**(-2.))
            Pmid =np.interp(kh,kh,Psmear[:,1]*(nlratio[:,1]/fidnlratio[:,1])*fid[:,1]*D_growth[1]**(-2.))
            Pfar =np.interp(kh,kh,Psmear[:,2]*(nlratio[:,2]/fidnlratio[:,2])*fid[:,2]*D_growth[2]**(-2.))

            # Define and rescale k
            self.k=self.kh*h*scaling
            # Weighted mean of the P(k) for each redshift bin
            P_lin=(0.395*Pnear+0.355*Pmid+0.250*Pfar)
            P_lin=np.interp(self.k,kh*h,P_lin)*(1./scaling)**3 # remember self.k is scaled but self.kh isn't

        else:
            # get rescaled values of k in 1/Mpc
            self.k = self.kh*h*scaling
            # get values of P(k) in Mpc**3
            for i in range(self.k_size):
                P_lin[i] = cosmo.pk(self.k[i], self.redshift)
            # get rescaled values of P(k) in (Mpc/h)**3
            P_lin *= (h/scaling)**3

        # infer P_th from P_lin. It is still in (Mpc/h)**3. TODO why was it
        # called P_lin in the first place ? Couldn't we use now P_th all the
        # way ?
        P_th = P_lin

        if self.use_sdssDR7:
            chisq =np.zeros(self.nptstot)
            chisqmarg = np.zeros(self.nptstot)

            Pth = P_th
            Pth_k = P_th*(self.k/h) # self.k has the scaling included, so self.k/h != self.kh
            Pth_k2 = P_th*(self.k/h)**2

            WPth = np.dot(self.window[0,:], Pth)
            WPth_k = np.dot(self.window[0,:], Pth_k)
            WPth_k2 = np.dot(self.window[0,:], Pth_k2)

            sumzerow_Pth = np.sum(self.zerowindowfxn*Pth)/self.zerowindowfxnsubtractdatnorm
            sumzerow_Pth_k = np.sum(self.zerowindowfxn*Pth_k)/self.zerowindowfxnsubtractdatnorm
            sumzerow_Pth_k2 = np.sum(self.zerowindowfxn*Pth_k2)/self.zerowindowfxnsubtractdatnorm

            covdat = np.dot(self.invcov[0,:,:],self.P_obs[0,:])
            covth  = np.dot(self.invcov[0,:,:],WPth)
            covth_k  = np.dot(self.invcov[0,:,:],WPth_k)
            covth_k2  = np.dot(self.invcov[0,:,:],WPth_k2)
            covth_zerowin  = np.dot(self.invcov[0,:,:],self.zerowindowfxnsubtractdat)
            sumDD = np.sum(self.P_obs[0,:] * covdat)
            sumDT = np.sum(self.P_obs[0,:] * covth)
            sumDT_k = np.sum(self.P_obs[0,:] * covth_k)
            sumDT_k2 = np.sum(self.P_obs[0,:] * covth_k2)
            sumDT_zerowin = np.sum(self.P_obs[0,:] * covth_zerowin)

            sumTT = np.sum(WPth*covth)
            sumTT_k = np.sum(WPth*covth_k)
            sumTT_k2 = np.sum(WPth*covth_k2)
            sumTT_k_k = np.sum(WPth_k*covth_k)
            sumTT_k_k2 = np.sum(WPth_k*covth_k2)
            sumTT_k2_k2 = np.sum(WPth_k2*covth_k2)
            sumTT_zerowin = np.sum(WPth*covth_zerowin)
            sumTT_k_zerowin = np.sum(WPth_k*covth_zerowin)
            sumTT_k2_zerowin = np.sum(WPth_k2*covth_zerowin)
            sumTT_zerowin_zerowin = np.sum(self.zerowindowfxnsubtractdat*covth_zerowin)

            currminchisq = 1000.0

            # analytic marginalization over a1,a2
            for i in range(self.nptstot):
                a1val = self.a1list[i]
                a2val = self.a2list[i]
                zerowinsub = -(sumzerow_Pth + a1val*sumzerow_Pth_k + a2val*sumzerow_Pth_k2)
                sumDT_tot = sumDT + a1val*sumDT_k + a2val*sumDT_k2 + zerowinsub*sumDT_zerowin
                sumTT_tot = sumTT + a1val**2.0*sumTT_k_k + a2val**2.0*sumTT_k2_k2 + \
                    zerowinsub**2.0*sumTT_zerowin_zerowin + \
                    2.0*a1val*sumTT_k + 2.0*a2val*sumTT_k2 + 2.0*a1val*a2val*sumTT_k_k2 + \
                    2.0*zerowinsub*sumTT_zerowin + 2.0*zerowinsub*a1val*sumTT_k_zerowin + \
                    2.0*zerowinsub*a2val*sumTT_k2_zerowin
                minchisqtheoryamp = sumDT_tot/sumTT_tot
                chisq[i] = sumDD - 2.0*minchisqtheoryamp*sumDT_tot + minchisqtheoryamp**2.0*sumTT_tot
                chisqmarg[i] = sumDD - sumDT_tot**2.0/sumTT_tot + math.log(sumTT_tot) - \
                    2.0*math.log(1.0 + math.erf(sumDT_tot/2.0/math.sqrt(sumTT_tot)))
                if(i == 0 or chisq[i] < currminchisq):
                    myminchisqindx = i
                    currminchisq = chisq[i]
                    currminchisqmarg = chisqmarg[i]
                    minchisqtheoryampminnuis = minchisqtheoryamp
                if(i == int(self.nptstot/2)):
                    chisqnonuis = chisq[i]
                    minchisqtheoryampnonuis = minchisqtheoryamp
                    if(abs(a1val) > 0.001 or abs(a2val) > 0.001):
                         print('sdss_lrgDR7: ahhhh! violation!!', a1val, a2val)

            # numerically marginalize over a1,a2 now using values stored in chisq
            minchisq = np.min(chisqmarg)
            maxchisq = np.max(chisqmarg)

            LnLike = np.sum(np.exp(-(chisqmarg-minchisq)/2.0)/(self.nptstot*1.0))
            if(LnLike == 0):
                #LnLike = LogZero
                raise io_mp.LikelihoodError(
                    'Error in likelihood %s ' % (self.name) +
                    'LRG LnLike LogZero error.' )
            else:
                chisq = -2.*math.log(LnLike) + minchisq
            #print('DR7 chi2/2=',chisq/2.)

        #if we are not using DR7
        else:
            W_P_th = np.zeros((self.n_size), 'float64')

            # starting analytic marginalisation over bias

            # Define quantities living in all the regions possible. If only a few
            # regions are selected in the .data file, many elements from these
            # arrays will stay at 0.
            P_data_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')
            W_P_th_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')
            cov_dat_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')
            cov_th_large = np.zeros(
                (self.n_size*self.num_regions_used), 'float64')

            normV = 0

            # Loop over all the available regions
            for i_region in range(self.num_regions):
                # In each region that was selected with the array of flags
                # self.used_region, define boundaries indices, and fill in the
                # corresponding windowed power spectrum. All the unused regions
                # will still be set to zero as from the initialization, which will
                # not contribute anything in the final sum.

                if self.used_region[i_region]:
                    imin = i_region*self.n_size
                    imax = (i_region+1)*self.n_size-1

                    W_P_th = np.dot(self.window[i_region, :], P_th)
                    #print(W_P_th)
                    for i in range(self.n_size):
                        P_data_large[imin+i] = self.P_obs[i_region, i]
                        W_P_th_large[imin+i] = W_P_th[i]
                        cov_dat_large[imin+i] = np.dot(
                            self.invcov[i_region, i, :],
                            self.P_obs[i_region, :])
                        cov_th_large[imin+i] = np.dot(
                            self.invcov[i_region, i, :],
                            W_P_th[:])

            # Explain what it is TODO
            normV += np.dot(W_P_th_large, cov_th_large)
            # Sort of bias TODO ?
            b_out = np.sum(W_P_th_large*cov_dat_large) / \
                np.sum(W_P_th_large*cov_th_large)

            # Explain this formula better, link to article ?
            chisq = np.dot(P_data_large, cov_dat_large) - \
                np.dot(W_P_th_large, cov_dat_large)**2/normV
            #print('WiggleZ chi2=',chisq/2.)

        return -chisq/2

    def remove_bao(self,k_in,pk_in):
        # De-wiggling routine by Mario Ballardini

        # This k range has to contain the BAO features:
        k_ref=[2.8e-2, 4.5e-1]

        # Get interpolating function for input P(k) in log-log space:
        _interp_pk = scipy.interpolate.interp1d( np.log(k_in), np.log(pk_in),
                                                 kind='quadratic', bounds_error=False )
        interp_pk = lambda x: np.exp(_interp_pk(np.log(x)))

        # Spline all (log-log) points outside k_ref range:
        idxs = np.where(np.logical_or(k_in <= k_ref[0], k_in >= k_ref[1]))
        _pk_smooth = scipy.interpolate.UnivariateSpline( np.log(k_in[idxs]),
                                                         np.log(pk_in[idxs]), k=3, s=0 )
        pk_smooth = lambda x: np.exp(_pk_smooth(np.log(x)))

        # Find second derivative of each spline:
        fwiggle = scipy.interpolate.UnivariateSpline(k_in, pk_in / pk_smooth(k_in), k=3, s=0)
        derivs = np.array([fwiggle.derivatives(_k) for _k in k_in]).T
        d2 = scipy.interpolate.UnivariateSpline(k_in, derivs[2], k=3, s=1.0)

        # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
        # low-order spline through zeros to subtract smooth trend from wiggles fn.
        wzeros = d2.roots()
        wzeros = wzeros[np.where(np.logical_and(wzeros >= k_ref[0], wzeros <= k_ref[1]))]
        wzeros = np.concatenate((wzeros, [k_ref[1],]))
        wtrend = scipy.interpolate.UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=0)

        # Construct smooth no-BAO:
        idxs = np.where(np.logical_and(k_in > k_ref[0], k_in < k_ref[1]))
        pk_nobao = pk_smooth(k_in)
        pk_nobao[idxs] *= wtrend(k_in[idxs])

        # Construct interpolating functions:
        ipk = scipy.interpolate.interp1d( k_in, pk_nobao, kind='linear',
                                          bounds_error=False, fill_value=0. )

        pk_nobao = ipk(k_in)

        return pk_nobao

    def get_flat_fid(self,cosmo,data,kh,z,sigma2bao):
        # SDSS DR7 LRG specific function
        # Compute fiducial properties for a flat fiducial
        # with Omega_m = 0.25, Omega_L = 0.75, h = 0.701
        param_backup = data.cosmo_arguments
        data.cosmo_arguments = {'P_k_max_h/Mpc': 1.5, 'ln10^{10}A_s': 3.0, 'N_ur': 3.04, 'h': 0.701,
                                'omega_b': 0.035*0.701**2, 'non linear': ' halofit ', 'YHe': 0.24, 'k_pivot': 0.05,
                                'n_s': 0.96, 'tau_reio': 0.084, 'z_max_pk': 0.5, 'output': ' mPk ',
                                'omega_cdm': 0.215*0.701**2, 'T_cmb': 2.726}
        cosmo.empty()
        cosmo.set(data.cosmo_arguments)
        cosmo.compute(['lensing'])
        h = data.cosmo_arguments['h']
        k = kh*h
        # P(k) *with* wiggles, both linear and nonlinear
        Plin = np.zeros(len(k), 'float64')
        Pnl = np.zeros(len(k), 'float64')
        # P(k) *without* wiggles, both linear and nonlinear
        Psmooth = np.zeros(len(k), 'float64')
        Psmooth_nl = np.zeros(len(k), 'float64')
        # Damping function and smeared P(k)
        fdamp = np.zeros([len(k), len(z)], 'float64')
        Psmear = np.zeros([len(k), len(z)], 'float64')
        # Ratio of smoothened non-linear to linear P(k)
        fidnlratio = np.zeros([len(k), len(z)], 'float64')
        # Loop over each redshift bin
        for j in range(len(z)):
            # Compute Pk *with* wiggles, both linear and nonlinear
            # Get P(k) at right values of k in Mpc**3, convert it to (Mpc/h)^3 and rescale it
            # Get values of P(k) in Mpc**3
            for i in range(len(k)):
                Plin[i] = cosmo.pk_lin(k[i], z[j])
                Pnl[i] = cosmo.pk(k[i], z[j])
            # Get rescaled values of P(k) in (Mpc/h)**3
            Plin *= h**3 #(h/scaling)**3
            Pnl *= h**3 #(h/scaling)**3
            # Compute Pk *without* wiggles, both linear and nonlinear
            Psmooth = self.remove_bao(kh,Plin)
            Psmooth_nl = self.remove_bao(kh,Pnl)
            # Apply Gaussian damping due to non-linearities
            fdamp[:,j] = np.exp(-0.5*sigma2bao[j]*kh**2)
            Psmear[:,j] = Plin*fdamp[:,j]+Psmooth*(1.0-fdamp[:,j])
            # Take ratio of smoothened non-linear to linear P(k)
            fidnlratio[:,j] = Psmooth_nl/Psmooth

        # Polynomials to shape small scale behavior from N-body sims
        kdata=kh
        fidpolyNEAR=np.zeros(np.size(kdata))
        fidpolyNEAR[kdata<=0.194055] = (1.0 - 0.680886*kdata[kdata<=0.194055] + 6.48151*kdata[kdata<=0.194055]**2)
        fidpolyNEAR[kdata>0.194055] = (1.0 - 2.13627*kdata[kdata>0.194055] + 21.0537*kdata[kdata>0.194055]**2 - 50.1167*kdata[kdata>0.194055]**3 + 36.8155*kdata[kdata>0.194055]**4)*1.04482
        fidpolyMID=np.zeros(np.size(kdata))
        fidpolyMID[kdata<=0.19431] = (1.0 - 0.530799*kdata[kdata<=0.19431] + 6.31822*kdata[kdata<=0.19431]**2)
        fidpolyMID[kdata>0.19431] = (1.0 - 1.97873*kdata[kdata>0.19431] + 20.8551*kdata[kdata>0.19431]**2 - 50.0376*kdata[kdata>0.19431]**3 + 36.4056*kdata[kdata>0.19431]**4)*1.04384
        fidpolyFAR=np.zeros(np.size(kdata))
        fidpolyFAR[kdata<=0.19148] = (1.0 - 0.475028*kdata[kdata<=0.19148] + 6.69004*kdata[kdata<=0.19148]**2)
        fidpolyFAR[kdata>0.19148] = (1.0 - 1.84891*kdata[kdata>0.19148] + 21.3479*kdata[kdata>0.19148]**2 - 52.4846*kdata[kdata>0.19148]**3 + 38.9541*kdata[kdata>0.19148]**4)*1.03753

        fidNEAR=np.interp(kh,kdata,fidpolyNEAR)
        fidMID=np.interp(kh,kdata,fidpolyMID)
        fidFAR=np.interp(kh,kdata,fidpolyFAR)

        cosmo.empty()
        data.cosmo_arguments = param_backup
        cosmo.set(data.cosmo_arguments)
        cosmo.compute(['lensing'])

        return fidnlratio, fidNEAR, fidMID, fidFAR

class Likelihood_sn(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # try and import pandas
        try:
            import pandas
        except ImportError:
            raise io_mp.MissingLibraryError(
                "This likelihood has a lot of IO manipulation. You have "
                "to install the 'pandas' library to use it. Please type:\n"
                "`(sudo) pip install pandas --user`")

        # check that every conflicting experiments is not present in the list
        # of tested experiments, in which case, complain
        if hasattr(self, 'conflicting_experiments'):
            for conflict in self.conflicting_experiments:
                if conflict in data.experiments:
                    raise io_mp.LikelihoodError(
                        'conflicting %s measurements, you can ' % conflict +
                        ' have either %s or %s ' % (self.name, conflict) +
                        'as an experiment, not both')

        # Read the configuration file, supposed to be called self.settings.
        # Note that we unfortunately can not
        # immediatly execute the file, as it is not formatted as strings.
        assert hasattr(self, 'settings') is True, (
            "You need to provide a settings file")
        self.read_configuration_file()

    def read_configuration_file(self):
        """
        Extract Python variables from the configuration file

        This routine performs the equivalent to the program "inih" used in the
        original c++ library.
        """
        settings_path = os.path.join(self.data_directory, self.settings)
        with open(settings_path, 'r') as config:
            for line in config:
                # Dismiss empty lines and commented lines
                if line and line.find('#') == -1 and line not in ['\n', '\r\n']:
                    lhs, rhs = [elem.strip() for elem in line.split('=')]
                    # lhs will always be a string, so set the attribute to this
                    # likelihood. The right hand side requires more work.
                    # First case, if set to T or F for True or False
                    if str(rhs) in ['T', 'F']:
                        rhs = True if str(rhs) == 'T' else False
                    # It can also be a path, starting with 'data/'. We remove
                    # this leading folder path
                    elif str(rhs).find('data/') != -1:
                        rhs = rhs.replace('data/', '')
                    else:
                        # Try  to convert it to a float
                        try:
                            rhs = float(rhs)
                        # If it fails, it is a string
                        except ValueError:
                            rhs = str(rhs)
                    # Set finally rhs to be a parameter of the class
                    setattr(self, lhs, rhs)

    def read_matrix(self, path):
        """
        extract the matrix from the path

        This routine uses the blazing fast pandas library (0.10 seconds to load
        a 740x740 matrix). If not installed, it uses a custom routine that is
        twice as slow (but still 4 times faster than the straightforward
        numpy.loadtxt method)

        .. note::

            the length of the matrix is stored on the first line... then it has
            to be unwrapped. The pandas routine read_table understands this
            immediatly, though.

        """
        from pandas import read_table
        path = os.path.join(self.data_directory, path)
        # The first line should contain the length.
        with open(path, 'r') as text:
            length = int(text.readline())

        # Note that this function does not require to skiprows, as it
        # understands the convention of writing the length in the first
        # line
        matrix = read_table(path).to_numpy().reshape((length, length))

        return matrix

    def read_light_curve_parameters(self):
        """
        Read the file jla_lcparams.txt containing the SN data

        .. note::

            the length of the resulting array should be equal to the length of
            the covariance matrices stored in C00, etc...

        """
        from pandas import read_table
        path = os.path.join(self.data_directory, self.data_file)

        # Recover the names of the columns. The names '3rdvar' and 'd3rdvar'
        # will be changed, because 3rdvar is not a valid variable name
        with open(path, 'r') as text:
            clean_first_line = text.readline()[1:].strip()
            names = [e.strip().replace('3rd', 'third')
                     for e in clean_first_line.split()]

        lc_parameters = read_table(
            path, sep=' ', names=names, header=0, index_col=False)
        return lc_parameters


class Likelihood_clocks(Likelihood):
    """Base implementation of H(z) measurements"""

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Read the content of the data file, containing z, Hz and error
        total = np.loadtxt(
            os.path.join(self.data_directory, self.data_file))

        # Store the columns separately
        self.z = total[:, 0]
        self.Hz = total[:, 1]
        self.err = total[:, 2]

    def loglkl(self, cosmo, data):

        # Store the speed of light in km/s
        c_light_km_per_sec = const.c/1000.
        chi2 = 0

        # Loop over the redshifts
        for index, z in enumerate(self.z):
            # Query the cosmo module for the Hubble rate (in 1/Mpc), and
            # convert it to km/s/Mpc
            H_cosmo = cosmo.Hubble(z)*c_light_km_per_sec
            # Add to the tota chi2
            chi2 += (self.Hz[index]-H_cosmo)**2/self.err[index]**2

        return -0.5 * chi2

###################################
# ISW-Likelihood
# by B. Stoelzner
###################################
class Likelihood_isw(Likelihood):
    def __init__(self, path, data, command_line):
        # Initialize
        Likelihood.__init__(self, path, data, command_line)
        self.need_cosmo_arguments(data, {'output': 'mPk','P_k_max_h/Mpc' : 300,'z_max_pk' : 5.1})

        # Read l,C_l, and the covariance matrix of the autocorrelation of the survey and the crosscorrelation of the survey with the CMB
        self.l_cross,cl_cross=np.loadtxt(os.path.join(self.data_directory,self.cl_cross_file),unpack=True,usecols=(0,1))
        self.l_auto,cl_auto=np.loadtxt(os.path.join(self.data_directory,self.cl_auto_file),unpack=True,usecols=(0,1))
        cov_cross=np.loadtxt(os.path.join(self.data_directory,self.cov_cross_file))
        cov_auto=np.loadtxt(os.path.join(self.data_directory,self.cov_auto_file))

        # Extract data in the specified range in l.
        self.l_cross=self.l_cross[self.l_min_cross:self.l_max_cross+1]
        cl_cross=cl_cross[self.l_min_cross:self.l_max_cross+1]
        self.l_auto=self.l_auto[self.l_min_auto:self.l_max_auto+1]
        cl_auto=cl_auto[self.l_min_auto:self.l_max_auto+1]
        cov_cross=cov_cross[self.l_min_cross:self.l_max_cross+1,self.l_min_cross:self.l_max_cross+1]
        cov_auto=cov_auto[self.l_min_auto:self.l_max_auto+1,self.l_min_auto:self.l_max_auto+1]

        # Create logarithically spaced bins in l.
        self.bins_cross=np.ceil(np.logspace(np.log10(self.l_min_cross),np.log10(self.l_max_cross),self.n_bins_cross+1))
        self.bins_auto=np.ceil(np.logspace(np.log10(self.l_min_auto),np.log10(self.l_max_auto),self.n_bins_auto+1))

        # Bin l,C_l, and covariance matrix in the previously defined bins
        self.l_binned_cross,self.cl_binned_cross,self.cov_binned_cross=self.bin_cl(self.l_cross,cl_cross,self.bins_cross,cov_cross)
        self.l_binned_auto,self.cl_binned_auto,self.cov_binned_auto=self.bin_cl(self.l_auto,cl_auto,self.bins_auto,cov_auto)

        # Read the redshift distribution of objects in the survey, perform an interpolation of dN/dz(z), and calculate the normalization in this redshift bin
        zz,dndz=np.loadtxt(os.path.join(self.data_directory,self.dndz_file),unpack=True,usecols=(0,1))
        self.dndz=scipy.interpolate.interp1d(zz,dndz,kind='cubic')
        self.norm=scipy.integrate.quad(self.dndz,self.z_min,self.z_max)[0]

    def bin_cl(self,l,cl,bins,cov=None):
        # This function bins l,C_l, and the covariance matrix in given bins in l
        B=[]
        for i in range(1,len(bins)):
            if i!=len(bins)-1:
                a=np.where((l<bins[i])&(l>=bins[i-1]))[0]
            else:
                a=np.where((l<=bins[i])&(l>=bins[i-1]))[0]
            c=np.zeros(len(l))
            c[a]=1./len(a)
            B.append(c)
        l_binned=np.dot(B,l)
        cl_binned=np.dot(B,cl)
        if cov is not None:
            cov_binned=np.dot(B,np.dot(cov,np.transpose(B)))
            return l_binned,cl_binned,cov_binned
        else:
            return l_binned,cl_binned

    def integrand_cross(self,z,cosmo,l):
        # This function will be integrated to calculate the exspected crosscorrelation between the survey and the CMB
        c= const.c/1000.
        H0=cosmo.h()*100
        Om=cosmo.Omega0_m()
        k=lambda z:(l+0.5)/(cosmo.angular_distance(z)*(1+z))
        return (3*Om*H0**2)/((c**2)*(l+0.5)**2)*self.dndz(z)*cosmo.Hubble(z)*cosmo.scale_independent_growth_factor(z)*scipy.misc.derivative(lambda z:cosmo.scale_independent_growth_factor(z)*(1+z),x0=z,dx=1e-4)*cosmo.pk(k(z),0)/self.norm

    def integrand_auto(self,z,cosmo,l):
        # This function will be integrated to calculate the expected autocorrelation of the survey
        c= const.c/1000.
        H0=cosmo.h()*100
        k=lambda z:(l+0.5)/(cosmo.angular_distance(z)*(1+z))
        return (self.dndz(z))**2*(cosmo.scale_independent_growth_factor(z))**2*cosmo.pk(k(z),0)*cosmo.Hubble(z)/(cosmo.angular_distance(z)*(1+z))**2/self.norm**2

    def compute_loglkl(self, cosmo, data,b):
        # Retrieve sampled parameter
        A=data.mcmc_parameters['A_ISW']['current']*data.mcmc_parameters['A_ISW']['scale']

        # Calculate the expected auto- and crosscorrelation by integrating over the redshift.
        cl_binned_cross_theory=np.array([(scipy.integrate.quad(self.integrand_cross,self.z_min,self.z_max,args=(cosmo,self.bins_cross[ll]))[0]+scipy.integrate.quad(self.integrand_cross,self.z_min,self.z_max,args=(cosmo,self.bins_cross[ll+1]))[0]+scipy.integrate.quad(self.integrand_cross,self.z_min,self.z_max,args=(cosmo,self.l_binned_cross[ll]))[0])/3 for ll in range(self.n_bins_cross)])
        cl_binned_auto_theory=np.array([scipy.integrate.quad(self.integrand_auto,self.z_min,self.z_max,args=(cosmo,ll),epsrel=1e-8)[0] for ll in self.l_binned_auto])

        # Calculate the chi-square of auto- and crosscorrelation
        chi2_cross=np.asscalar(np.dot(self.cl_binned_cross-A*b*cl_binned_cross_theory,np.dot(np.linalg.inv(self.cov_binned_cross),self.cl_binned_cross-A*b*cl_binned_cross_theory)))
        chi2_auto=np.asscalar(np.dot(self.cl_binned_auto-b**2*cl_binned_auto_theory,np.dot(np.linalg.inv(self.cov_binned_auto),self.cl_binned_auto-b**2*cl_binned_auto_theory)))
        return -0.5*(chi2_cross+chi2_auto)


###################################
# Dataset Likelihood
# by A. Lewis
# adapted to montepython by Gen Ye
###################################
class Likelihood_dataset(Likelihood):
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        try:
            from camb.mathutils import chi_squared as fast_chi_squared
        except ImportError:
            def fast_chi_squared(covinv, x):
                return covinv.dot(x).dot(x)
        self._fast_chi_squared = fast_chi_squared

        if os.path.isabs(self.dataset_file):
            data_file = self.dataset_file
            self.path = os.path.dirname(data_file)
        else:
            raise io_mp.LikelihoodError("No path given for %s."%(self.dataset_file))

        data_file = os.path.normpath(os.path.join(self.path, self.dataset_file))
        if not os.path.exists(data_file):
            raise io_mp.LikelihoodError("The data file '%s' could not be found at '%s'. "
                          "Either you have not installed this likelihood, "
                          "or have given the wrong packages installation path."%(self.dataset_file, self.path))
        self.load_dataset_file(data_file, getattr(self, 'dataset_params', {}))

    def load_dataset_file(self, filename, dataset_params=None):
        if '.dataset' not in filename:
            filename += '.dataset'
        ini = IniFile(filename)
        self.dataset_filename = filename
        # ini.params.update(self._default_dataset_params)
        ini.params.update(dataset_params or {})
        self.init_params(ini)
    
    def init_params(self, ini):
        pass

###################################
# CMBlikes Likelihood
# by A. Lewis
# adapted to montepython by Gen Ye
###################################
CMB_keys = ['tt', 'te', 'ee', 'bb']
class Likelihood_cmblikes(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "CMB"

    # used to form spectra names, e.g. AmapxBmap
    map_separator: str = 'x'
    
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)
        
        try:
            from camb.mathutils import chi_squared as fast_chi_squared
        except ImportError:
            def fast_chi_squared(covinv, x):
                return covinv.dot(x).dot(x)
        self._fast_chi_squared = fast_chi_squared

        if os.path.isabs(self.dataset_file):
            data_file = self.dataset_file
            self.path = os.path.dirname(data_file)
        else:
            raise io_mp.LikelihoodError("No path given for %s."%(self.dataset_file))

        data_file = os.path.normpath(os.path.join(self.path, self.dataset_file))
        if not os.path.exists(data_file):
            raise io_mp.LikelihoodError("The data file '%s' could not be found at '%s'. "
                          "Either you have not installed this likelihood, "
                          "or have given the wrong packages installation path."%(self.dataset_file, self.path))
        self.load_dataset_file(data_file, getattr(self, 'dataset_params', {}))

        # l_max has to take into account the window function of the lensing
        # so we check the computed l_max ("l_max" option) is higher than the requested one
        requested_l_max = int(np.max(self.cl_lmax))
        if (getattr(self, "l_max", None) or np.inf) < requested_l_max:
            raise io_mp.LikelihoodError("You are setting a very low l_max. "
                          "The likelihood value will probably not be correct. "
                          "Make sure to make 'l_max'>=%d"% requested_l_max)
        self.l_max = max(requested_l_max, getattr(self, "l_max", 0) or 0)
        self.need_cosmo_arguments(data, {'lensing': 'yes', 'output': 'tCl lCl pCl', 'l_max_scalars': self.l_max})

    def load_dataset_file(self, filename, dataset_params=None):
        if '.dataset' not in filename:
            filename += '.dataset'
        ini = IniFile(filename)
        self.dataset_filename = filename
        # ini.params.update(self._default_dataset_params)
        ini.params.update(dataset_params or {})
        self.init_params(ini)

    def init_params(self, ini):
        self.field_names = getattr(self, 'field_names', ['T', 'E', 'B', 'P'])
        self.tot_theory_fields = len(self.field_names)
        self.map_names = ini.split('map_names', default=[])
        self.has_map_names = bool(self.map_names)
        if self.has_map_names:
            # e.g. have multiple frequencies for given field measurement
            map_fields = ini.split('map_fields')
            if len(map_fields) != len(self.map_names):
                raise io_mp.LikelihoodError('number of map_fields does not match map_names')
            self.map_fields = [self.typeIndex(f) for f in map_fields]
        else:
            self.map_names = self.field_names
            self.map_fields = np.arange(len(self.map_names), dtype=int)
        fields_use = ini.split('fields_use', [])
        if len(fields_use):
            index_use = [self.typeIndex(f) for f in fields_use]
            use_theory_field = [i in index_use for i in range(self.tot_theory_fields)]
        else:
            if not self.has_map_names:
                io_mp.LikelihoodError('must have fields_use or map_names')
            use_theory_field = [True] * self.tot_theory_fields
        maps_use = ini.split('maps_use', [])
        if len(maps_use):
            if any(not i for i in use_theory_field):
                print('maps_use overrides fields_use')
            self.use_map = [False] * len(self.map_names)
            for j, map_used in enumerate(maps_use):
                if map_used in self.map_names:
                    self.use_map[self.map_names.index(map_used)] = True
                else:
                    raise io_mp.LikelihoodError('maps_use item not found - %s' % map_used)
        else:
            self.use_map = [use_theory_field[self.map_fields[i]]
                            for i in range(len(self.map_names))]
        # Bandpowers can depend on more fields than are actually used in likelihood
        # e.g. for correcting leakage or other linear corrections
        self.require_map = self.use_map[:]
        if self.has_map_names:
            if ini.hasKey('fields_required'):
                io_mp.LikelihoodError('use maps_required not fields_required')
            maps_use = ini.split('maps_required', [])
        else:
            maps_use = ini.split('fields_required', [])
        if len(maps_use):
            for j, map_used in enumerate(maps_use):
                if map_used in self.map_names:
                    self.require_map[self.map_names.index(map_used)] = True
                else:
                    io_mp.LikelihoodError('required item not found %s' % map_used)
        self.required_theory_field = [False for _ in self.field_names]
        for i in range(len(self.map_names)):
            if self.require_map[i]:
                self.required_theory_field[self.map_fields[i]] = True
        self.ncl_used = 0  # set later reading covmat
        self.like_approx = ini.string('like_approx', 'gaussian')
        self.nmaps = np.count_nonzero(self.use_map)
        self.nmaps_required = np.count_nonzero(self.require_map)
        self.required_order = np.zeros(self.nmaps_required, dtype=int)
        self.map_required_index = -np.ones(len(self.map_names), dtype=int)
        ix = 0
        for i in range(len(self.map_names)):
            if self.require_map[i]:
                self.map_required_index[i] = ix
                self.required_order[ix] = i
                ix += 1
        self.map_used_index = -np.ones(len(self.map_names), dtype=int)
        ix = 0
        self.used_map_order = []
        for i, map_name in enumerate(self.map_names):
            if self.use_map[i]:
                self.map_used_index[i] = ix
                self.used_map_order.append(map_name)
                ix += 1
        self.ncl = (self.nmaps * (self.nmaps + 1)) // 2
        self.pcl_lmax = ini.int('cl_lmax')
        self.pcl_lmin = ini.int('cl_lmin')
        self.binned = ini.bool('binned', True)
        if self.binned:
            self.nbins = ini.int('nbins')
            self.bin_min = ini.int('use_min', 1) - 1
            self.bin_max = ini.int('use_max', self.nbins) - 1
            # needed by read_bin_windows:
            self.nbins_used = self.bin_max - self.bin_min + 1
            self.bins = self.read_bin_windows(ini, 'bin_window')
        else:
            if self.nmaps != self.nmaps_required:
                io_mp.LikelihoodError('unbinned likelihood must have nmaps==nmaps_required')
            self.nbins = self.pcl_lmax - self.pcl_lmin + 1
            if self.like_approx != 'exact':
                print('Unbinned likelihoods untested in this version')
            self.bin_min = ini.int('use_min', self.pcl_lmin)
            self.bin_max = ini.int('use_max', self.pcl_lmax)
            self.nbins_used = self.bin_max - self.bin_min + 1
        self.full_bandpower_headers, self.full_bandpowers, self.bandpowers = \
            self.read_cl_array(ini, 'cl_hat', return_full=True)
        if self.like_approx == 'HL':
            self.cl_fiducial = self.read_cl_array(ini, 'cl_fiducial')
        else:
            self.cl_fiducial = None
        includes_noise = ini.bool('cl_hat_includes_noise', False)
        self.cl_noise = None
        if self.like_approx != 'gaussian' or includes_noise:
            self.cl_noise = self.read_cl_array(ini, 'cl_noise')
            if not includes_noise:
                self.bandpowers += self.cl_noise
            elif self.like_approx == 'gaussian':
                self.bandpowers -= self.cl_noise
        self.cl_lmax = np.zeros((self.tot_theory_fields, self.tot_theory_fields))
        for i in range(self.tot_theory_fields):
            if self.required_theory_field[i]:
                self.cl_lmax[i, i] = self.pcl_lmax
        if self.required_theory_field[0] and self.required_theory_field[1]:
            self.cl_lmax[1, 0] = self.pcl_lmax

        if self.like_approx != 'gaussian':
            cl_fiducial_includes_noise = ini.bool('cl_fiducial_includes_noise', False)
        else:
            cl_fiducial_includes_noise = False
        self.bandpower_matrix = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
        self.noise_matrix = self.bandpower_matrix.copy()
        self.fiducial_sqrt_matrix = self.bandpower_matrix.copy()
        if self.cl_fiducial is not None and not cl_fiducial_includes_noise:
            self.cl_fiducial += self.cl_noise
        for b in range(self.nbins_used):
            self.elements_to_matrix(self.bandpowers[:, b], self.bandpower_matrix[b, :, :])
            if self.cl_noise is not None:
                self.elements_to_matrix(self.cl_noise[:, b], self.noise_matrix[b, :, :])
            if self.cl_fiducial is not None:
                self.elements_to_matrix(self.cl_fiducial[:, b],
                                        self.fiducial_sqrt_matrix[b, :, :])
                self.fiducial_sqrt_matrix[b, :, :] = (
                    sqrtm(self.fiducial_sqrt_matrix[b, :, :]))
        if self.like_approx == 'exact':
            self.fsky = ini.float('fullsky_exact_fksy')
        else:
            self.cov = self.ReadCovmat(ini)
            self.covinv = np.linalg.inv(self.cov)
        if 'linear_correction_fiducial_file' in ini.params:
            self.fid_correction = self.read_cl_array(ini, 'linear_correction_fiducial')
            self.linear_correction = self.read_bin_windows(ini,
                                                           'linear_correction_bin_window')
        else:
            self.linear_correction = None
        if ini.hasKey('nuisance_params'):
            s = ini.relativeFileName('nuisance_params')
            self.nuisance_params = ParamNames(s)
            if ini.hasKey('calibration_param'):
                raise Exception('calibration_param not allowed with nuisance_params')
            if ini.hasKey('calibration_paramname'):
                self.calibration_param = ini.string('calibration_paramname')
            else:
                self.calibration_param = None
        elif ini.string('calibration_param', ''):
            s = ini.relativeFileName('calibration_param')
            if '.paramnames' not in s:
                raise io_mp.LikelihoodError('calibration_param must be paramnames file unless '
                              'nuisance_params also specified')
            self.nuisance_params = ParamNames(s)
            self.calibration_param = self.nuisance_params.list()[0]
        else:
            self.calibration_param = None
        if ini.hasKey('log_calibration_prior'):
            print('log_calibration_prior in .dataset ignored, '
                             'set separately in .yaml file')
        self.aberration_coeff = ini.float('aberration_coeff', 0.0)

        self.map_cls = self.init_map_cls(self.nmaps_required, self.required_order)

    def typeIndex(self, field):
        return self.field_names.index(field)

    def PairStringToMapIndices(self, S):
        if len(S) == 2:
            if self.has_map_names:
                raise io_mp.LikelihoodError('CL names must use MAP1xMAP2 names')
            return self.map_names.index(S[0]), self.map_names.index(S[1])
        else:
            try:
                i = S.index(self.map_separator)
            except ValueError:
                raise io_mp.LikelihoodError('invalid spectrum name %s' % S)
            return self.map_names.index(S[0:i]), self.map_names.index(S[i + 1:])

    def PairStringToUsedMapIndices(self, used_index, S):
        i1, i2 = self.PairStringToMapIndices(S)
        i1 = used_index[i1]
        i2 = used_index[i2]
        if i2 > i1:
            return i2, i1
        else:
            return i1, i2

    def UseString_to_cols(self, L):
        cl_i_j = self.UseString_to_Cl_i_j(L, self.map_used_index)
        cols = -np.ones(cl_i_j.shape[1], dtype=int)
        for i in range(cl_i_j.shape[1]):
            i1, i2 = cl_i_j[:, i]
            if i1 == -1 or i2 == -1:
                continue
            ix = 0
            for ii in range(self.nmaps):
                for jj in range(ii + 1):
                    if ii == i1 and jj == i2:
                        cols[i] = ix
                    ix += 1
        return cols

    def UseString_to_Cl_i_j(self, S, used_index):
        if not isinstance(S, (list, tuple)):
            S = S.split()
        cl_i_j = np.zeros((2, len(S)), dtype=int)
        for i, p in enumerate(S):
            cl_i_j[:, i] = self.PairStringToUsedMapIndices(used_index, p)
        return cl_i_j

    def MapPair_to_Theory_i_j(self, order, pair):
        i = self.map_fields[order[pair[0]]]
        j = self.map_fields[order[pair[1]]]
        if i <= j:
            return i, j
        else:
            return j, i

    def Cl_used_i_j_name(self, pair):
        return self.Cl_i_j_name(self.used_map_order, pair)

    def Cl_i_j_name(self, names, pair):
        name1 = names[pair[0]]
        name2 = names[pair[1]]
        if self.has_map_names:
            return name1 + self.map_separator + name2
        else:
            return name1 + name2

    def get_cols_from_order(self, order):
        # Converts string Order = TT TE EE XY... or AAAxBBB AAAxCCC BBxCC
        # into indices into array of power spectra (and -1 if not present)
        cols = np.empty(self.ncl, dtype=int)
        cols[:] = -1
        names = order.strip().split()
        ix = 0
        for i in range(self.nmaps):
            for j in range(i + 1):
                name = self.Cl_used_i_j_name([i, j])
                if name not in names and i != j:
                    name = self.Cl_used_i_j_name([j, i])
                if name in names:
                    if cols[ix] != -1:
                        raise io_mp.LikelihoodError('get_cols_from_order: duplicate CL type')
                    cols[ix] = names.index(name)
                ix += 1
        return cols

    def elements_to_matrix(self, X, M):
        ix = 0
        for i in range(self.nmaps):
            M[i, 0:i] = X[ix:ix + i]
            M[0:i, i] = X[ix:ix + i]
            ix += i
            M[i, i] = X[ix]
            ix += 1

    def matrix_to_elements(self, M, X):
        ix = 0
        for i in range(self.nmaps):
            X[ix:ix + i + 1] = M[i, 0:i + 1]
            ix += i + 1

    def read_cl_array(self, ini, file_stem, return_full=False):
        # read file of CL or bins (indexed by L)
        filename = ini.relativeFileName(file_stem + '_file')
        cl = np.zeros((self.ncl, self.nbins_used))
        order = ini.string(file_stem + '_order', '')
        if not order:
            incols = last_top_comment(filename)
            if not incols:
                raise io_mp.LikelihoodError('No column order given for ' + filename)
        else:
            incols = 'L ' + order
        cols = self.get_cols_from_order(incols)
        data = np.loadtxt(filename)
        Ls = data[:, 0].astype(int)
        if self.binned:
            Ls -= 1
        for i, L in enumerate(Ls):
            if self.bin_min <= L <= self.bin_max:
                for ix in range(self.ncl):
                    if cols[ix] != -1:
                        cl[ix, L - self.bin_min] = data[i, cols[ix]]
        if Ls[-1] < self.bin_max:
            raise io_mp.LikelihoodError('CMBLikes_ReadClArr: C_l file does not go up to maximum used: '
                          '%s', self.bin_max)
        if return_full:
            return incols.split(), data, cl
        else:
            return cl

    def read_bin_windows(self, ini, file_stem):
        bins = BinWindows(self.pcl_lmin, self.pcl_lmax, self.nbins_used, self.ncl)
        in_cl = ini.split(file_stem + '_in_order')
        out_cl = ini.split(file_stem + '_out_order', in_cl)
        bins.cols_in = self.UseString_to_Cl_i_j(in_cl, self.map_required_index)
        bins.cols_out = self.UseString_to_cols(out_cl)
        norder = bins.cols_in.shape[1]
        if norder != bins.cols_out.shape[0]:
            raise io_mp.LikelihoodError('_in_order and _out_order must have same number of entries')
        bins.binning_matrix = np.zeros(
            (norder, self.nbins_used, self.pcl_lmax - self.pcl_lmin + 1))
        windows = ini.relativeFileName(file_stem + '_files')
        for b in range(self.nbins_used):
            window = np.loadtxt(windows % (b + 1 + self.bin_min))
            err = False
            for i, L in enumerate(window[:, 0].astype(int)):
                if self.pcl_lmin <= L <= self.pcl_lmax:
                    bins.binning_matrix[:, b, L - self.pcl_lmin] = window[i, 1:]
                else:
                    err = err or any(window[i, 1:] != 0)
            if err:
                print('%s %u outside pcl_lmin-cl_max range: %s' %
                                 (file_stem, b, windows % (b + 1)))
        if ini.hasKey(file_stem + '_fix_cl_file'):
            raise io_mp.LikelihoodError('fix_cl_file not implemented yet')
        return bins

    def init_map_cls(self, nmaps, order):
        if nmaps != len(order):
            raise io_mp.LikelihoodError('init_map_cls: size mismatch')

        class CrossPowerSpectrum:
            map_ij: List[int]
            theory_ij: List[int]
            CL: np.ndarray

        cls = np.empty((nmaps, nmaps), dtype=object)
        for i in range(nmaps):
            for j in range(i + 1):
                CL = CrossPowerSpectrum()
                cls[i, j] = CL
                CL.map_ij = [order[i], order[j]]
                CL.theory_ij = self.MapPair_to_Theory_i_j(order, [i, j])
                CL.CL = np.zeros(self.pcl_lmax - self.pcl_lmin + 1)
        return cls
    
    def ReadCovmat(self, ini):
        """Read the covariance matrix, and the array of which CL are in the covariance,
        which then defines which set of bandpowers are used
        (subject to other restrictions)."""
        covmat_cl = ini.string('covmat_cl', allowEmpty=False)
        self.full_cov = np.loadtxt(ini.relativeFileName('covmat_fiducial'))
        covmat_scale = ini.float('covmat_scale', 1.0)
        cl_in_index = self.UseString_to_cols(covmat_cl)
        self.ncl_used = np.sum(cl_in_index >= 0)
        self.cl_used_index = np.zeros(self.ncl_used, dtype=int)
        cov_cl_used = np.zeros(self.ncl_used, dtype=int)
        ix = 0
        for i, index in enumerate(cl_in_index):
            if index >= 0:
                self.cl_used_index[ix] = index
                cov_cl_used[ix] = i
                ix += 1
        if self.binned:
            num_in = len(cl_in_index)
            pcov = np.empty((self.nbins_used * self.ncl_used,
                             self.nbins_used * self.ncl_used))
            for binx in range(self.nbins_used):
                for biny in range(self.nbins_used):
                    pcov[binx * self.ncl_used: (binx + 1) * self.ncl_used,
                    biny * self.ncl_used: (biny + 1) * self.ncl_used] = (
                            covmat_scale * self.full_cov[
                        np.ix_((binx + self.bin_min) * num_in + cov_cl_used,
                               (biny + self.bin_min) * num_in + cov_cl_used)])
        else:
            raise io_mp.LikelihoodError('unbinned covariance not implemented yet')
        return pcov

    # noinspection PyTypeChecker
    def writeData(self, froot):  # pragma: no cover
        np.savetxt(froot + '_cov.dat', self.cov)
        np.savetxt(froot + '_bandpowers.dat', self.full_bandpowers,
                   header=" ".join(self.full_bandpower_headers))
        self.bins.write(froot, 'bin')
        if self.linear_correction is not None:
            self.linear_correction.write(froot, 'linear_correction_bin')

            with open(froot + '_lensing_fiducial_correction', 'w', encoding="utf-8") as f:
                f.write("#%4s %12s \n" % ('bin', 'PP'))
                for b in range(self.nbins):
                    f.write("%5u %12.5e\n" % (b + 1, self.fid_correction[b]))

    def diag_sigma(self):
        return np.sqrt(np.diag(self.full_cov))

    def plot_lensing(self, cosmo, column='PP', ells=None, units="muK2", ax=None):
        if not np.count_nonzero(self.map_cls):
            raise io_mp.LikelihoodError("No Cl's have been computed yet. "
                          "Make sure you have evaluated the likelihood.")
        try:
            Cl_theo = self.get_cl(cosmo)
            Cl = Cl_theo.get(column.lower())
        except KeyError:
            raise io_mp.LikelihoodError("'%s' spectrum has not been computed." % column)
        import matplotlib.pyplot as plt
        lbin = self.full_bandpowers[:, self.full_bandpower_headers.index('L_av')]
        binned_phicl_err = self.diag_sigma()
        ax = ax or plt.gca()
        bandpowers = self.full_bandpowers[:, self.full_bandpower_headers.index('PP')]
        if 'L_min' in self.full_bandpower_headers:
            lmin = self.full_bandpowers[:, self.full_bandpower_headers.index('L_min')]
            lmax = self.full_bandpowers[:, self.full_bandpower_headers.index('L_max')]
            ax.errorbar(lbin, bandpowers,
                        yerr=binned_phicl_err, xerr=[lbin - lmin, lmax - lbin], fmt='o')
        else:
            ax.errorbar(lbin, bandpowers, yerr=binned_phicl_err, fmt='o')
        if ells is not None:
            Cl = Cl[ells]
        else:
            ells = Cl_theo["ell"]
        ax.plot(ells, Cl, color='k')
        ax.set_xlim([2, ells[-1]])
        return ax

    def get_binned_map_cls(self, Cls, corrections=True):
        band = self.bins.bin(Cls)
        if self.linear_correction is not None and corrections:
            band += self.linear_correction.bin(Cls) - self.fid_correction.T
        return band

    def get_theory_map_cls(self, Cls, data_params=None):
        for i in range(self.nmaps_required):
            for j in range(i + 1):
                CL = self.map_cls[i, j]
                combination = "".join([self.field_names[k] for k in CL.theory_ij]).lower()
                cls = Cls.get(combination)
                if cls is not None:
                    CL.CL[:] = cls[self.pcl_lmin:self.pcl_lmax + 1]
                else:
                    CL.CL[:] = 0
        self.adapt_theory_for_maps(self.map_cls, data_params or {})

    def adapt_theory_for_maps(self, cls, data_params):
        if self.aberration_coeff:
            self.add_aberration(cls)
        self.add_foregrounds(cls, data_params)
        if self.calibration_param is not None and self.calibration_param in data_params:
            for i in range(self.nmaps_required):
                for j in range(i + 1):
                    CL = cls[i, j]
                    if CL is not None:
                        if CL.theory_ij[0] <= 2 and CL.theory_ij[1] <= 2:
                            CL.CL /= data_params[self.calibration_param] ** 2

    def add_foregrounds(self, cls, data_params):
        pass

    def add_aberration(self, cls):
        # adapted from CosmoMC function by Christian Reichardt
        ells = np.arange(self.pcl_lmin, self.pcl_lmax + 1)
        cl_norm = ells * (ells + 1)
        for i in range(self.nmaps_required):
            for j in range(i + 1):
                CL = cls[i, j]
                if CL is not None:
                    if CL.theory_ij[0] <= 2 and CL.theory_ij[1] <= 2:
                        # first get Cl instead of Dl
                        cl_deriv = CL.CL / cl_norm
                        # second take derivative dCl/dl
                        cl_deriv[1:-1] = (cl_deriv[2:] - cl_deriv[:-2]) / 2
                        # handle endpoints approximately
                        cl_deriv[0] = cl_deriv[1]
                        cl_deriv[-1] = cl_deriv[-2]
                        # reapply to Dl's.
                        # note never took 2pi out, so not putting it back either
                        cl_deriv *= cl_norm
                        # also multiply by ell since really wanted ldCl/dl
                        cl_deriv *= ells
                        CL.CL += self.aberration_coeff * cl_deriv

    def write_likelihood_data(self, filename, data_params=None):
        cls = self.init_map_cls(self.nmaps_required, self.required_order)
        self.add_foregrounds(cls, data_params or {})
        with open(filename, 'w', encoding="utf-8") as f:
            cols = []
            for i in range(self.nmaps_required):
                for j in range(i + 1):
                    cols.append(self.Cl_i_j_name(self.map_names, cls[i, j].map_ij))
            f.write('#    L' + ("%17s " * len(cols)) % tuple(cols) + '\n')
            for b in range(self.pcl_lmin, self.pcl_lmax + 1):
                c = [b]
                for i in range(self.nmaps_required):
                    for j in range(i + 1):
                        c.append(cls[i, j].CL[b - self.pcl_lmin])
                f.write(("%I5 " + "%17.8e " * len(cols)) % tuple(c))

    @staticmethod
    def transform(C, Chat, Cfhalf):
        # HL transformation of the matrices
        if C.shape[0] == 1:
            rat = Chat[0, 0] / C[0, 0]
            C[0, 0] = (np.sign(rat - 1) *
                       np.sqrt(2 * np.maximum(0, rat - np.log(rat) - 1)) *
                       Cfhalf[0, 0] ** 2)
            return
        diag, U = np.linalg.eigh(C)
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        diag, rot = np.linalg.eigh(rot)
        diag = np.sign(diag - 1) * np.sqrt(2 * np.maximum(0, diag - np.log(diag) - 1))
        Cfhalf.dot(rot, U)
        for i, d in enumerate(diag):
            rot[:, i] = U[:, i] * d
        rot.dot(U.T, C)

    def exact_chi_sq(self, C, Chat, L):
        if C.shape[0] == 1:
            return ((2 * L + 1) * self.fsky *
                    (Chat[0, 0] / C[0, 0] - 1 - np.log(Chat[0, 0] / C[0, 0])))
        else:
            M = np.linalg.inv(C).dot(Chat)
            return ((2 * L + 1) * self.fsky *
                    (np.trace(M) - self.nmaps - np.linalg.slogdet(M)[1]))
        
    def loglkl(self, cosmo, data):
        dls = self.get_cl(cosmo, l_max=self.l_max)
        ells_factor = ((dls["ell"] + 1) * dls["ell"] / (2 * np.pi))[2:]
        for cl in dls:
            if cl not in ['pp', 'ell']:
                dls[cl][2:] *= ells_factor
            if cl == 'pp':
                dls['pp'][2:] *= ells_factor * ells_factor * (2 * np.pi)
        
        nuisance_pars = {}
        for par in self.use_nuisance:
            nuisance_pars[par] = data.mcmc_parameters[par]['current'] * data.mcmc_parameters[par]['scale']
        
        self.get_theory_map_cls(dls, nuisance_pars)
        C = np.empty((self.nmaps, self.nmaps))
        big_x = np.empty(self.nbins_used * self.ncl_used)
        vecp = np.empty(self.ncl)
        chisq = 0
        if self.binned:
            binned_theory = self.get_binned_map_cls(self.map_cls)
        else:
            Cs = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
            for i in range(self.nmaps):
                for j in range(i + 1):
                    CL = self.map_cls[i, j]
                    if CL is not None:
                        Cs[:, i, j] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
                        Cs[:, j, i] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
        for b in range(self.nbins_used):
            if self.binned:
                self.elements_to_matrix(binned_theory[b, :], C)
            else:
                C[:, :] = Cs[b, :, :]
            if self.cl_noise is not None:
                C += self.noise_matrix[b]
            if self.like_approx == 'exact':
                chisq += self.exact_chi_sq(
                    C, self.bandpower_matrix[b], self.bin_min + b)
                continue
            elif self.like_approx == 'HL':
                try:
                    self.transform(
                        C, self.bandpower_matrix[b], self.fiducial_sqrt_matrix[b])
                except np.linalg.LinAlgError:
                    print("Likelihood computation failed.")
                    return -np.inf
            elif self.like_approx == 'gaussian':
                C -= self.bandpower_matrix[b]
            self.matrix_to_elements(C, vecp)
            big_x[b * self.ncl_used:(b + 1) * self.ncl_used] = vecp[
                self.cl_used_index]
        if self.like_approx == 'exact':
            return -0.5 * chisq
        return -0.5 * self._fast_chi_squared(self.covinv, big_x)
    
class BinWindows:
    cols_in: np.ndarray
    cols_out: np.ndarray
    binning_matrix: np.ndarray

    def __init__(self, lmin, lmax, nbins, ncl):
        self.lmin = lmin
        self.lmax = lmax
        self.nbins = nbins
        self.ncl = ncl

    def bin(self, theory_cl, cls=None):
        if cls is None:
            cls = np.zeros((self.nbins, self.ncl))
        for i, ((x, y), ix_out) in enumerate(zip(self.cols_in.T, self.cols_out)):
            cl = theory_cl[x, y]
            if cl is not None and ix_out >= 0:
                cls[:, ix_out] += np.dot(self.binning_matrix[i, :, :], cl.CL)
        return cls

    def write(self, froot, stem):
        if not os.path.exists(froot + stem + '_window'):
            os.mkdir(froot + '_window')
        for b in range(self.nbins):
            with open(froot + stem + '_window/window%u.dat' % (b + 1),
                      'w', encoding="utf-8") as f:
                for L in np.arange(self.lmin[b], self.lmax[b] + 1):
                    f.write(
                        ("%5u " + "%10e" * len(self.cols_in) + "\n") %
                        (L, self.binning_matrix[:, b, L]))


def last_top_comment(fname):
    result = None
    with open(fname, encoding="utf-8-sig") as f:
        x = f.readline()
        while x:
            x = x.strip()
            if x:
                if x[0] != '#':
                    return result
                result = x[1:].strip()
            x = f.readline()
    return None


def white_noise_from_muK_arcmin(noise_muK_arcmin):
    return (noise_muK_arcmin * np.pi / 180 / 60.) ** 2


def save_cl_dict(filename, array_dict, lmin=2, lmax=None,
                 cl_dict_lmin=0):  # pragma: no cover
    """
    Save a Cobaya dict of CL to a text file, with each line starting with L.

    :param filename: filename to save
    :param array_dict: dictionary of power spectra
    :param lmin: minimum L to save
    :param lmax: maximum L to save
    :param cl_dict_lmin: L to start output in file (usually 0 or 2)
    """
    cols = []
    labels = []
    for key in CMB_keys:
        if key in array_dict:
            lmax = lmax or array_dict[key].shape[0] - 1 + cl_dict_lmin
            cols.append(array_dict[key][lmin - cl_dict_lmin:lmax - cl_dict_lmin + 1])
            labels.append(key.upper())
    if 'pp' in array_dict:
        lmax = lmax or array_dict['pp'].shape[0] - 1 + cl_dict_lmin
        cols.append(array_dict['pp'][lmin - cl_dict_lmin:lmax - cl_dict_lmin + 1])
        labels.append('PP')
    ls = np.arange(lmin, lmax + 1)
    np.savetxt(filename, np.vstack((ls,) + tuple(cols)).T,
               fmt=['%4u'] + ['%12.7e'] * len(cols),
               header=' L ' + ' '.join(['{:13s}'.format(lab) for lab in labels]))


def make_forecast_cmb_dataset(fiducial_Cl, output_root, output_dir=None,
                              noise_muK_arcmin_T=None,
                              noise_muK_arcmin_P=None,
                              NoiseVar=None, ENoiseFac=2, fwhm_arcmin=None,
                              lmin=2, lmax=None, fsky=1.0,
                              lens_recon_noise=None, cl_dict_lmin=0):  # pragma: no cover
    """
    Make a simulated .dataset and associated files with 'data' set at the input fiducial
    model. Uses the exact full-sky log-likelihood, scaled by fsky.

    If you want to use numerical N_L CMB noise files, you can just replace the noise
    .dat text file produced by this function.

    :param fiducial_Cl: dictionary of Cls to use, combination of tt, te, ee, bb, pp;
                        note te must be included with tt and ee when using them
    :param output_root: root name for output files, e.g. 'my_sim1'
    :param output_dir: output directory
    :param noise_muK_arcmin_T: temperature noise in muK-arcmin
    :param noise_muK_arcmin_P: polarization noise in muK-arcmin
    :param NoiseVar: alternatively if noise_muK_arcmin_T is None, effective
        isotropic noise variance for the temperature (N_L=NoiseVar with no beam)
    :param ENoiseFac: factor by which polarization noise variance is higher than
                NoiseVar (usually 2, for Planck about 4
                        as only half the detectors polarized)
    :param fwhm_arcmin: beam fwhm in arcminutes
    :param lmin: l_min
    :param lmax: l_max
    :param fsky: sky fraction
    :param lens_recon_noise: optional array, starting at L=0, for the
       pp lensing reconstruction noise, in [L(L+1)]^2C_L^phi/2pi units
    :param cl_dict_lmin: l_min for the arrays in fiducial_Cl
    :return: IniFile that was saved
    """
    ini = IniFile()
    dataset = ini.params

    cl_keys = fiducial_Cl.keys()
    use_CMB = set(cl_keys).intersection(CMB_keys)
    use_lensing = lens_recon_noise is not None

    if use_CMB:
        if NoiseVar is None:
            if noise_muK_arcmin_T is None:
                raise ValueError('Must specify noise')
            NoiseVar = white_noise_from_muK_arcmin(noise_muK_arcmin_T)
            if noise_muK_arcmin_P is not None:
                ENoiseFac = (noise_muK_arcmin_P / noise_muK_arcmin_T) ** 2
        elif noise_muK_arcmin_T is not None or noise_muK_arcmin_P is not None:
            raise ValueError('Specific either noise_muK_arcmin or NoiseVar')
        fields_use = ''
        if 'tt' in cl_keys or 'te' in cl_keys:
            fields_use = 'T'
        if 'ee' in cl_keys or 'te' in cl_keys:
            fields_use += ' E'
        if 'bb' in cl_keys:
            fields_use += ' B'
        if 'pp' in cl_keys and use_lensing:
            fields_use += ' P'
        if 'tt' in cl_keys and 'ee' in cl_keys and 'te' not in cl_keys:
            raise ValueError('Input power spectra should have te if using tt and ee -'
                             'using the exact likelihood requires the full covariance.')
    else:
        fields_use = 'P'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset['fields_use'] = fields_use

    if use_CMB:
        fwhm = fwhm_arcmin / 60
        xlc = 180 * np.sqrt(8. * np.log(2.)) / np.pi
        sigma2 = (fwhm / xlc) ** 2
        noise_cols = 'TT           EE          BB'
        if use_lensing:
            noise_cols += '          PP'
    elif use_lensing:
        noise_cols = 'PP'
    else:
        raise ValueError('Must use CMB or lensing C_L')
    noise_file = output_root + '_Noise.dat'
    with open(os.path.join(output_dir, noise_file), 'w') as f:
        f.write('#L %s\n' % noise_cols)

        for ell in range(lmin, lmax + 1):
            noises = []
            if use_CMB:
                # noinspection PyUnboundLocalVariable
                noise_cl = ell * (ell + 1.) / 2 / np.pi * NoiseVar * np.exp(
                    ell * (ell + 1) * sigma2)
                noises += [noise_cl, ENoiseFac * noise_cl, ENoiseFac * noise_cl]
            if use_lensing:
                noises += [lens_recon_noise[ell]]
            f.write("%d " % ell + " ".join("%E" % elem for elem in noises) + "\n")

    dataset['fullsky_exact_fksy'] = fsky
    dataset['dataset_format'] = 'CMBLike2'
    dataset['like_approx'] = 'exact'

    dataset['cl_lmin'] = lmin
    dataset['cl_lmax'] = lmax

    dataset['binned'] = False

    dataset['cl_hat_includes_noise'] = False

    save_cl_dict(os.path.join(output_dir, output_root + '.dat'),
                 fiducial_Cl, cl_dict_lmin=cl_dict_lmin)
    dataset['cl_hat_file'] = output_root + '.dat'
    dataset['cl_noise_file '] = noise_file

    ini.saveFile(os.path.join(output_dir, output_root + '.dataset'))
    return ini


###################################
# Planck NPIPE Camspec Likelihood
# by A. Lewis
# adapted to montepython by Gen Ye
###################################
class Likelihood_camspec(Likelihood_dataset):

    def __init__(self, path, data, command_line):
        Likelihood_dataset.__init__(self, path, data, command_line)

    def init_params(self, ini, silent=False):
        spectra = np.loadtxt(ini.relativeFileName('cl_hat_file'))
        covmat_cl = ini.split('covmat_cl')
        self.use_cl = ini.split('use_cl', covmat_cl)
        if ini.hasKey('use_range'):
            used_ell = ini.params['use_range']
            if isinstance(used_ell, dict):
                print('Using range %s' % used_ell)
                for key, value in used_ell.items():
                    used_ell[key] = self.range_to_ells(value)
            else:
                if silent:
                    print('CamSpec using range: %s' % used_ell)
                used_ell = self.range_to_ells(used_ell)
        else:
            used_ell = None
        data_vector = []
        nX = 0
        used_indices = []
        with open(ini.relativeFileName('data_ranges'), "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            while not lines[-1].strip():
                lines = lines[:-1]
            self.Nspec = len(lines)
            lmin = np.zeros(self.Nspec, dtype=int)
            lmax = np.zeros(self.Nspec, dtype=int)
            self.cl_names = []
            self.ell_ranges = np.empty(self.Nspec, dtype=object)
            self.used_sizes = np.zeros(self.Nspec, dtype=int)
            for i, line in enumerate(lines):
                items = line.split()
                tp = items[0]
                self.cl_names.append(tp)
                lmin[i], lmax[i] = [int(x) for x in items[1:]]
                if lmax[i] and lmax[i] >= lmin[i]:
                    n = lmax[i] - lmin[i] + 1
                    data_vector.append(spectra[lmin[i]:lmax[i] + 1, i])
                    if tp in self.use_cl:
                        if used_ell is not None and (
                                not isinstance(used_ell, dict) or tp in used_ell):
                            if isinstance(used_ell, dict):
                                ells = used_ell[tp]
                            else:
                                ells = used_ell
                            self.ell_ranges[i] = np.array(
                                [L for L in range(lmin[i], lmax[i] + 1) if L in ells],
                                dtype=int)
                            used_indices.append(self.ell_ranges[i] + (nX - lmin[i]))
                        else:
                            used_indices.append(range(nX, nX + n))
                            self.ell_ranges[i] = range(lmin[i], lmax[i] + 1)
                        self.used_sizes[i] = len(self.ell_ranges[i])
                    else:
                        lmax[i] = -1
                    nX += n

        self.cl_used = np.array([name in self.use_cl for name in self.cl_names],
                                dtype=bool)
        covfile = ini.relativeFileName('covmat_fiducial')
        with open(covfile, "rb") as cov_f:
            cov = np.fromfile(cov_f, dtype=[np.float32, np.float64]['64.bin' in covfile])
        assert (nX ** 2 == cov.shape[0])
        used_indices = np.concatenate(used_indices)
        self.data_vector = np.concatenate(data_vector)[used_indices]
        self.cov = cov.reshape(nX, nX)[np.ix_(used_indices, used_indices)].astype(
            np.float64)
        if not silent:
            for name, mn, mx in zip(self.cl_names, lmin, lmax):
                if name in self.use_cl:
                    print(name, mn, mx)
            print('Number of data points: %s' % self.cov.shape[0])
        self.lmax = lmax
        self.lmin = lmin
        max_l = np.max(self.lmax)
        self.ls = np.arange(max_l + 1)
        self.llp1 = self.ls * (self.ls + 1)

        if np.any(self.cl_used[:4]):
            pivot = 3000
            self.sz_143 = self.read_normalized(
                ini.relativeFileName('sz143file'), pivot)[:max_l + 1]
            self.ksz = self.read_normalized(
                ini.relativeFileName('kszfile'), pivot)[:max_l + 1]
            self.tszxcib = self.read_normalized(
                ini.relativeFileName('tszxcibfile'), pivot)[:max_l + 1]

            self.cib_217 = self.read_normalized(
                ini.relativeFileName('cib217file'), pivot)[:max_l + 1]

            self.dust = np.vstack(
                (self.read_normalized(ini.relativeFileName('dust100file'))[:max_l + 1],
                 self.read_normalized(ini.relativeFileName('dust143file'))[:max_l + 1],
                 self.read_normalized(ini.relativeFileName('dust217file'))[:max_l + 1],
                 self.read_normalized(ini.relativeFileName('dust143x217file'))[
                 :max_l + 1]))
            self.lnrat = self.ls * 0
            l_min = np.min(lmin[self.cl_used])
            self.lnrat[l_min:] = np.log(self.ls[l_min:] / np.float64(pivot))

        import hashlib
        cache_file = self.dataset_filename.replace('.dataset',
                                                   '_covinv_%s.npy' % hashlib.md5(
                                                       str(ini.params).encode(
                                                           'utf8')).hexdigest())
        if self.use_cache and os.path.exists(cache_file):
            self.covinv = np.load(cache_file).astype(np.float64)
        else:
            self.covinv = np.linalg.inv(self.cov)
            if self.use_cache:
                np.save(cache_file, self.covinv.astype(np.float32))
    
    def range_to_ells(self, use_range):
        """splits range string like '2-5 7 15-3000' into list of specific numbers"""

        if isinstance(use_range, str):
            ranges = []
            for ell_range in use_range.split():
                if '-' in ell_range:
                    mn, mx = [int(x) for x in ell_range.split('-')]
                    ranges.append(range(mn, mx + 1))
                else:
                    ranges.append(int(ell_range))
            return np.concatenate(ranges)
        else:
            return use_range
    
    def read_normalized(self, filename, pivot=None):
        # arrays all based at L=0, in L(L+1)/2pi units
        print('Loading: ', filename)
        dat = np.loadtxt(filename)
        assert int(dat[0, 0]) == 2
        dat = np.hstack(([0, 0], dat[:, 1]))
        if pivot is not None:
            assert pivot < dat.shape[0] + 2
            dat /= dat[pivot]
        return dat

    def get_foregrounds(self, data_params):

        sz_bandpass100_nom143 = 2.022
        cib_bandpass143_nom143 = 1.134
        sz_bandpass143_nom143 = 0.95
        cib_bandpass217_nom217 = 1.33

        Aps = np.empty(4)
        Aps[0] = data_params['aps100']
        Aps[1] = data_params['aps143']
        Aps[2] = data_params['aps217']
        Aps[3] = data_params['psr'] * np.sqrt(Aps[1] * Aps[2])
        Aps *= 1e-6 / 9  # scaling convention

        Adust = np.atleast_2d(
            [data_params['dust100'], data_params['dust143'], data_params['dust217'],
             data_params['dust143x217']]).T

        acib143 = data_params.get('acib143', -1)
        acib217 = data_params['acib217']
        cibr = data_params['cibr']
        ncib = data_params['ncib']
        cibrun = data_params['cibrun']

        asz143 = data_params['asz143']
        xi = data_params['xi']
        aksz = data_params['aksz']

        lmax = np.max(self.lmax)

        cl_cib = np.exp(ncib * self.lnrat + cibrun * self.lnrat ** 2 / 2) * self.cib_217
        if acib143 < 0:
            # fix 143 from 217
            acib143 = .094 * acib217 / cib_bandpass143_nom143 * cib_bandpass217_nom217
            # The above came from ratioing Paolo's templates, which were already
            # colour-corrected, and assumed perfect correlation

        ksz = aksz * self.ksz
        C_foregrounds = np.empty((4, lmax + 1))
        # 100
        C_foregrounds[0, :] = ksz + asz143 * sz_bandpass100_nom143 * self.sz_143

        # 143
        A_sz_143_bandpass = asz143 * sz_bandpass143_nom143
        A_cib_143_bandpass = acib143 * cib_bandpass143_nom143
        zCIB = A_cib_143_bandpass * cl_cib
        C_foregrounds[1, :] = (zCIB + ksz + A_sz_143_bandpass * self.sz_143
                               - 2.0 * np.sqrt(
                    A_cib_143_bandpass * A_sz_143_bandpass) * xi * self.tszxcib)

        # 217
        A_cib_217_bandpass = acib217 * cib_bandpass217_nom217
        zCIB = A_cib_217_bandpass * cl_cib
        C_foregrounds[2, :] = zCIB + ksz

        # 143x217
        zCIB = np.sqrt(A_cib_143_bandpass * A_cib_217_bandpass) * cl_cib
        C_foregrounds[3, :] = (cibr * zCIB + ksz - np.sqrt(
            A_cib_217_bandpass * A_sz_143_bandpass) * xi * self.tszxcib)

        # Add dust and point sources
        C_foregrounds += Adust * self.dust + np.outer(Aps, self.llp1)

        return C_foregrounds

    def get_cals(self, data_params):
        calPlanck = data_params.get('A_planck', 1) ** 2
        cal0 = data_params.get('cal0', 1)
        cal2 = data_params.get('cal2', 1)
        calTE = data_params.get('calTE', 1)
        calEE = data_params.get('calEE', 1)
        return np.array([cal0, 1, cal2, np.sqrt(cal2), calTE, calEE]) * calPlanck

    def chi_squared(self, CT, CTE, CEE, data_params):

        cals = self.get_cals(data_params)
        if np.any(self.cl_used[:4]):
            foregrounds = self.get_foregrounds(data_params)
        delta_vector = self.data_vector.copy()
        ix = 0
        for i, (cal, n) in enumerate(zip(cals, self.used_sizes)):
            if n > 0:
                if i <= 3:
                    # noinspection PyUnboundLocalVariable
                    delta_vector[ix:ix + n] -= (CT[self.ell_ranges[i]] +
                                                foregrounds[i][self.ell_ranges[i]]) / cal
                elif i == 4:
                    delta_vector[ix:ix + n] -= CTE[self.ell_ranges[i]] / cal
                elif i == 5:
                    delta_vector[ix:ix + n] -= CEE[self.ell_ranges[i]] / cal
                ix += n
        return self._fast_chi_squared(self.covinv, delta_vector)

    def loglkl(self, cosmo, data):
        dls = self.get_cl(cosmo)
        ells_factor = ((dls["ell"] + 1) * dls["ell"] / (2 * np.pi))[2:]
        for cl in dls:
            if cl not in ['pp', 'ell']:
                dls[cl][2:] *= ells_factor
            if cl == 'pp':
                dls['pp'][2:] *= ells_factor * ells_factor * (2 * np.pi)

        nuisance_pars = {}
        for par in self.use_nuisance:
            nuisance_pars[par] = data.mcmc_parameters[par]['current'] * data.mcmc_parameters[par]['scale']
        
        return -0.5 * self.chi_squared(dls.get('tt'), dls.get('te'), dls.get('ee'), nuisance_pars)

    def coadded_TT(self, data_params=None, foregrounds=None, cals=None,
                   want_cov=True, data_vector=None):
        nTT = np.sum(self.used_sizes[:4])
        assert nTT
        if foregrounds is not None and cals is not None and data_params is not None:
            raise ValueError('data_params not used')
        if foregrounds is None:
            assert data_params is not None
            foregrounds = self.get_foregrounds(data_params)
        if cals is None:
            assert data_params is not None
            cals = self.get_cals(data_params)
        if data_vector is None:
            data_vector = self.data_vector
        delta_vector = data_vector[:nTT].copy()
        cal_vector = np.zeros(delta_vector.shape)
        lmin = np.min([min(r) for r in self.ell_ranges[:4]])
        lmax = np.max([max(r) for r in self.ell_ranges[:4]])
        n_p = lmax - lmin + 1
        LS = np.zeros(delta_vector.shape, dtype=int)
        ix = 0
        for i, (cal, n) in enumerate(zip(cals[:4], self.used_sizes[:4])):
            if n > 0:
                delta_vector[ix:ix + n] -= foregrounds[i][self.ell_ranges[i]] / cal
                LS[ix:ix + n] = self.ell_ranges[i]
                cal_vector[ix:ix + n] = cal
                ix += n
        pcov = np.zeros((n_p, n_p))
        d = self.covinv[:nTT, :nTT].dot(delta_vector)
        dL = np.zeros(n_p)
        ix1 = 0
        ell_offsets = [LS - lmin for LS in self.ell_ranges[:4]]
        contiguous = not np.any(np.count_nonzero(LS - np.arange(LS[0],
                                                                LS[-1] + 1, dtype=int))
                                for LS in self.ell_ranges[:4])
        for i, (cal, LS, n) in enumerate(zip(cals[:4], ell_offsets, self.used_sizes[:4])):
            dL[LS] += d[ix1:ix1 + n] / cal
            ix = 0
            for cal2, r in zip(cals[:4], ell_offsets):
                if contiguous:
                    pcov[LS[0]:LS[0] + n, r[0]:r[0] + len(r)] += \
                        self.covinv[ix1:ix1 + n, ix:ix + len(r)] / (cal2 * cal)
                else:
                    pcov[np.ix_(LS, r)] += \
                        self.covinv[ix1:ix1 + n, ix:ix + len(r)] / (cal2 * cal)
                ix += len(r)
            ix1 += n

        CTot = np.zeros(self.ls[-1] + 1)
        if want_cov:
            pcovinv = np.linalg.inv(pcov)
            CTot[lmin:lmax + 1] = pcovinv.dot(dL)
            return CTot, pcovinv
        else:
            try:
                CTot[lmin:lmax + 1] = scipy.linalg.solve(pcov, dL, assume_a='pos')
            except:
                CTot[lmin:lmax + 1] = np.linalg.solve(pcov, dL)
            return CTot

    def get_weights(self, data_params):
        # get weights for each temperature spectrum as function of L
        ix = 0
        f = self.get_foregrounds(data_params) * 0
        weights = []
        for i in range(4):
            ells = self.ell_ranges[i]
            vec = np.zeros(self.data_vector.shape)
            vec[ix:ix + len(ells)] = 1
            Ti = self.coadded_TT(data_params, data_vector=vec, want_cov=False,
                                 foregrounds=f)
            weights.append((ells, Ti[ells]))
            ix += len(ells)
        return weights

    def diff(self, spec1, spec2, data_params):
        """
        Get difference (residual) between frequency spectra and the covariance
        :param spec1: name of spectrum 1
        :param spec2:  name of spectrum 2
        :param data_params: dictionary of parameters
        :return: ell range array, difference array, covariance matrix
        """
        foregrounds = self.get_foregrounds(data_params)
        cals = self.get_cals(data_params)
        i = self.cl_names.index(spec1)
        j = self.cl_names.index(spec2)
        off1 = np.sum(self.used_sizes[:i])
        off2 = np.sum(self.used_sizes[:j])
        lmax = np.min([max(r) for r in self.ell_ranges[[i, j]]])
        lmin = np.max([min(r) for r in self.ell_ranges[[i, j]]])

        diff = np.zeros(self.ls[-1] + 1)
        diff[self.ell_ranges[i]] = (self.data_vector[off1:off1 + self.used_sizes[i]] *
                                    cals[i] - foregrounds[i][self.ell_ranges[i]])
        diff[self.ell_ranges[j]] -= (
                self.data_vector[off2:off2 + self.used_sizes[j]] * cals[j] -
                foregrounds[j][
                    self.ell_ranges[j]])
        cov = self.cov
        n_p = lmax - lmin + 1
        off1 += lmin - np.min(self.ell_ranges[i])
        off2 += lmin - np.min(self.ell_ranges[j])
        pcov = cals[i] ** 2 * cov[off1:off1 + n_p, off1:off1 + n_p] \
               + cals[j] ** 2 * cov[off2:off2 + n_p, off2:off2 + n_p] \
               - cals[i] * cals[j] * (
                       cov[off2:off2 + n_p, off1:off1 + n_p] + cov[off1:off1 + n_p,
                                                               off2:off2 + n_p])
        return range(lmin, lmax + 1), diff[lmin:lmax + 1], pcov