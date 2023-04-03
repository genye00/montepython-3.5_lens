from scipy import interpolate
import os
import numpy as np
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood

class bbn(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        if "parthenope" in self.usedata:
          # Read grid 'parth' from Parthenope (omegab,DeltaN,YpBBN,DH) and prepare interpolation
          parth_data = np.loadtxt(os.path.join(self.data_directory, self.parthenopefile),usecols=(1,2,4,6))
          eta10_size_parth = 48
          deltaneff_size_parth = 11
          if(len(parth_data) != eta10_size_parth*deltaneff_size_parth):
            raise io_mp.LikelihoodError(
                "In likelihood %s: " % self.name +
                "BBN data from file '%s' " % os.path.join(self.data_directory, self.parthenopefile) +
                "in incorrect format.")

          # for omega_b, take first eta10_size points in 1st column
          eta10_parth_data = np.array(parth_data[:eta10_size_parth,0])
          # for deltaneff, take one point every eta10_size points in 2nd column
          deltaneff_parth_data = np.array(parth_data[::eta10_size_parth,1])
          ypbbn_parth_data = np.array(parth_data[:,2]).reshape(deltaneff_size_parth,eta10_size_parth)
          dh_parth_data = 1.e5*np.array(parth_data[:,3]).reshape(deltaneff_size_parth,eta10_size_parth)
          YpBBN_parth = interpolate.interp2d(eta10_parth_data,deltaneff_parth_data,ypbbn_parth_data,kind='cubic')
          DH_parth = interpolate.interp2d(eta10_parth_data,deltaneff_parth_data,dh_parth_data,kind='cubic')
          # ????
          dh_parth_DeltaN0 = 1.e5*np.array(parth_data[3*eta10_size_parth:4*eta10_size_parth,3])
          eta10_parth = interpolate.interp1d(dh_parth_DeltaN0,eta10_parth_data,kind='cubic')
          #print YpBBN_parth(0.022,0.)
          #print DH_parth(0.022,0.)
          self.get_Yp = YpBBN_parth
          self.get_DH = DH_parth

          self.get_Yperr = (lambda x,y: 0.5*self.two_sig_neutron_lifetime)
          self.get_DHerr = (lambda x,y: 0.5*self.two_sig_dh_parth)

          self.eta10_bounds = [eta10_parth_data[0],eta10_parth_data[-1]]
          self.dNeff_bounds = [deltaneff_parth_data[0],deltaneff_parth_data[-1]]

        elif "marcucci" in self.usedata:
          # Read grid 'marcucci' from Parthenope (omegab,DeltaN,YpBBN,DH) and prepare interpolation
          marcucci_data = np.loadtxt(os.path.join(self.data_directory, self.marcuccifile),usecols=(1,2,4,6))
          eta10_size_marcucci = 48
          deltaneff_size_marcucci = 11
          if(len(marcucci_data) != eta10_size_marcucci*deltaneff_size_marcucci):
            raise io_mp.LikelihoodError(
                "In likelihood %s: " % self.name +
                "BBN data from file '%s' " % os.path.join(self.data_directory, self.marcuccifile) +
                "in incorrect format.")
          # for omega_b, take first eta10_size_marcucci points in 1st column
          eta10_marcucci_data = np.array(marcucci_data[:eta10_size_marcucci,0])
          # for deltaneff, take one poiunt every eta10_size_marcucci points in 2nd column
          deltaneff_marcucci_data = np.array(marcucci_data[::eta10_size_marcucci,1])
          ypbbn_marcucci_data = np.array(marcucci_data[:,2]).reshape(deltaneff_size_marcucci,eta10_size_marcucci)
          dh_marcucci_data = 1.e5*np.array(marcucci_data[:,3]).reshape(deltaneff_size_marcucci,eta10_size_marcucci)
          YpBBN_marcucci = interpolate.interp2d(eta10_marcucci_data,deltaneff_marcucci_data,ypbbn_marcucci_data,kind='cubic')
          DH_marcucci = interpolate.interp2d(eta10_marcucci_data,deltaneff_marcucci_data,dh_marcucci_data,kind='cubic')
          # ????
          dh_marcucci_DeltaN0 = 1.e5*np.array(marcucci_data[3*eta10_size_marcucci:4*eta10_size_marcucci,3])
          eta10_marcucci = interpolate.interp1d(dh_marcucci_DeltaN0,eta10_marcucci_data,kind='cubic')
          #print YpBBN_marcucci(0.022,0.)
          #print DH_marcucci(0.022,0.)
          self.get_Yp = YpBBN_marcucci
          self.get_DH = DH_marcucci

          self.get_Yperr = (lambda x,y: 0.5*self.two_sig_neutron_lifetime)
          self.get_DHerr = (lambda x,y: 0.5*self.two_sig_dh_marcucci)

          self.eta10_bounds = [eta10_marcucci_data[0],eta10_marcucci_data[-1]]
          self.dNeff_bounds = [deltaneff_marcucci_data[0],deltaneff_marcucci_data[-1]]

        elif "primat" in self.usedata:
          # Read grid 'primat' from PRIMAT (omegab,DeltaN,YpBBN,DH) and prepare interpolation
          primat_data = np.loadtxt(os.path.join(self.data_directory, self.primatfile),usecols=(1,2,4,5,6,7))
          eta10_size_primat = 52
          deltaneff_size_primat = 25
          if(len(primat_data) != eta10_size_primat*deltaneff_size_primat):
            raise io_mp.LikelihoodError(
                "In likelihood %s: " % self.name +
                "BBN data from file '%s' " % os.path.join(self.data_directory, self.primatfile) +
                "in incorrect format.")
          # for omega_b, take first eta10_size_primat points in 1st column
          eta10_primat_data = np.array(primat_data[:eta10_size_primat,0])
          # for deltaneff, take one poiunt every eta10_size_primat points in 2nd column
          deltaneff_primat_data = np.array(primat_data[::eta10_size_primat,1])
          ypbbn_primat_data = np.array(primat_data[:,2]).reshape(deltaneff_size_primat,eta10_size_primat)
          dh_primat_data = 1.e5*np.array(primat_data[:,4]).reshape(deltaneff_size_primat,eta10_size_primat)
          YpBBN_primat = interpolate.interp2d(eta10_primat_data,deltaneff_primat_data,ypbbn_primat_data,kind='cubic')
          DH_primat = interpolate.interp2d(eta10_primat_data,deltaneff_primat_data,dh_primat_data,kind='cubic')
          # Primat stores also the errors of its estimation
          sigma_ypbbn_primat_data = np.array(primat_data[:,3]).reshape(deltaneff_size_primat,eta10_size_primat)
          sigma_dh_primat_data = 1.e5*np.array(primat_data[:,5]).reshape(deltaneff_size_primat,eta10_size_primat)
          sigma_YpBBN_primat = interpolate.interp2d(eta10_primat_data,deltaneff_primat_data,sigma_ypbbn_primat_data,kind='cubic')
          sigma_DH_primat = interpolate.interp2d(eta10_primat_data,deltaneff_primat_data,sigma_dh_primat_data,kind='cubic')
          # ????
          dh_primat_DeltaN0 = 1.e5*np.array(primat_data[3*eta10_size_primat:4*eta10_size_primat,3])
          eta10_primat = interpolate.interp1d(dh_primat_DeltaN0,eta10_primat_data,kind='cubic')
          #print YpBBN_primat(0.022,0.)
          #print DH_primat(0.022,0.)
          self.get_Yp = YpBBN_primat
          self.get_DH = DH_primat

          self.get_Yperr = sigma_YpBBN_primat
          self.get_DHerr = sigma_DH_primat

          self.eta10_bounds = [eta10_primat_data[0],eta10_primat_data[-1]]
          self.dNeff_bounds = [deltaneff_primat_data[0],deltaneff_primat_data[-1]]

        else:
            raise io_mp.LikelihoodError(
                "In likelihood %s: " % self.name +
                "unrecognized 'usedata' option %s." % self.usedata)

        if not (("dh" in self.include_bbn_type) or ("yp" in self.include_bbn_type)):
          raise io_mp.LikelihoodError(
              "In likelihood %s: " % self.name +
              "include_bbn_type ('%s') has to include either 'dh' or 'yp'." % self.include_bbn_type)

        # end of initialization


    # compute likelihood

    def loglkl(self, cosmo, data):

      YHe = cosmo.get_current_derived_parameters(['YHe'])['YHe']
      # TODO: change classy.pyx for more direct access to YHe
      # eta10 from eq.(9) in https://arxiv.org/pdf/0809.0631.pdf, assuming a standard value of Newton's constant
      eta10 = 273.45*cosmo.omega_b()/(1.-0.007*YHe)*pow(2.725/cosmo.T_cmb(),3)
      dNeff = cosmo.Neff()-self.Neff0

      #TODO :: ideally, we would want to get N_eff from a call to class at BBN time, not like this

      if(eta10 < self.eta10_bounds[0]):
        raise io_mp.LikelihoodError("The value of eta10 = %e was below the BBN table. Aborting."%eta10)
      if(eta10 > self.eta10_bounds[1]):
        raise io_mp.LikelihoodError("The value of eta10 = %e was above the BBN table. Aborting."%eta10)

      if(dNeff < self.dNeff_bounds[0]):
        raise io_mp.LikelihoodError("The value of delta Neff = %e was below the BBN table. Aborting."%dNeff)
      if(dNeff > self.dNeff_bounds[1]):
        raise io_mp.LikelihoodError("The value of delta Neff = %e was above the BBN table. Aborting."%dNeff)

      yp = self.get_Yp(eta10,dNeff)
      dh = self.get_DH(eta10,dNeff)
      yperr = self.get_Yperr(eta10,dNeff)
      dherr = self.get_DHerr(eta10,dNeff)

      try:
        if len(yp)>0:
          yp = yp[0]
        if len(dh)>0:
          dh = dh[0]
        if len(yperr)>0:
          yperr = yperr[0]
        if len(dherr)>0:
          dherr = dherr[0]
      except:
        pass
      #print("From (eta10,N_eff): Theoretical : Y_p = {:.5g} \pm {:.5g} , D_H = {:.5g} \pm {:.5g}".format(yp,yperr,dh,dherr))

      chi_square = 0.
      #Deal with deuterium
      if "dh" in self.include_bbn_type:
        chisquare_dh = (dh - self.dh_cooke_mean)**2/(self.dh_cooke_one_sig**2+dherr**2)
        chi_square += chisquare_dh
        #print("Chi square DH = ",chisquare_dh)

      #Deal with helium
      if "yp" in self.include_bbn_type:
        try:
          if 'cooke' in self.yp_measurement_type:
            if(yp>self.yp_cooke_mean):
              chisquare_yp = (yp - self.yp_cooke_mean)**2/(self.yp_cooke_one_sig_p**2+yperr**2)
            else:
              chisquare_yp = (yp - self.yp_cooke_mean)**2/(self.yp_cooke_one_sig_m**2+yperr**2)
          elif ('aver' in self.yp_measurement_type):
            chisquare_yp = (yp-self.yp_means['aver2015'])**2/(self.yp_sigs['aver2015']**2+yperr**2)
          elif ('peimbert' in self.yp_measurement_type):
            chisquare_yp = (yp-self.yp_means['peimbert2016'])**2/(self.yp_sigs['peimbert2016']**2+yperr**2)
          elif ('izotov' in self.yp_measurement_type):
            chisquare_yp = (yp-self.yp_means['izotov2014'])**2/(self.yp_sigs['izotov2014']**2+yperr**2)
          else:
            raise io_mp.LikelihoodError("Unrecognized experimental value of yp")
        except KeyError:
          raise io_mp.LikelihoodError("Unrecognized experimental value of yp")
        chi_square += chisquare_yp
        #print("Chi square YP = ",chisquare_yp)

      return -0.5*chi_square
