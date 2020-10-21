# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:06:08 2020

@author: peter
"""

###################################################################################################

import numpy as np

from scipy.interpolate import interp1d

import ppxf.ppxf_util as util

###################################################################################################


class Convolutions():
    
    '''Class to convolve a spectrum to a desired resolution.
    The spectrum is assumed to have already been shifted to the rest frame.
    
    Parameters
    ----------                
    lam:        1D array, the wavelengths of the spectra. Assumed to be in Angstroms.
                
    flux:       1D array, the corresponding flux spectrum.
    
    vel:        Float, the peculiar velocity of the spectrum relative to the rest frame. Accounts 
                for slight shifts after cross correlation with templates, and will correct sigma, 
                z and lam for this effect.
                
    sigma:      Float, the intrinsic velocity dispersion of the spectrum (in km/s). 
    
    z:          Float, the redshift of the spectrum used to shift to the rest frame.
    
    FWHM_blue:  Float, the FWHM of the blue arm of the spectra. The default measurement, 2.6556 
                Angstroms, is taken from Van de Sande et al. 2017.
                
    FWHM_red:   Float, the FWHM of the red arm of the spectra. The default measurement, 1.5917 
                Angstroms, is taken from Van de Sande et al. 2017.
                
    resolution: Float. If using the LIS system, this is the resolution the spectrum will be 
                convolved to, otherwise it will be ignored. Defaults to 8.4 Angstroms.
    
    system:     String: 'IDS' or 'LIS'. 
                The system to convolve the spectra to.
                'IDS' -> Lick/IDS system, see Worthey & Ottaviani 1997.
                'LIS' -> Line Index System, see Vazdekis et al. 2010.
    
    
    Output
    ------
    IDS_flux:   1D array, the flux spectrum convolved to the appropriate resolution.
    (LIS_flux)
    
    IDS_flag:   Integer.
                0 -> The spectrum has been convolved without any problems.
                1 -> The combined instrumental and intrinsic dispersion was already higher than 
                     that required by the chosen system. The original flux spectrum has been 
                     returned instead.
    
    
    Attributes
    ----------
    output:     Tuple, containing the appropriate flux and flag.
    '''
    
    
<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, flux, lam, vel, sigma, z, FWHM_blue=2.6556, FWHM_red=1.5917, 
=======
    def __init__(self, lam, flux, vel, sigma, z, FWHM_blue=2.6556, FWHM_red=1.5917, 
>>>>>>> 7825a90... Previous work
=======
    def __init__(self, flux, lam, vel, sigma, z, FWHM_blue=2.6556, FWHM_red=1.5917, 
>>>>>>> ce01334... New
                 resolution = 8.4, system = 'IDS', flux_err = None):
        
        self.c = 299792.458
        self.factor = np.sqrt(8.*np.log(2.))
        self.flux = flux
        self.vel = vel
        self.FWHM_blue = FWHM_blue
        self.FWHM_red = FWHM_red
        self.resolution = resolution
        self.flux_err = flux_err
        self.lam = lam
        self.system = system
        
        ### Sanity check for inputs
        assert self.flux.keys() == self.lam.keys(), \
            "The wavelength and flux inputs do not match.\n"
        if self.flux_err is not None:
            assert self.flux.keys() == self.flux_err.keys(), \
                "The flux and flux error inputs do not match.\n"
        
        ### Correct parameters for peculiar velocity
        if self.vel != 0:
            for key in self.lam.keys():
                self.lam[key] = self.lam[key]/(1 + np.sqrt((self.c + self.vel)/
                                                           (self.c - self.vel)) - 1)
            self.z = z + np.sqrt((self.c + self.vel)/(self.c - self.vel)) - 1
            self.sigma = sigma * (np.sqrt((self.c - self.vel)/(self.c + self.vel)))
        else:
            self.z = z
            self.sigma = sigma
        
        self.main_routine()
    
###################################################################################################
        
    def main_routine(self):
    
        ### Initialise output dicts
        if self.system == 'IDS':
            self.IDS_flux = dict()
            self.IDS_flux_err = dict()
            self.IDS_flag = dict()
        elif self.system == 'LIS':
            self.LIS_flux = dict()
            self.LIS_flux_err = dict()
            self.LIS_flag = dict()
        else:
            raise Exception("Undefined index system. Please choose either 'IDS' or 'LIS'.")
        
        for key in self.flux.keys():
        
            ### Calculate the existing instrumental and intrinsic broadening
            self.obs_broad(key)
        
            ### Convolve the spectrum to the appropriate resolution
            if self.system == 'IDS':
                self.IDS_flux[key], self.IDS_flux_err[key], self.IDS_flag[key] \
                    = self.IDS_broadening(key)
            elif self.system == 'LIS':
                 self.LIS_flux[key], self.LIS_flux_err[key], self.LIS_flag[key] \
                    = self.LIS_broadening(key)
                    
        if self.system == 'IDS':
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> ce01334... New
            self.output = {
                "fluxes_convolved":self.IDS_flux, 
                "fluxes_convolved_err":self.IDS_flux_err,
                "convolution_flags":self.IDS_flag,
                }
<<<<<<< HEAD
        else:
            self.output = {
                "fluxes_convolved":self.LIS_flux, 
                "fluxes_convolved_err":self.LIS_flux_err,
                "convolution_flags":self.LIS_flag,
                }
=======
            self.output = (self.IDS_flux, self.IDS_flux_err, self.IDS_flag)
        else:
            self.output = (self.LIS_flux, self.LIS_flux_err, self.LIS_flag)
>>>>>>> 7825a90... Previous work
=======
        else:
            self.output = {
                "fluxes_convolved":self.LIS_flux, 
                "fluxes_convolved_err":self.LIS_flux_err,
                "convolution_flags":self.LIS_flag,
                }
>>>>>>> ce01334... New
            
        return
        
###################################################################################################
        
    def obs_broad(self, name):
        
        '''Calculates the combined instrumental and intrinsic broadening of the spectrum.
        
        Attributes
        ----------
        obs_FWHM:   1D array, the combined FWHM as a function of wavelength.
        '''
        
        # Calculate the wavelength spacing, may not be entirely accurate if lam was log binned
        self.dw = np.diff(self.lam[name])
        self.dw = np.append(self.dw,self.dw[-1])
    
        instr_FWHM_blue = self.FWHM_blue/(1+self.z)
        
        instr_FWHM = np.ones_like(self.lam[name]) * instr_FWHM_blue
                
        # sigma_corr = self.sigma * (np.sqrt((self.c - self.vel)/(self.c + self.vel)))
        vel_FWHM = self.lam[name] * (np.sqrt((self.sigma + self.c)/
                                       (self.c - self.sigma)) - 1) * self.factor
        
        self.obs_FWHM = np.sqrt(instr_FWHM**2 + vel_FWHM**2)
        
        return
        
###################################################################################################
        
    def IDS_broadening(self, name):
        
        '''Convolves the given spectrum to the Lick/IDS resolution.
        
        Returns
        -------
        IDS_flux:   1D array, the flux spectrum convolved to the appropriate resolution.
    
        IDS_flag:   Integer.
                    0 -> The spectrum has been convolved without any problems.
                    1 -> The combined instrumental and intrinsic dispersion was already higher than
                         that required by the Lick/IDS system. The original flux spectrum has been 
                         returned instead.
        '''
        
        lam_IDS = np.array([2000., 3000., 4000., 4400., 4900., 5400., 6000., 7000., 8000.])
        
        IDS_res = np.array([11.5, 11.5, 11.5, 9.2, 8.4, 8.4, 9.8, 9.8, 9.8])
        
        func_IDS = interp1d(lam_IDS, IDS_res, kind="linear", bounds_error=False,
                              fill_value="extrapolate")
        
        IDS_FWHM = func_IDS(self.lam[name])
        
        end = len(self.lam[name])
        
        for i, (obs,lick) in enumerate(zip(self.obs_FWHM, IDS_FWHM)):
            if obs**2 >= lick**2:
                end = i
                break
        
        # 2360 pixels was chosen as the end of the blue arm of the SAMI spectra, 
        # after log rebinning by pPXF.
        if end == len(self.lam[name]): 
        
            broad_sigma = np.sqrt(IDS_FWHM**2 - self.obs_FWHM**2) / \
                                                        self.dw / self.factor
            
            IDS_flux = util.gaussian_filter1d(self.flux[name], broad_sigma)
            
            IDS_flag = 0
            
            if self.flux_err[name] is not None:
                
                IDS_flux_err = util.gaussian_filter1d(self.flux_err[name], broad_sigma)
                
                return IDS_flux, IDS_flux_err, IDS_flag
            
            else:
                return IDS_flux, None, IDS_flag
        
        else:
            IDS_flux = np.copy(self.flux[name])
            
            IDS_flag = 1
            
            if self.flux_err[name] is not None:
                
                IDS_flux_err = np.copy(self.flux_err[name])
                
                return IDS_flux, IDS_flux_err, IDS_flag
            
            else:
                return IDS_flux, None, IDS_flag
        
###################################################################################################
        
    def LIS_broadening(self, name):
    
        '''Convolves the given spectrum to the LIS resolution, defaulting to 8.4 Angstroms.
        
        Returns
        -------
        LIS_flux:   1D array, the flux spectrum convolved to the appropriate resolution.
    
        LIS_flag:   Integer.
                    0 -> The spectrum has been convolved without any problems.
                    1 -> The combined instrumental and intrinsic dispersion was already higher than
                         that required by the Line Index System. The original flux spectrum has 
                         been returned instead.
        '''
        
        
        LIS_FWHM = np.full_like(self.lam[name], self.resolution)
        
        end = len(self.lam[name])
        
        for i, (obs, LIS) in enumerate(zip(self.obs_FWHM, LIS_FWHM)):
            if obs**2 >= LIS**2:
                end = i
                break
        
        if end == len(self.lam[name]):
        
            broad_sigma = np.sqrt(LIS_FWHM**2 - self.obs_FWHM**2) / \
                                                        self.dw / self.factor
            
            LIS_flux = util.gaussian_filter1d(self.flux[name], broad_sigma)
            
            LIS_flag = 0
            
            if self.flux_err[name] is not None:
                
                LIS_flux_err = util.gaussian_filter1d(self.flux_err[name], broad_sigma)
                
                return LIS_flux, LIS_flux_err, LIS_flag
            
            else:
                return LIS_flux, None, LIS_flag
        
        else:
            LIS_flux = np.copy(self.flux[name])
            
            LIS_flag = 1
            
            if self.flux_err[name] is not None:
                
                LIS_flux_err = np.copy(self.flux_err[name])
                
                return LIS_flux, LIS_flux_err, LIS_flag
            
            else:
                return LIS_flux, None, LIS_flag
        

###################################################################################################