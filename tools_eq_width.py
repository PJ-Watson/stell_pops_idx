# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:08:24 2020

@author: peter
"""

###################################################################################################

import numpy as np
import bottleneck as bn

from scipy.interpolate import InterpolatedUnivariateSpline

from astropy.table import Table

import matplotlib.pyplot as plt

import logging

###################################################################################################

    
class Index_Measure():
    
    '''Class to measure the absorption line indices of a given spectrum.
    The spectrum is assumed to have already been shifted to the rest frame.
    Note that the keys for the dicts below should all be named consistently.
    
    Parameters
    ----------                
    lam:        Dict of 1D arrays, the wavelengths of the spectra. Assumed to 
                be in Angstroms, although consistency with the index 
                definitions is the only real requirement.
                
    spec:       Dict of 1D arrays, the corresponding flux spectrum.
    
    spec_err:   Dict of 1D arrays, the error associated with the flux. If 
                none is given, this will assume the standard deviation of the 
                spectrum.
    
    dlam:       Float, the extra range outside of the bands to use for 
                interpolation. Will default 
                to 5 wavelength units.
                
    bands:      Nx7 array, containing the passbands of N indices and units, in
                the format given by Worthey et al.,  
                http://astro.wsu.edu/worthey/html/index.table.html . 
                Each row should be a separate index, with the wavelength 
                definitions sorted by column.
                The bounds of the index should be labelled "Index_1" and 
                "Index_2", the blue pseudocontinuum "Blue_1" and "Blue_2", and
                the red pseudocontinuum "Red_1" and "Red_2". 
                The "Units" column dictates the unit used, with 0 
                corresponding to Angstroms, and 1 corresponding to magnitudes.
                "Name" must be consistent with the keys for the dicts above.
                If no definitions are given, this will default to the standard
                Lick indices, including Balmer lines.
                
    plot:       Boolean, will produce plots of all index regions if True.
    
    no_error:   Boolean, will skip calculations of the width error if True.
    
    variance_weight:    Boolean. If True, will use the formulae of Cenarro et 
                        al. 2001, https://arxiv.org/abs/astro-ph/0109157v1 , 
                        to calculate the pseudo-continuum.
                        [Note that eq_width_2, the interpolated width, is used
                        instead of the pixel sum.]
    
    
    Attributes
    ----------
    eq_width:   1D array, the equivalent width of the indices.
    
    width_err:  1D array, the corresponding error (calculated from 100 
                                                   realisations).
 
    results:    Tuple, containing eq_width and width_err.
    '''

    def __init__(self, specs, lams, spec_errs = None, goodpixels = None,
                 dlam = 5, bands = None, plot = False,
                 no_error = False, realisations = 100, variance_weight = False):
        self.lams = lams
        self.specs = specs
        self.dlam = dlam
        self.plot = plot
        self.no_error = no_error
        self.realisations = realisations
        
        ### Sanity check for inputs
        assert self.specs.keys() == self.lams.keys(), \
            "The wavelength and flux inputs do not match.\n"
        if spec_errs is not None:
            self.spec_errs = spec_errs
            assert self.specs.keys() == self.spec_errs.keys(), \
                "The flux and flux error inputs do not match.\n"
                
        ### Initialise error spectrum
        else:
           
            self.spec_errs = dict()
            for k in self.specs.keys():
                self.spec_errs[k] = np.full_like(self.specs[k], bn.nanstd(self.specs[k]))
                
        if goodpixels is not None:
            self.goodpixels = goodpixels
            assert self.specs.keys() == self.goodpixels.keys(), \
                "The flux and flux error inputs do not match.\n"
                
        ### Initialise error spectrum
        else:
           
            self.goodpixels = dict()
            for k in self.specs.keys():
                self.goodpixels[k] = np.full_like(self.specs[k], 1, dtype=int)
        
        ### Initialise index definitions
        if bands is not None:
            self.bands = bands
            assert all([*self.specs.keys()] == np.array(self.bands["Name"])), \
                "Mismatch between spectral slices and given index definitions.\n"
            
        else:
            try:
                self.bands = Table.read('templates/Lick_Indices.txt', format = 'ascii')
                assert all([*self.specs.keys()] == np.array(self.bands["Name"])), \
                ("Mismatch between spectral slices and assumed index definitions.\n" +
                 "Try passing the correct index definitions, or check the spectrum is " +
                 "being sliced correctly.")
            except Exception as e:
                logging.exception("Failed to initialise index definitions.")
                raise e
                
        self.variance_weight = variance_weight
        
        ### Performs the calculations and formats output
        self.main_routine()
        
###################################################################################################
        
    def main_routine(self):   
        
        '''Main method.
        Selects the correct method based on initialisation parameters above.
        '''
    
        self.eq_width = dict()        
        self.width_err = dict()
              
        for row in self.bands:
            
            n = row["Name"]
            
            if not self.variance_weight:
            
                self.eq_width[n] = self.index_measurement_simple(self.specs[n], 
                                                                 self.lams[n], 
                                                                 self.goodpixels[n], 
                                                                 row)
                
                if self.no_error:
                    self.width_err[n] = None
                else:
                    self.index_errors(n, row)
                    
            else:
                
                self.eq_width[n] = self.index_measurement_cenarro(self.specs[n], 
                                                                  self.spec_errs[n], 
                                                                  self.lams[n],
                                                                  self.goodpixels[n], 
                                                                  row)
                
                self.index_errors_cenarro(n, row)
                
        ### Formats output as a tuple
        # self.results = (self.eq_width, self.width_err)
        self.results = {
            "equivalent_widths":self.eq_width, 
            "equivalent_widths_err":self.width_err,
            }
        
        return
    
###################################################################################################
    
    def index_measurement_simple(self, spec, lam, gpix, b, no_plot = False):
        
        '''Returns the equivalent width. 
        Widths are measured in either Angstroms (or other equivalent 
        wavelength unit) or magnitudes. All pixels are weighted equally.
        
        Returns
        -------
        eq_width:   1D array, contains the equivalent widths.
        '''
            
        ab_idx = np.where((lam > b["Index_1"] - self.dlam) & 
                          (lam < b["Index_2"] + self.dlam))[0]
            
        # Check the index falls in the wavelength range
        if (b["Blue_1"] - self.dlam < lam[0]) or \
            (b["Red_2"] + self.dlam > lam[-1]):
            eq_width = np.na

        # Check the index has not been entirely replaced 
        elif np.sum(gpix[ab_idx])/len(gpix[ab_idx]) <= 0.75:
            eq_width = np.nan
        
        else:
            # Find indices for each band
            # ab_idx = np.where((lam > b["Index_1"] - self.dlam) & 
            #                   (lam < b["Index_2"] + self.dlam))[0]
            cb_idx = np.where((lam > b["Blue_1"] - self.dlam) & 
                              (lam < b["Blue_2"] + self.dlam))[0]
            cr_idx = np.where((lam > b["Red_1"] - self.dlam) & 
                              (lam < b["Red_2"] + self.dlam))[0]
            
            # Integrate and find the average pseudocontinuum flux
            sb = InterpolatedUnivariateSpline(lam[cb_idx], 
                                              spec[cb_idx], k = 1)
            pscb = sb.integral(b["Blue_1"], b["Blue_2"]) / (b["Blue_2"] - b["Blue_1"])
        
            sr = InterpolatedUnivariateSpline(lam[cr_idx], 
                                              spec[cr_idx], k = 1)
            pscr = sr.integral(b["Red_1"], b["Red_2"]) / (b["Red_2"] - b["Red_1"])
            
            # Calculate the midpoints of the continuum bandpasses
            mid1 = (b["Blue_1"] + b["Blue_2"]) / 2.
            mid2 = (b["Red_1"] + b["Red_2"]) / 2.
            
            # Calculate the average linear continuum flux
            c_lin = pscb + (pscr - pscb)/ (mid2 - mid1) * \
                        (lam[ab_idx] - mid1)
            
            # Integrate to find the line flux
            f_lin = InterpolatedUnivariateSpline(lam[ab_idx], \
                        spec[ab_idx] / c_lin / (b["Index_2"] - b["Index_1"]), k = 1)

            integral = f_lin.integral(b["Index_1"], b["Index_2"])
            
            # Convert the integral to a width in Angstroms or magnitudes
            if b["Units"] == 0:
                eq_width = (1 - integral) * (b["Index_2"] - b["Index_1"])
            elif b["Units"] == 1:
                eq_width = -2.5 * np.log10(integral)
                
            if self.plot and not no_plot:# and b["Name"] == "Hgamma_F":
                
                fig, ax = plt.subplots(figsize=(20,10))
                ax.plot(lam[cb_idx[0]:cr_idx[-1]],
                        spec[cb_idx[0]:cr_idx[-1]])
                
                ax.plot(lam[ab_idx],c_lin)
                ax.scatter(mid1, pscb, color='green')
                ax.scatter(mid2, pscr, color='green')
                
                plt.title(b["Name"])
                ax.axvspan(b["Index_1"], b["Index_2"], alpha=0.3, color='red')
                ax.axvspan(b["Blue_1"], b["Blue_2"], alpha=0.3, color='blue')
                ax.axvspan(b["Red_1"], b["Red_2"], alpha=0.3, color='blue')
                
                ax.text(ax.get_xlim()[0], ax.get_ylim()[0], eq_width,
                        fontsize=16)
                    
        return eq_width
    
###################################################################################################

    def index_errors(self, name, band):
        
        '''Returns the estimated errors on the index measurements, based on 
        100 realisations of the spectra with the supplied error array.
        
        Attributes
        ----------
        width_err:  1D array, contains the errors on all indices.
        '''
        
        tmp_idx = np.zeros(self.realisations)
        
        for i in range(self.realisations):
            tmp_spec = self.specs[name] + np.random.normal(scale = self.spec_errs[name])
            tmp_idx[i] = self.index_measurement_simple(tmp_spec, self.lams[name],
                                                       self.goodpixels[name], band,
                                                       no_plot = True)
            
        self.width_err[name] = bn.nanstd(tmp_idx, axis = 0)
        
        return
    
###################################################################################################
        
    def index_measurement_cenarro(self, spec, spec_err, lam, gpix, b, no_plot = False):
        
        '''Returns the equivalent width. 
        Widths are measured in either Angstroms (or other equivalent 
        wavelength unit) or magnitudes. Pixels are weighted by their variance,
        as in Cenarro et al., 2001.
        
        Returns
        -------
        eq_width_2:   1D array, contains the equivalent widths.
        
        [Can also return eq_width, which uses the pixel sums of Cenarro's 
        full formula.]
        '''
        
        ### Find the array location for the absorption feature
        ab_idx = np.where((lam >= b["Index_1"]) &
                          (lam <= b["Index_2"]))[0]
        blue_idx = np.where((lam >= b["Blue_1"]) &
                          (lam <= b["Blue_2"]))[0]
        red_idx = np.where((lam >= b["Red_1"]) &
                          (lam <= b["Red_2"]))[0]
            
        ### Check the index falls in the wavelength range
        if (b["Blue_1"] - self.dlam < lam[0]) or \
            (b["Red_2"] + self.dlam > lam[-1]):
            # eq_width = np.nan
            eq_width_2 = np.nan
            
        # Check the index has not been entirely replaced
        # elif np.all(gpix[ab_idx] == 0):
        #     eq_width_2 = np.nan
            
        elif np.sum(gpix[ab_idx])/len(gpix[ab_idx]) <= 0.75:
            eq_width_2 = np.nan
            
        elif np.sum(gpix[blue_idx])/len(gpix[blue_idx]) <= 0.75:
            eq_width_2 = np.nan
        elif np.sum(gpix[red_idx])/len(gpix[red_idx]) <= 0.75:
            eq_width_2 = np.nan
            
            
        else:
            ### Find indices for each band
            # ab_idx = np.where((lam >= b["Index_1"]) &
            #                   (lam <= b["Index_2"]))[0]
            ab_idx_wide = np.where((lam >= b["Index_1"] - self.dlam) & 
                              (lam <= b["Index_2"] + self.dlam))[0]
            cb_idx = np.where((lam >= b["Blue_1"]) &
                              (lam <= b["Blue_2"]))[0]
            cr_idx = np.where((lam >= b["Red_1"]) &
                              (lam <= b["Red_2"]))[0]
            
            variance = spec_err**2
            
            a_15, a_16, a_17, a_18, a_19 = 0., 0., 0., 0., 0.
            
            for indices in [cb_idx, cr_idx]:
                
                a_15 += bn.nansum(1/variance[indices])
                
                a_16 += bn.nansum(lam[indices]/variance[indices])
                
                a_17 += bn.nansum((lam[indices]**2)/variance[indices])
                
                a_18 += bn.nansum(spec[indices]/variance[indices])
                
                a_19 += bn.nansum(lam[indices]*spec[indices]/variance[indices])
            
            delta = a_15*a_17 - a_16**2
            
            alpha_1 = 1/delta * (a_17*a_18 - a_16*a_19)
            
            alpha_2 = 1/delta * (a_15*a_19 - a_16*a_18)
            
            continuum = alpha_1 + alpha_2*lam
            
            
            # c_lin = alpha_1 + alpha_2*lam[ab_idx]
            
            # atomic_index = 1.05031753786 * bn.nansum(1-(spec[ab_idx]/c_lin))
            
            # if b["Units"] == 0:
            #     eq_width = atomic_index
            # elif b["Units"] == 1:
            #     eq_width = -2.5 * np.log10(1 - atomic_index/(b["Index_2"] - b["Index_1"]))
            
            
            c_lin_wide = alpha_1 + alpha_2*lam[ab_idx_wide]
            
            ### Integrate to find the line flux
            f_lin = InterpolatedUnivariateSpline(lam[ab_idx_wide], \
                        spec[ab_idx_wide] / c_lin_wide / (b["Index_2"] - b["Index_1"]), k = 1)

            integral = f_lin.integral(b["Index_1"], b["Index_2"])
            
            ### Convert the integral to a width in Angstroms or magnitudes
            if b["Units"] == 0:
                eq_width_2 = (1 - integral) * (b["Index_2"] - b["Index_1"])
            elif b["Units"] == 1:
                eq_width_2 = -2.5 * np.log10(integral)
                
            if self.plot and not no_plot and b["Name"]=="Mg_2":
                
                fig, ax = plt.subplots(figsize=(15,8))
                plt.title(b["Name"], fontsize=16)
                ax.plot(lam[cb_idx[0]:cr_idx[-1]],
                        spec[cb_idx[0]:cr_idx[-1]])
                ax.axvspan(b["Index_1"], b["Index_2"], alpha=0.3, color='green')
                ax.axvspan(b["Blue_1"], b["Blue_2"], alpha=0.3, color='blue')
                ax.axvspan(b["Red_1"], b["Red_2"], alpha=0.3, color='red')
                
                ax.text(b["Index_1"]-20, (0.85*(ax.get_ylim()[1]-ax.get_ylim()[0])
                                       +ax.get_ylim()[0]),
                        r"$\lambda_{c_1}$", fontsize=18)
                ax.text(b["Index_2"]+2, (0.85*(ax.get_ylim()[1]-ax.get_ylim()[0])
                                       +ax.get_ylim()[0]),
                        r"$\lambda_{c_2}$", fontsize=18)
                
                ax.plot(lam, continuum)
                
                ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
                plt.xlabel(r'$\rm{Galaxy\ Rest\ Frame\ Wavelength\ (Angstroms)}$', 
                           fontsize=18)
                plt.ylabel(r'$\rm{Relative\ Flux}$', fontsize=18)
                
                plt.show()
                    
        return eq_width_2
    
###################################################################################################

    def index_errors_cenarro(self, name, band):
        
        '''Returns the estimated errors on the index measurements, based on 
        100 realisations of the spectra with the supplied error array.
        
        Attributes
        ----------
        width_err:  1D array, contains the errors on all indices.
        '''
        
        tmp_idx = np.zeros(self.realisations)
        
        for i in range(self.realisations):
            tmp_spec = self.specs[name] + np.random.normal(scale = self.spec_errs[name])
            tmp_idx[i] = self.index_measurement_cenarro(tmp_spec, self.spec_errs[name],
                                                        self.lams[name], self.goodpixels[name],
                                                        band, no_plot = True)
            
        self.width_err[name] = bn.nanstd(tmp_idx, axis = 0)
        
        return
    
    
###################################################################################################
###################################################################################################
       
        
class Spectrum_Cut():
    
    def __init__(self, spectrum, wavelengths, spec_err = None, goodpixels = None,
                 bands = None, dlam = 30):
        
        self.spec = spectrum
        self.lam = wavelengths
        self.dlam = dlam
        self.spec_err = spec_err
        
        ### Initialise error spectrum
        if spec_err is not None:
            self.spec_err = spec_err
        else:
            self.spec_err = np.full_like(self.spec, bn.nanstd(self.spec))
            
        ### Initialise error spectrum
        if goodpixels is not None:
            self.goodpixels = goodpixels
        else:
            self.goodpixels = np.full_like(self.spec, 1, dtype=int)
        
        ### Initialise index definitions
        if bands is not None:
            self.bands = bands
        else:
            try:
                self.bands = Table.read("templates/Lick_Indices.txt",format = 'ascii')
            except Exception as e:
                logging.exception("Failed to initialise index definitions.")
                raise e
                
        self.main_routine()
        
###################################################################################################
        
    def main_routine(self):
        
        self.spectra_c = dict()
        self.lambda_c = dict()
        self.spec_err_c = dict()
        self.goodpixels_c = dict()
        
        for row in self.bands:
            idx = self.find_limits(row["Blue_1"], row["Red_2"])
            self.spectra_c["{}".format(row["Name"])] = self.spec[idx]
            self.lambda_c["{}".format(row["Name"])] = self.lam[idx]
            if self.spec_err is not None:
                self.spec_err_c["{}".format(row["Name"])] = self.spec_err[idx]
            else:
                self.spec_err_c["{}".format(row["Name"])] = None
            if self.goodpixels is not None:
                self.goodpixels_c["{}".format(row["Name"])] = self.goodpixels[idx]
            else:
                self.goodpixels_c["{}".format(row["Name"])] = None
        
        self.output = {
            "fluxes":self.spectra_c,
            "fluxes_err":self.spec_err_c,
            "wavelengths":self.lambda_c, 
            "good_pixels":self.goodpixels_c,
            }
        
        return
        
###################################################################################################
        
    def find_limits(self, low, high):
        
        indices = np.where((self.lam >= (low - self.dlam)) &
                           (self.lam <= (high + self.dlam)))
        
        return indices
        

###################################################################################################