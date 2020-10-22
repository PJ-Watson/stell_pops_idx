# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:16:17 2020

@author: peter
"""

###################################################################################################

import numpy as np
import bottleneck as bn

import ppxf.ppxf_util as util

import astropy.io.fits as pf
from astropy.table import Table

# import numpy.lib.recfunctions as rf

import matplotlib.pyplot as plt

import glob, logging, os, pickle, copy, pathlib

from tools_eq_width import Index_Measure, Spectrum_Cut

from tools_convolution import Convolutions

###################################################################################################


class Dispersion_Correction():
    
    '''Class to calculate velocity dispersion corrections to Lick/IDS indices.
    The corrections are calculated using the MILES SSP models, from Vazdekis 
    et al. 2010. Other models with the same naming convention can be used 
    without any problem; models with a different name format will require a 
    few changes to make sure the parameters are being extracted correctly.
    
    Parameters
    ----------                
    temp_dir:   String, the directory containing all the necessary SSP models.
                
    MILES_FWHM: Float, the spectral FWHM of the SSP models, assumed to be in 
                Angstroms.
    
    sigma_min:  Float, the minimum velocity dispersion to consider for 
                correction.
    
    sigma_max:  Float, the maximum velocity dispersion to use.
    
    model_sig_samples:  Float, the number of values of velocity dispersion to 
                        use when calculating the base corrections over the 
                        entire parameter space.
                        
    output_sig_samples: Float, the number of values of velocity dispersion to 
                        use when calculating the average correction tables.
                        
    output_ages:    List or 1D array, the ages to calculate an average 
                    correction for, summing over all metallicities and some 
                    set of velocity dispersions.
                
    bands:  Nx7 array, containing the passbands of N indices and units, in the 
            format given by Worthey et al., 
            http://astro.wsu.edu/worthey/html/index.table.html . Columns 1-2 
            should be the wavelength bounds of the index, 3-4 the blue 
            pseudocontinuum, and 5-6 the red pseudocontinuum. Column 7 
            dictates the unit used, with 0 corresponding to Angstroms, and 1 
            corresponding to a magnitude. If no definitions are given, this 
            will default to the standard Lick indices, including Balmer lines.
                
    re_run: Boolean, will force a full re-calculation even if the output files 
            already exist.
    
    plot:   Boolean, will produce plots for all indices if True.
    
    
    Attributes
    ----------
    index_array:    4D array, the relative equivalent width of the indices.
    
    age_vals:       1D array, the values of age (assumed to be in Gyr) used 
                    by the templates.
    
    Z_vals:         1D array, the values of log metallicity [Z/H] used by the 
                    templates.
        
    sigma_range:    1D array, the range of velocity dispersions used for the 
                    modelling.
    
    
    Files
    -----    
    vel_disp_corr_{}_gyr:   The correction tables for each set of ages. 
                            Column 1 is the velocity dispersion used; all 
                            other columns are the corrections to the indices. 
                            Corrections are multiplicative for those in units 
                            of Angstroms, and additive for those measured in 
                            magnitudes. Saved in both .npy and .csv formats.
    
    '''
    
    def __init__(self, temp_dir = "templates\MILES_SSP\MILES_BASTI_BI_Ep0.00\\Mbi1.30Z*",
                 out_dir = "templates/vel_disp_corrs",
                 MILES_FWHM = 2.50, sigma_min = 175, sigma_max = 400, model_sig_samples = 91,
                 output_sig_samples = 46, output_ages = [1.5,3.5,8,14],
                 bands = None, re_run = False, plot = False):
        
        self.c = 299792.458
        self.factor = np.sqrt(8.*np.log(2.))
        
        self.temp_dir = temp_dir
        self.out_dir = out_dir
        self.MILES_FWHM = MILES_FWHM
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mod_sig_samples = model_sig_samples
        self.out_sig_samples = output_sig_samples
        self.out_ages = output_ages
        
        ### Note that no exceptions are caught in the main code.
        ### This is intentional, as any errors here must be resolved 
        ### before running other tools, such as SSP_params.
                
        ### Initialise index definitions to be used in the corrections
        if bands is not None:
            self.bands = bands
            
        else:
            try:
                self.bands = Table.read('templates/Lick_Indices.txt', format = 'ascii')
                
            except Exception as e:
                logging.exception("Failed to initialise index definitions.")
                raise e
        
        if not re_run:
            try:
                self.age_vals = np.load(self.out_dir+"/age_range.npy")
                self.Z_vals = np.load(self.out_dir+"/Z_range.npy")
                self.sigma_range = np.load(self.out_dir+"/sigma_range.npy")
                self.index_array = dict()
                
                for n in self.bands["Name"]:
                    self.index_array[n] = \
                        np.load(self.out_dir+"/full_{0}.npy".format(n))
                      
            except Exception as e:
                print ("Base correction files could not be loaded.\n"
                        "Calculating files again.\n")
                logging.exception(e)
                self.calculation()

            finally:
                try:
                    for age in self.out_ages:
                        with open(self.out_dir+"/corr_{0:04.1f}_gyr.pkl".format(age), 
                                  'rb') as outfile:
                            pickle.load(outfile)
                    print ("Correction tables already created for required ages.\n")
                    
                except:
                    print ("Creating correction tables for given parameters.\n")
                    self.binning_corrections()
                    
        if re_run:
            try:
                self.calculation()
                print ("Creating correction tables for given parameters.\n")
                self.binning_corrections()
            except Exception as e:
                print ("Something's gone wrong here.\n")
                raise e
            
        if plot == True:
            self.make_plots()
            
###################################################################################################
    
    def calculation(self):
        
        '''The main routine to calculate dispersion corrections.
        Assuming the templates are similar to the MILES SSPs, the age and 
        metallicity of each template is extracted and stored in an array. 
        Each template is convolved to the ideal Lick/IDS resolution, then for 
        each sigma, convolved with the appropriate Gaussian and compared to 
        the index reference at the IDS resolution. 
        The attributes below are also saved as *.npy files, so that 
        correction tables can be generated for a different range of ages, or 
        plots made of the index variation, without having to re-calculate the 
        underlying measurements.
        
        Attributes
        ----------
        index_array:    4D array. Each template is uniquely defined by its 
                        age and metallicity, the first two indices of the 
                        array. The third index is the velocity dispersion 
                        used for the convolution, and the fourth index the 
                        relative equivalent width of the indices compared to 
                        the reference at the Lick/IDS resolution.
        
        age_vals:       1D array, the values of age (in Gyr) used by the 
                        templates.
        
        Z_vals:         1D array, the values of metallicity Z used by the 
                        templates.
            
        sigma_range:    1D array, the range of velocity dispersions used for 
                        the modelling.
        
        '''
        
        # if isinstance(self.temp_dir, list):
        #     templates = glob.glob(self.temp_dir[0])
        #     for path in self.temp_dir[1:]:
        #         templates += glob.glob(path)
        # elif isinstance(self.temp_dir, str):
        #     templates = glob.glob(self.temp_dir)
        
        template_details = []
        
        for i, temp_path in enumerate(self.temp_dir):
            
            # try:
            #     temp_name = temp_path.split("\\")[-1]
            #     print ("first")
            # except:
            #     temp_name = temp_path.split("/")[-1]
            #     print ("second")
            
            # print (temp_name)
            
            temp_name = temp_path.split("\\")[-1]
            
            Z_str = temp_name[8:13]
            if Z_str[0] == "m":
                Z = - np.float(Z_str[1:])
            else:
                Z = np.float(Z_str[1:])
                
            age = np.float(temp_name[14:21])
            
            template_details.append([temp_path, age, Z])
        
        self.age_vals = np.sort(np.unique([row[1] for row in template_details]))
        self.Z_vals = np.sort(np.unique([row[2] for row in template_details]))
        
        self.sigma_range = np.linspace(self.sigma_min, self.sigma_max, 
                                       num = self.mod_sig_samples)
        
        self.index_array = dict()
        
        for n in self.bands["Name"]:
            self.index_array[n] = np.empty((len(self.age_vals), len(self.Z_vals), 
                                            self.mod_sig_samples)) 
        
        # self.index_array = np.empty((len(self.age_vals), len(self.Z_vals), 
        #                              self.mod_sig_samples, len(self.bands)))
        
        for row in template_details:
            
            print (row)
            
            age_idx = np.searchsorted(self.age_vals, row[1])
            Z_idx = np.searchsorted(self.Z_vals, row[2])
            
            temp_FITS = pf.open(row[0])
            hdr = temp_FITS[0].header
            
            temp_spec = (temp_FITS[0].data).reshape((4300,))
            temp_lam = hdr['CRVAL1'] + hdr['CDELT1'] *(np.arange(hdr['NAXIS1']) 
                                                       + 1 - hdr['CRPIX1'])
        
            SC = Spectrum_Cut(temp_spec, temp_lam, bands = self.bands)
            
            ### This is the template at the Lick/IDS resolution
            conv_obj = Convolutions(
                SC.output["fluxes"],
                SC.output["wavelengths"], 
                0, 0, 0, 
                FWHM_blue = self.MILES_FWHM, 
                FWHM_red = self.MILES_FWHM,
                system = 'IDS', 
                flux_err = SC.output["fluxes_err"],
                )
        
            # if row[1] <= 5 and row[1] >= 4:
            #     ### Measure the equivalent widths of the indices
            #     base_IM = Index_Measure(lams, temp_conv, res_conv, plot = True,
            #                             bands = self.bands, no_error = True)
            # else:
            base_IM = Index_Measure(
                conv_obj.output["fluxes_convolved"],
                SC.output["wavelengths"], 
                conv_obj.output["fluxes_convolved_err"],
                bands = self.bands,
                no_error = True,
                )
            base_indices = base_IM.eq_width
            
            
            for j, sigma in enumerate(self.sigma_range):
                
                vel_sigma = temp_lam * (np.sqrt((sigma + self.c)/
                                               (self.c - sigma)) - 1)/hdr["CDELT1"]
                
                temp_disp = util.gaussian_filter1d(temp_spec, vel_sigma)
                
                disp_SC = Spectrum_Cut(temp_disp, temp_lam, bands = self.bands)
            
                ### This is the template at the Lick/IDS resolution
                # disp_conv_obj = Convolutions(lams, disp_specs, 0, sigma, 0, 
                #                              FWHM_blue = self.MILES_FWHM, 
                #                              FWHM_red = self.MILES_FWHM,
                #                              system = 'IDS', flux_err = res)
                disp_conv_obj = Convolutions(
                    disp_SC.output["fluxes"],
                    disp_SC.output["wavelengths"], 
                    0,
                    sigma, 
                    0, 
                    FWHM_blue = self.MILES_FWHM, 
                    FWHM_red = self.MILES_FWHM,
                    system = 'IDS', 
                    flux_err = disp_SC.output["fluxes_err"],
                    )
                temp_conv_disp, res_conv_disp, disp_flags = disp_conv_obj.output
            
                ### Measure the equivalent widths of the indices
                # if row[1] <= 5 and row[1] >= 4:
                #     disp_IM = Index_Measure(lams, temp_conv_disp, res_conv_disp, plot = True,
                #                             bands = self.bands, no_error = True)
                    
                #     plt.text(plt.xlim()[0], plt.ylim()[1], (row[1], row[2]),
                #         fontsize=16)
                #     plt.show()
                    
                # else:
                # disp_IM = Index_Measure(lams, temp_conv_disp, res_conv_disp, 
                #                         bands = self.bands, no_error = True)
                disp_IM = Index_Measure(
                    disp_conv_obj.output["fluxes_convolved"],
                    disp_SC.output["wavelengths"], 
                    disp_conv_obj.output["fluxes_convolved_err"],
                    bands = self.bands,
                    no_error = True,
                    )
                
                for n, u in zip(self.bands["Name"], self.bands["Units"]):
                    if disp_conv_obj.output["convolution_flags"][n] == 0:
                        self.index_array[n][age_idx, Z_idx, j] = np.nan
                    elif u == 0:
                        val = base_indices[n]/disp_IM.eq_width[n]
                        if val >= 0.25 and val <= 4:
                            self.index_array[n][age_idx, Z_idx, j] = val
                        else:
                            self.index_array[n][age_idx, Z_idx, j] = np.nan
                        # self.index_array[n][age_idx, Z_idx, j] = val
                    elif u == 1:
                        self.index_array[n][age_idx, Z_idx, j] = \
                            base_indices[n] - disp_IM.eq_width[n]
                
                # for i, (new, base, unit) in enumerate( \
                #             zip(disp_IM.eq_width, base_indices, self.bands["Units"])):
                #     if unit == 0:
                #         self.index_array[age_idx,Z_idx,j,i] = base/new
                #     else:
                #         self.index_array[age_idx,Z_idx,j,i] = base - new
                            
            temp_FITS.close()
                                
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(self.out_dir.joinpath("sigma_range.npy"), self.sigma_range)
        np.save(self.out_dir.joinpath("age_range.npy"), self.age_vals)
        np.save(self.out_dir.joinpath("Z_range.npy"), self.Z_vals)
        
        for n in self.bands["Name"]:
            np.save(self.out_dir.joinpath("full_{0}.npy".format(n)), self.index_array[n])
        
###################################################################################################
                        
    def make_plots(self, idx = None):
        
        '''A method to generate plots of the Lick/IDS index variation.
        For best results, no bands will have been passed in the first 
        instance, allowing this to access the names and units of the feature. 
        
        Parameters
        ----------
        idx:    Float, the index of the bands array to plot results for. If 
                no number is passed here, all features in bands will be 
                plotted.
        '''
        
        if idx == None:
            for band in self.bands:
                try:
                    name = band["Name"]
                    unit = band["Units"]
                    self.idx_plot(unit, name)
                except:
                    print ("Either the name or units of this index are unavailable.\n"
                           "Proceeding to the next index.")
                    
        else:      
            try:
                name = idx["Name"]
                unit = idx["Units"]
                self.idx_plot(unit, name)
            except:
                print ("Either the name or units of this index are unavailable.\n"
                       "Check the index is formatted correctly.")
        
###################################################################################################
                
    def idx_plot(self, unit, name):
        
        '''The actual plotting method.
        Given the position of a feature in the bands array, the units it is 
        measured in, and (optionally) a name, 6 graphs will be generated. 
        These show the relative difference between the measured equivalent 
        width, and the reference width at the Lick/IDS resolution, as a 
        function of the model parameters.
        
        Parameters
        ----------
        index:  Float, the index of the bands array to plot results for. 
        
        unit:   Integer, dictating the units used, with 0 corresponding to 
                Angstroms, and 1 corresponding to a magnitude. 
            
        name:   String, the name of the feature being plotted.
        
        '''
        
        if unit == 0:
            vmin = 0.5
            vmax = 1.5
        else:
            vmin = -0.01
            vmax = 0.01
    
        idx_arr_slice = self.index_array[name]
        
        fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (15,10), 
                               constrained_layout = True)
        
        fig.suptitle("{}".format(name))
        
        ax[0,0].plot(self.age_vals, np.squeeze(np.nanmean(idx_arr_slice, axis = (1,2))))
        ax[0,0].set_aspect(1./ax[0,0].get_data_ratio())
        ax[0,0].set_xlabel("Age (Gyr)")
        ax[0,0].set_ylabel("Difference")
        
        ax[0,1].plot(self.Z_vals, np.squeeze(np.nanmean(idx_arr_slice, axis = (0,2))))
        ax[0,1].set_aspect(1./ax[0,1].get_data_ratio())
        ax[0,1].set_xlabel("[Z/H]")
        ax[0,1].set_ylabel("Difference")
        
        ax[0,2].plot(self.sigma_range, np.squeeze(np.nanmean(idx_arr_slice, axis = (0,1))))
        ax[0,2].set_aspect(1./ax[0,2].get_data_ratio())
        ax[0,2].set_xlabel("sigma")
        ax[0,2].set_ylabel("Difference")
        
        age, sig = np.meshgrid(self.age_vals, self.sigma_range, indexing = 'ij')
        age_sigma = np.squeeze(np.nanmean(idx_arr_slice, axis = (1)))
        im = ax[1,0].pcolormesh(sig, age, age_sigma, 
                                cmap = 'jet', vmin = vmin, vmax = vmax)
        ax[1,0].set_aspect(aspect = ((self.sigma_range[0]- self.sigma_range[-1])/
                                               (self.age_vals[0]- self.age_vals[-1])))
        ax[1,0].set_ylabel("Age (Gyr)")
        ax[1,0].set_xlabel("sigma")
        
        age, Z = np.meshgrid(self.age_vals, self.Z_vals, indexing = 'ij')
        age_Z = np.squeeze(np.nanmean(idx_arr_slice, axis = (2)))
        ax[1,1].pcolormesh(Z, age, age_Z, cmap = 'jet', vmin = vmin, vmax = vmax)
        ax[1,1].set_aspect(aspect = ((self.Z_vals[0]- self.Z_vals[-1])/
                                     (self.age_vals[0]- self.age_vals[-1])))
        ax[1,1].set_ylabel("Age (Gyr)")
        ax[1,1].set_xlabel("[Z/H]")
        
        sig, Z = np.meshgrid(self.Z_vals, self.sigma_range, indexing = 'ij')
        Z_sigma = np.squeeze(np.nanmean(idx_arr_slice, axis = (0)))
        ax[1,2].pcolormesh(Z, sig, Z_sigma, cmap = 'jet', vmin = vmin, vmax = vmax)
        ax[1,2].set_aspect(aspect = ((self.sigma_range[0]- self.sigma_range[-1])/
                                     (self.Z_vals[0]- self.Z_vals[-1])))
        ax[1,2].set_ylabel("[Z/H]")
        ax[1,2].set_xlabel("sigma")
        
        fig.colorbar(im, ax = ax.flat)
        
        # plt.savefig("plots/filter_2/{}_all.pdf".format(name))
        plt.savefig("plots/filter_2/{}_all.png".format(name))
        
        return
    
###################################################################################################
        
    def binning_corrections(self):
        
        '''A method to create the age-dependent correction tables.
        The corrections are binned up to and including the age values passed 
        in the class initialisation. The values are averaged over all 
        metallicities and relevant ages, and depending on the rebinning of 
        velocity dispersions, may also be averaged over some set of these.
        
        Files
        -----
        vel_disp_corr_{}_gyr:   Tables, saved as both *.npy and *.csv files.
                                The file name includes the last age included 
                                in the corrections. Column 1 is the velocity 
                                dispersion used; all other columns are the 
                                corrections to the indices. Corrections are 
                                multiplicative for those in units of 
                                Angstroms, and additive for those measured in 
                                magnitudes. 
        
        '''
        
        out_sig = np.linspace(self.sigma_range[0], self.sigma_range[-1], 
                                       num = self.out_sig_samples)
        
        for j, age in enumerate(self.out_ages):
            
            if self.out_ages[j-1] >= age:
                age_inds = np.where((self.age_vals <= age))[0]
            else:
                age_inds = np.where((self.age_vals <= age) & 
                                    (self.age_vals > self.out_ages[j-1]))[0]
                    
            age_corr_dict = dict()
            age_corr_dict["sigma"] = out_sig
            
            for n, u in zip(self.bands["Name"], self.bands["Units"]):
                
                age_corr_dict[n+"_units"] = u
                age_corr_dict[n] = np.empty_like(out_sig)
                
                for i, sig in enumerate(out_sig):
                    
                    if out_sig[i-1] >= sig:
                        sig_inds = np.where((self.sigma_range <= sig))[0]
                    else:
                        sig_inds = np.where((self.sigma_range <= sig) & 
                                            (self.sigma_range > out_sig[i-1]))[0]
                        
                    cut = self.index_array[n][age_inds[:,np.newaxis], :,
                                              np.expand_dims(sig_inds, axis = 0)]
                    
                    avg = bn.nanmean(cut)
                    
                    age_corr_dict[n][i] = avg
            
            with open(self.out_dir.joinpath("corr_{0:04.1f}_gyr.pkl".format(age)), 
                      'wb') as outfile:
                pickle.dump(age_corr_dict, outfile)
        
        return
        
    
###################################################################################################

def get_files(folder, pattern):
    
    all_files = [] 
    
    for ext in pattern: 
        all_files.extend(folder.glob(ext))
        
    string_out = [os.fspath(a) for a in all_files]
    
    return string_out


def poly_fn(coef, x):
    
    out = 0
    
    for n, c in enumerate(coef):
        out += c * x ** n
    
    return out



def corr_fit(age_list, subset_dir, out_dir, bands=None):
    
    if bands==None:
        bands = Table.read('templates/Lick_Indices.txt', format = 'ascii')
    
    for age in age_list:
        
        # with open("templates/vel_disp_corrs/Z_subset__Ep0.00/"+
        #           "corr_{0:04.1f}_gyr.pkl".format(age), 
        #           'rb') as outfile:
        #     corr_tab_alpha_0 = pickle.load(outfile)
        # print (out_dir)
        with open(subset_dir.joinpath(
                  "corr_{0:04.1f}_gyr.pkl".format(age)), 
                  'rb') as outfile:
            corr_tab_alpha_4 = pickle.load(outfile)
            
        new_corr = copy.deepcopy(corr_tab_alpha_4)
        
        for n in bands["Name"]:
            
            # data = np.nanmean([corr_tab_alpha_4[n], corr_tab_alpha_0[n]], axis = 0)
              
            data = corr_tab_alpha_4[n]
            
            print (data)
            
            y = np.ma.masked_array(data, mask = np.isnan(data))
            
            # print (y)
            
            x = np.ma.masked_array(corr_tab_alpha_4["sigma"], mask = y.mask, fill_value = np.nan)
            
            # print (x)
            
            x_vals = x[~x.mask]
            y_vals = y[~y.mask]
            
            fit = np.polynomial.polynomial.polyfit(x_vals, y_vals, 3)
            
            corr = poly_fn(fit, x)
            
            new_corr[n] = corr.filled()
        
        with open(out_dir.joinpath("corr_{0:04.1f}_gyr.pkl".format(age)), 
                  'wb') as outfile:
            pickle.dump(new_corr, outfile)

def plotting_fn():
    
    bands = Table.read('templates/Lick_Indices.txt', format = 'ascii')
    
    for n in bands["Name"]:
        
        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12,10), 
                                constrained_layout = True)
        
        fig.suptitle("{}".format(n), fontsize = 16)
    
        for age, pos in zip([1.5,3.5,8,14], [(0,0),(0,1),(1,0),(1,1)]):
            
            with open("templates/vel_disp_corrs/Z_subset__Ep0.00/"+
                      "corr_{0:04.1f}_gyr.pkl".format(age), 
                      'rb') as outfile:
                corr_tab_alpha_0 = pickle.load(outfile)
                 
            ax[pos].scatter(corr_tab_alpha_0["sigma"], corr_tab_alpha_0[n])
            
            with open("templates/vel_disp_corrs/Z_subset__Ep0.40/"+
                      "corr_{0:04.1f}_gyr.pkl".format(age), 
                      'rb') as outfile:
                corr_tab_alpha_4 = pickle.load(outfile)
                 
            ax[pos].scatter(corr_tab_alpha_4["sigma"], corr_tab_alpha_4[n])
            
            with open("templates/vel_disp_corrs/"+
                      "corr_{0:04.1f}_gyr.pkl".format(age), 
                      'rb') as outfile:
                corr_tab_avg = pickle.load(outfile)
                
            ax[pos].plot(corr_tab_avg["sigma"], corr_tab_avg[n])
            
            ax[pos].set_title(str(age)+" Gyr", fontsize=14)
        
        # plt.savefig("plots/limit_5_alpha_0/{}_corr_tabs.pdf".format(n))
            
# plotting_fn() 

            