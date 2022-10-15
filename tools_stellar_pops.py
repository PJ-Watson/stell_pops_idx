# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:11:38 2020

@author: peter
"""

###################################################################################################

import numpy as np
# from numpy import ma
import bottleneck as bn

from scipy.interpolate import griddata #, InterpolatedUnivariateSpline, Rbf
# from scipy.stats import chi2 as scipy_chi2

# from astropy.table import Table

# import numpy.lib.recfunctions as rf

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numba
import pathlib

import glob, traceback, logging, json, pickle, copy, time

###################################################################################################


class SSP_Params():
    
    '''Class to calculate the most likely SSP parameters based on a set of 
    Lick index measurements, using chi-squared minimisation.
    
    Parameters
    ----------
    tmj_mods:   An array containing the models used, in the format given by
                Thomas, Maraston and Johansson (2011).
    INSERT OTHER MODELS
    lick_idx:   An array containing both the index measurement and the
                associated error.
    hi_res:     A boolean, corresponding to the resolution used to interpolate
                the models.
    sigma:      Only passed if the indices require correction, due to a larger
                dispersion than required.
    
    Output
    ------
    
    '''

    def __init__(
            self, 
            equiv_widths, 
            width_errs, 
            temp_dir,
            log_file_path,
            hi_res = False, 
            sigma = None, 
            sig_flags = None,
            tmj_mods = None, 
            tmj_errs = None,
            # sch_mods = None, 
            tmj_names = None, 
            # sch_names = None,
            ):
        
        self.eq_widths = copy.deepcopy(equiv_widths)
        self.width_errs = width_errs
        self.err_log_file = log_file_path
        
        self.temp_dir = temp_dir
        
        with self.err_log_file.open("a") as f:
            f.write("-" * 62 + "\n"+"Beginning SSP fit."+"\n")
        
        ### Sanity check for inputs
        assert self.eq_widths.keys() == self.width_errs.keys(), \
            "The widths and the associated errors do not match.\n"
            
        ### Checking the flags, and defaulting to 0 if none are given
        if sig_flags is not None:
            self.sig_flags = sig_flags
            assert self.eq_widths.keys() == self.sig_flags.keys(), \
                "The widths and dispersion correction flags do not match.\n"
        else:
            self.sig_flags = dict.fromkeys(self.eq_widths, 0)
        
        self.tmj_mods = tmj_mods
        self.tmj_errs = tmj_errs
        # self.sch_mods = sch_mods
        
        if self.tmj_mods is None:# and self.sch_mods is None:
            err_msg_init = "Requires at least one model as input."
            raise Exception(err_msg_init)
            
        self.tmj_names = tmj_names
        # self.sch_names = sch_names
            
        # if self.sch_mods is not None:
        #     # try:
        #     #     with open(name_conversion) as infile:
        #     #         self.LI_to_sch = json.load(infile)
                    
        #     # except Exception:
        #     #     err_conv_dict_sch = ("Could not load file to translate index "+
        #     #                          "names from Schiavon templates.\n")
        #     #     raise Exception(err_conv_dict_sch)
            
        #     if self.sch_names is None:
        #         self.sch_names = ['Hdelta_F', 'Hdelta_A', 'Hgamma_F', 
        #                           'Hgamma_A', 'Fe4383','H_beta','Fe5015',
        #                           'Mg_b','Mg_2','Fe5270','Fe5335']
            
            
        if self.tmj_mods is not None:
            # try:
            #     with open(name_conversion) as infile:
            #         self.LI_to_tmj = json.load(infile)
                    
            # except Exception:
            #     err_conv_dict_tmj = ("Could not load file to translate index "+
            #                          "names from TMJ templates.\n")
            #     raise Exception(err_conv_dict_tmj)
                
                
        ### Do everything up to calculating chi^2 using individual indices
        ### Store the extrapolated/interpolated models per index
        ### Check for existence of models before running all these checks
                
                
            if self.tmj_names is None:
                self.tmj_names = ["Hdelta_A", "Hdelta_F", "CN_1", "CN_2",
                                  "Ca4227", "G4300", "Hgamma_A", "Hgamma_F",
                                  "Fe4383", "Ca4455", "Fe4531", "Fe4668",
                                  "H_beta", "Fe5015", "Mg_1", "Mg_2", 
                                  "Mg_b", "Fe5270", "Fe5335", "Fe5406",
                                   "Fe5709", "Fe5782", 
                                   #"Na_D", 
                                   "TiO_1", "TiO_2"]
            
        self.hi_res = hi_res
        self.sigma = sigma
        
        ### Begin running methods
        
            
        # if self.sch_mods is not None:
        #     self.sch_interp()
        #     if any(self.sig_flags.values()) == 1:
        #         print ("Beginning correction")
        #         self.sch_disp_correction()
        #         print ("Ending correction")
        #     else:
        #         self.sch_measure()
            
        if self.tmj_mods is not None:
            
            self.start_time = time.time()
            
            with self.err_log_file.open("a") as f:
                f.write("Beginning methods: {}".format(
                        time.time() - self.start_time
                )+"\n")
            
            self.tmj_interp()
            
            with self.err_log_file.open("a") as f:
                f.write("Loaded models: {}".format(
                    time.time() - self.start_time
                )+"\n")
            
            if any(self.sig_flags.values()) == 1:
                self.tmj_disp_correction()
                
                with self.err_log_file.open("a") as f:
                    f.write("Finished analysis: {}".format(
                        time.time() - self.start_time
                    )+"\n")
            else:
                self.tmj_measure()
        
        # self.sch_disp_correction()
        # self.tmj_disp_correction()
        
        # if self.flag:
        #     print ("")
        #     print ("Needs correction")
        #     print ("")
        #     self.sch_disp_correction()
        #     print ("")
        #     print ("Schiavon corrected")
        #     print ("")
        #     self.tmj_disp_correction()
        #     print ("")
        #     print ("TMJ corrected")
        #     print ("")
        # else:
        #     self.sch_measure()
        #     self.tmj_measure()
            
        self.ssp_errors()
            
###################################################################################################
    
    def tmj_interp(self):
        
        '''Interpolates the TMJ models to a given resolution. If hi_res is True, then the parameter
        spacing is 0.01 in [alpha/Fe], and 0.02 in log age and metallicity.
        
        Attributes
        ----------
        tmj_free_param (float):         The number of free parameters in the model.
        
        tmj_age_interp (1d array):      The log interpolated values for the age of the stellar 
                                        population.
                                    
        tmj_Z_interp (1d array):        The linearly interpolated values for the metallicity, Z.
        
        tmj_alpha_interp (1d array):    The linearly interpolated values for [alpha/Fe].
        
        tmj_mod_interp (4d array):      The first 3 axes are the parameter values used, and the 4th
                                        axis the corresponding index measurements from the model.
        
        '''
        
        tmj_age = np.unique(self.tmj_mods["age"])
        tmj_Z = np.unique(self.tmj_mods["[Z/H]"])
        tmj_alpha = np.unique(self.tmj_mods["[alpha/Fe]"])
        
        self.tmj_free_param = len(self.tmj_names) - 3
        
        self.mod_arr = None
        
        if self.hi_res == True:
            
            self.tmj_age_interp = np.geomspace(bn.nanmin(tmj_age),
                                              bn.nanmax(tmj_age), 220)
            self.tmj_Z_interp = np.linspace(bn.nanmin(tmj_Z),
                                              bn.nanmax(tmj_Z), 141)
            self.tmj_alpha_interp = np.linspace(bn.nanmin(tmj_alpha),
                                              bn.nanmax(tmj_alpha), 81)
            
            self.tmj_mod_interp = dict()
            
            for name in self.tmj_names:
                try:
                    self.tmj_mod_interp[name] = np.load(
                        self.temp_dir.joinpath(
                            "tmj_interpolated_hi_res/{}.npy".format(name)))
                    
                except:
                    self.tmj_mod_interp[name] = self.tmj_create_interp(
                        name, 
                        self.tmj_mods, 
                        "tmj_interpolated_hi_res",
                    )
                
            if self.tmj_errs is not None:
                self.tmj_err_interp = dict()
                for name in self.tmj_names:
                    try:
                        self.tmj_err_interp[name] = np.load(
                            self.temp_dir.joinpath(
                                "tmj_err_interpolated_hi_res/{}.npy".format(name)))
                        
                    except:
                        self.tmj_err_interp[name] = self.tmj_create_interp(
                            name, 
                            self.tmj_errs, 
                            "tmj_err_interpolated_hi_res",
                        )
                
        else:
            
            self.tmj_age_interp = np.geomspace(bn.nanmin(tmj_age),
                                              bn.nanmax(tmj_age), 12)
            self.tmj_Z_interp = np.linspace(bn.nanmin(tmj_Z),
                                              bn.nanmax(tmj_Z), 15)
            self.tmj_alpha_interp = np.linspace(bn.nanmin(tmj_alpha),
                                              bn.nanmax(tmj_alpha), 9)
            
            self.tmj_mod_interp = dict()
            
            for name in self.tmj_names:
                try:
                    self.tmj_mod_interp[name] = np.load(
                        self.temp_dir.joinpath(
                            "tmj_interpolated_lo_res/{}.npy".format(name)))
                    
                except:
                    self.tmj_mod_interp[name] = self.tmj_create_interp(
                        name, 
                        self.tmj_mods, 
                        "tmj_interpolated_lo_res",
                    )
                
            if self.tmj_errs is not None:
                self.tmj_err_interp = dict()
                for name in self.tmj_names:
                    try:
                        self.tmj_err_interp[name] = np.load(
                            self.temp_dir.joinpath(
                                "tmj_err_interpolated_lo_res/{}.npy".format(name)))
                        
                    except:
                        self.tmj_err_interp[name] = self.tmj_create_interp(
                            name, 
                            self.tmj_errs, 
                            "tmj_err_interpolated_lo_res",
                        )
        
        return
    
###################################################################################################
   
    def tmj_create_interp(self, name, data, out_path):
        
        '''Generates the interpolated TMJ models. 
        Saves the output to a file for future use.
        
        Attributes
        ----------
        tmj_mod_interp:     4D array. The first 3 axes are the parameter values used, and the 4th
                            axis the corresponding index measurements from the model.
        '''
        
        if np.nanmin(data["[alpha/Fe]"])==np.nanmax(data["[alpha/Fe]"]):
            
            X, Y = np.meshgrid(self.tmj_age_interp, self.tmj_Z_interp,
                               indexing="ij")
            
            interp = griddata(
                (data["age"],data["[Z/H]"]),
                data[name], 
                (X,Y),
                method = 'linear',
            )
            
            interp = np.repeat(
                interp[:, :, np.newaxis], 
                len(self.tmj_alpha_interp), 
                axis=2,
            )
            
            del X,Y
            
        else:
        
            X, Y, Z = np.meshgrid(self.tmj_age_interp, self.tmj_Z_interp, 
                                  self.tmj_alpha_interp, indexing='ij')
        
            interp = griddata(
                (data["age"],data["[Z/H]"],data["[alpha/Fe]"]),
                data[name], 
                (X,Y,Z),
                method = 'linear',
            )
            
            del X,Y,Z
        
        folder = self.temp_dir.joinpath(out_path)
        folder.mkdir(parents=True, exist_ok=True)
        
        # if self.hi_res == True:
        #     folder = self.temp_dir.joinpath("tmj_interpolated_hi_res")
        #     folder.mkdir(parents=True, exist_ok=True)
        # else:            
        #     folder = self.temp_dir.joinpath("tmj_interpolated_lo_res")
        #     folder.mkdir(parents=True, exist_ok=True)
            
        np.save(folder.joinpath("{}.npy".format(name)), interp)
        
        return interp
    
# ###################################################################################################
    
#     def sch_interp(self):

#         '''Interpolates the Schiavon models to a given resolution. 
#         If hi_res is True, then the parameter spacing is 0.01 in [alpha/Fe], and 0.02 in log age 
#         and metallicity. If no existing template file is found, a new one will be generated and 
#         saved for future use.
        
#         Attributes
#         ----------
#         sch_free_param (float):         The number of free parameters in the model.
        
#         sch_age_interp (1d array):      The log interpolated values for the age of the stellar 
#                                         population.
                                    
#         sch_Z_interp (1d array):        The linearly interpolated values for the metallicity, Z.
        
#         sch_alpha_interp (1d array):    The linearly interpolated values for [alpha/Fe].
        
#         sch_mod_interp (4d array):      The first 3 axes are the parameter values used, and the 4th
#                                         axis the corresponding index measurements from the model.
        
#         '''
        
#         sch_age = np.unique(self.sch_mods["Age"])
#         sch_Z = np.unique(self.sch_mods["[Fe/H]"])
#         sch_alpha = np.unique(self.sch_mods["[alpha/Fe]"])
        
#         self.sch_free_param = len(self.sch_names) - 3
        
#         if self.hi_res == True:
            
#             self.sch_age_interp = np.geomspace(bn.nanmin(sch_age),
#                                               bn.nanmax(sch_age), 220)
#             self.sch_Z_interp = np.linspace(bn.nanmin(sch_Z),
#                                               bn.nanmax(sch_Z), 90)
#             self.sch_alpha_interp = np.linspace(bn.nanmin(sch_alpha),
#                                               bn.nanmax(sch_alpha), 43)
            
#             self.sch_mod_interp = dict()
            
#             for name in self.sch_names:
#                 try:
#                     self.sch_mod_interp[name] = np.load(
#                         self.temp_dir.joinpath(
#                             "sch_interpolated_hi_res/{}.npy".format(name)))
                    
#                 except:
                    
#                     self.sch_create_interp(name)
                
#         else:
            
#             self.sch_age_interp = np.geomspace(bn.nanmin(sch_age),
#                                               bn.nanmax(sch_age), 12)
#             self.sch_Z_interp = np.linspace(bn.nanmin(sch_Z),
#                                               bn.nanmax(sch_Z), 15)
#             self.sch_alpha_interp = np.linspace(bn.nanmin(sch_alpha),
#                                               bn.nanmax(sch_alpha), 9)
            
#             self.sch_mod_interp = dict()
            
#             for name in self.sch_names:
#                 try:
#                     self.sch_mod_interp[name] = np.load(
#                         self.temp_dir.joinpath(
#                             "sch_interpolated_lo_res/{}.npy".format(name)))
                    
#                 except:
                    
#                     self.sch_create_interp(name)
        
#         return
    
# ###################################################################################################
   
#     def sch_create_interp(self, name):
        
#         '''Generates the interpolated sch models. 
#         Saves the output to a file for future use.
        
#         Attributes
#         ----------
#         sch_mod_interp:     4D array. The first 3 axes are the parameter values used, and the 4th
#                             axis the corresponding index measurements from the model.
#         '''
        
#         X, Y, Z = np.meshgrid(self.sch_age_interp, self.sch_Z_interp, 
#                               self.sch_alpha_interp, indexing='ij')
        
        
#         interp = griddata((self.sch_mods["Age"],self.sch_mods["[Fe/H]"],
#                                               self.sch_mods["[alpha/Fe]"]), 
#                                              self.sch_mods[name], (X,Y,Z),
#                                              method = 'linear')
        
#         self.sch_mod_interp[name] = interp
        
#         if self.hi_res == True:
#             folder = self.temp_dir.joinpath("sch_interpolated_hi_res")
#             folder.mkdir(parents=True, exist_ok=True)
#         else:            
#             folder = self.temp_dir.joinpath("sch_interpolated_lo_res")
#             folder.mkdir(parents=True, exist_ok=True)
            
#         np.save(folder.joinpath("{}.npy".format(name)), interp)
        
#         del X,Y,Z, interp
#         return
    
###################################################################################################
    
#     def sch_interp(self):
        
#         '''Interpolates the Schiavon models to a given resolution. 
#         If hi_res is True, then the parameter spacing is 0.01 in [alpha/Fe], and 0.02 in log age 
#         and metallicity. If no existing template file is found, a new one will be generated and 
#         saved for future use.
        
#         Attributes
#         ----------
#         sch_free_param (float):         The number of free parameters in the model.
        
#         sch_age_interp (1d array):      The log interpolated values for the age of the stellar 
#                                         population.
                                    
#         sch_Z_interp (1d array):        The linearly interpolated values for the metallicity, Z.
        
#         sch_alpha_interp (1d array):    The linearly interpolated values for [alpha/Fe].
        
#         sch_mod_interp (4d array):      The first 3 axes are the parameter values used, and the 4th
#                                         axis the corresponding index measurements from the model.
        
#         '''
        
#         sch_age = np.unique(self.sch_mods["Age"])
#         sch_Z = np.unique(self.sch_mods["[Fe/H]"])
        
#         self.sch_free_param = len(self.sch_names) - 3
        
#         if self.hi_res == True:
            
#             self.sch_age_interp = np.geomspace(bn.nanmin(sch_age),
#                                               bn.nanmax(sch_age), 220)
#             self.sch_Z_interp = np.linspace(bn.nanmin(sch_Z),
#                                               bn.nanmax(sch_Z), 90)
#             self.sch_alpha_interp = np.linspace(-0.3, 0.5, 81)
            
#             self.sch_mod_interp = dict()
            
#             for name in self.sch_names:
#                 try:
#                     self.sch_mod_interp[name] = \
#                         np.load("templates/sch_interpolated_hi_res/{}.npy".format(name))
                    
#                 except:
                    
#                     self.sch_create_interp(name)
                
#         else:
            
#             self.sch_age_interp = np.linspace(bn.nanmin(sch_age),
#                                               bn.nanmax(sch_age), 12)
#             self.sch_Z_interp = np.linspace(bn.nanmin(sch_Z),
#                                               bn.nanmax(sch_Z), 9)
#             self.sch_alpha_interp = np.linspace(-0.3, 0.5, 9)
            
#             self.sch_mod_interp = dict()
            
#             for name in self.sch_names:
#                 try:
#                     self.sch_mod_interp[name] = \
#                         np.load("templates/sch_interpolated_lo_res/{}.npy".format(name))
                    
#                 except:
                    
#                     self.sch_create_interp(name)

#         return
    
# ###################################################################################################
    
#     def sch_create_interp(self, name):
        
#         '''Generates the interpolated Schiavon models. 
#         The models are first interpolated to a regular grid within the confines of the model. The 
#         [alpha/Fe] axis is then extrapolated to the range [-0.3, 0.5].
#         Saves the output to a file for future use.
        
#         Attributes
#         ----------
#         sch_mod_interp:     4D array. The first 3 axes are the parameter values used, and the 4th
#                             axis the corresponding index measurements from the model.
#         '''
        
#         sch_alpha = np.unique(self.sch_mods["[alpha/Fe]"])
        
#         if self.hi_res == True:
#             old_alpha_interp = np.linspace(bn.nanmin(sch_alpha),
#                                               bn.nanmax(sch_alpha), 43)
#         else:
#             old_alpha_interp = np.linspace(bn.nanmin(sch_alpha),
#                                        bn.nanmax(sch_alpha), 5)
        
#         X, Y, Z = np.meshgrid(self.sch_age_interp, self.sch_Z_interp, 
#                               old_alpha_interp, indexing='ij')
        
#         interp = griddata((self.sch_mods["Age"],self.sch_mods["[Fe/H]"],
#                                               self.sch_mods["[alpha/Fe]"]), 
#                                              self.sch_mods[self.LI_to_sch[name]], (X,Y,Z),
#                                              method = 'linear')
        
#         new_array = np.full((len(self.sch_age_interp), len(self.sch_Z_interp), 
#                              len(self.sch_alpha_interp)), np.nan)
        
#         for i,age in enumerate(self.sch_age_interp):
#             for j,Z in enumerate(self.sch_Z_interp):
                
#                 alpha = interp[i,j,:]
                
#                 if bn.anynan(alpha) == False:
                    
#                     spl = InterpolatedUnivariateSpline(old_alpha_interp, alpha, k=1)
#                     alpha_extrap = spl(self.sch_alpha_interp)
#                     new_array[i,j,:] = alpha_extrap
                    
#                     continue
                    
#                 if bn.allnan(alpha) == True:
                    
#                     continue
                    
#                 idx_check = np.where(np.isfinite(alpha) == True)[0]
                    
#                 if len(idx_check)>=2:
                    
#                     alpha_x_cut = old_alpha_interp[idx_check]
                    
#                     alpha_vals = alpha[idx_check]
                    
#                     spl = InterpolatedUnivariateSpline(alpha_x_cut, alpha_vals, k = 1)
                    
#                     if idx_check[0] == 0:
#                         lower_lim = -0.3
#                     else:
#                         lower_lim = round(old_alpha_interp[idx_check[0]],2)
#                     if idx_check[-1] == len(old_alpha_interp)-1:
#                         upper_lim = 0.5
#                     else:
#                         upper_lim = round(old_alpha_interp[idx_check[-1]],2)
                    
#                     lo_idx = np.searchsorted(self.sch_alpha_interp, lower_lim)
#                     hi_idx = np.searchsorted(self.sch_alpha_interp, upper_lim, 
#                                              side = 'right')
                    
#                     alpha_extrap = spl(self.sch_alpha_interp[lo_idx:hi_idx])
                    
#                     new_array[i,j,lo_idx:hi_idx] = alpha_extrap
                    
#                     continue
                
#                 else:
                    
#                     val = old_alpha_interp[idx_check]
                    
#                     idx = np.searchsorted(self.sch_alpha_interp, val)
                    
#                     new_array[i,j,idx] = alpha[idx_check]
                    
#         self.sch_mod_interp[name] = new_array
    
#         if self.hi_res == True:
#             if not os.path.exists("templates/sch_interpolated_hi_res/"):
#                 os.mkdir("templates/sch_interpolated_hi_res/")
#             np.save("templates/sch_interpolated_hi_res/{}.npy".format(name), new_array)
#         else:            
#             if not os.path.exists("templates/sch_interpolated_lo_res/"):
#                 os.mkdir("templates/sch_interpolated_lo_res/")
#             np.save("templates/sch_interpolated_lo_res/{}.npy".format(name), new_array)
                
#         del X,Y,Z, interp
#         return
    
# ###################################################################################################

#     def sch_disp_correction(self):
        
#         '''Dispersion correction for the Schiavon models.
        
#         '''
        
#         print ("Beginning correction")
        
#         self.sch_copy = copy.deepcopy(self.eq_widths)
        
#         try:
#             templates = glob.glob("templates\\vel_disp_corrs\\corr_????_gyr.pkl")
#             templates.sort()
            
#             ages = [float(t.split("\\")[-1][5:9]) for t in templates]
            
#             if len(templates) == 0:
                
#                 templates = glob.glob("templates/vel_disp_corrs/corr_????_gyr.pkl")
#                 templates.sort()
                
#                 ages = [float(t.split("/")[-1][5:9]) for t in templates]
                
#         except Exception:
#             err_corr_files = ("Could not load dispersion correction files.")
#             raise Exception(err_corr_files)
        
        
#         # print (templates)
        
#         for p, a in zip(templates, ages):
            
#             end = self.sch_corr_sub(p, a)
            
#             if end:
#                 break
            
            
#         self.eq_widths, self.sch_copy = self.sch_copy, self.eq_widths
        
        
#         return
        
        
#     def sch_corr_sub(self, tab_path, age):
        
#         with open(tab_path, 'rb') as outfile:
#             corr_tab = pickle.load(outfile)
        
#         sig_idx = np.searchsorted(corr_tab["sigma"], self.sigma)
        
        
#         for n in self.sch_names:
#             if not np.isfinite(corr_tab[n][sig_idx]):
#                 self.eq_widths[n] = self.sch_copy[n]
#             elif corr_tab[n+"_units"] == 1:
#                 self.eq_widths[n] = self.sch_copy[n] + corr_tab[n][sig_idx]
#             else:
#                 self.eq_widths[n] = self.sch_copy[n] * corr_tab[n][sig_idx]
                
#         self.sch_measure(iterate = False)
#         # self.chi_plot()
        
#         if self.ssp_age_sch <= age:
#             print ("Successfully converged to a solution, with age below "
#                    "{} Gyr.\n".format(age))
#             return True
#         else:
#             print ("No solution found yet.\n")
#             return False 
    
###################################################################################################

    def tmj_disp_correction(self):
        
        '''Dispersion correction for the TMJ models.
        
        THIS NEEDS A SERIOUS REWRITE.
        
        To do:
            
        - This function needs to be rewritten. Should not be linear progression, but looped with
          relevant function calls.
        - Write an equivalent set for TMJ - or combine the two into one set of corrections.
        - Should be able to run automatically, and choose if indices need correction for dispersion
        - Output should be as tuple, 
        
        IMPORTANT -- USE SIGMA_FLAG FROM INITIAL TABLE TO DECIDE IF THIS METHOD IS RUN!
            
            
        '''
        
        '''Dispersion correction for the Schiavon models.
        
        '''
        
        self.tmj_copy = copy.deepcopy(self.eq_widths)
        self.tmj_errs_copy = copy.deepcopy(self.width_errs)
        
        
        with self.err_log_file.open("a") as f:
            f.write("Copied data: {}".format(time.time() - self.start_time)+"\n")
        
        try:
            template_gen = self.temp_dir.joinpath(
                "vel_disp_corrs").expanduser().glob("*.pkl")
            templates = [t for t in template_gen]
            templates.sort()
            
            ages = [float(t.parts[-1][5:9]) for t in templates]
            
            # if len(templates) == 0:
                
            #     templates = glob.glob("templates/vel_disp_corrs/corr_????_gyr.pkl")
            #     templates.sort()
                
            #     ages = [float(t.split("/")[-1][5:9]) for t in templates]
                
        except Exception:
            err_corr_files = ("Could not load dispersion correction files.")
            raise Exception(err_corr_files)
            
        with self.err_log_file.open("a") as f:
            f.write("Loaded correction_templates: {}".format(
                time.time() - self.start_time
            )+"\n")
        
        self.tmj_check()
        
        with self.err_log_file.open("a") as f:
            f.write("Checked indices: {}".format(
                time.time() - self.start_time
            )+"\n")
        
        
        for p, a in zip(templates, ages):
            
            end = self.tmj_corr_sub(p, a)
            
            if end:
                break
        
        # print (self.eq_widths)
        # self.corr_widths = copy.deepcopy(self.eq_widths)
        # print (self.corr_widths)
        # print (self.tmj_copy)
        # self.eq_widths = copy.deepcopy(self.tmj_copy)
        # print (self.eq_widths)
        # print (self.corr_widths)
            
        self.eq_widths, self.tmj_copy = self.tmj_copy, self.eq_widths
        self.width_errs, self.tmj_errs_copy = self.tmj_errs_copy, self.width_errs
        
        self.corr_widths = self.tmj_copy
        self.corr_width_errs = self.tmj_errs_copy
        
            
        return
        
        
    def tmj_corr_sub(self, tab_path, age):
        
        with open(tab_path, 'rb') as outfile:
            corr_tab = pickle.load(outfile)
            
        with self.err_log_file.open("a") as f:
            f.write("Loaded correction table: {}".format(
                time.time() - self.start_time
            )+"\n")
        
        sig_idx = np.searchsorted(corr_tab["sigma"], self.sigma)
        
        
        for n in self.tmj_names:
            sig_idx = np.clip(
                sig_idx, 
                a_min=0, 
                a_max=len(corr_tab[n])-1,
            )
            if not np.isfinite(corr_tab[n][sig_idx]):
                self.eq_widths[n] = self.tmj_copy[n]
            elif corr_tab[n+"_units"] == 1:
                self.eq_widths[n] = self.tmj_copy[n] + corr_tab[n][sig_idx]
                self.width_errs[n] = self.tmj_errs_copy[n] + corr_tab[n][sig_idx]
            else:
                self.eq_widths[n] = self.tmj_copy[n] * corr_tab[n][sig_idx]
                self.width_errs[n] = self.tmj_errs_copy[n] * corr_tab[n][sig_idx]
                
        with self.err_log_file.open("a") as f:
            f.write("Applied corrections to indices: {}".format(
                time.time() - self.start_time
            )+"\n")
        
        self.tmj_measure_sub()
        
        if self.ssp_age_tmj <= age:
            msg = (
                "Successfully converged to a solution, with age below "
                "{} Gyr.\n".format(age)+"Current time: {}".format(
                    time.time() - self.start_time
                )+"\n"
            )
            with self.err_log_file.open("a") as f:
                f.write(msg)
                print (msg)
            return True
        else:
            msg = (
                "No solution found yet for age < {} Gyr.\n".format(age)
                +"Current time: {}".format(
                    time.time() - self.start_time
                )+"\n"
            )
            with self.err_log_file.open("a") as f:
                f.write(msg)
                print (msg)
        
            return False
    
###################################################################################################
        
    def tmj_measure_sub(self):
        
        '''A method to calculate the chi^2 array for all possible parameter values, given 
        tmj_idx_check exists, using the models from Thomas, Maraston and Johansson (2011). Finds 
        the minimum chi^2 value and the corresponding SSP parameters.
        
        Attributes
        ----------
        tmj_chi (3d array):         The reduced chi^2 values corresponding to the fit of the model 
                                    to the observations.
        
        tmj_chi_idx (1d array):     The index of the minimum value in the chi^2 array.
        
        ssp_age_tmj (float):        The most likely value for the age of the stellar population.
                                    
        ssp_Z_tmj (float):          The most likely value for the metallicity, Z.
        
        ssp_alpha_tmj (float):      The most likely value for [alpha/Fe].
        
        '''
        
        names = [n for n in self.tmj_names if self.tmj_idx_check[n]]
        
        # self.tmj_chi = np.empty_like(self.tmj_mod_interp[names[0]])
        
        self.tmj_free_param = len(names) - 3
        
        # for ijk in np.ndindex(self.tmj_chi.shape):
        #     self.tmj_chi[ijk] = (bn.nansum([((self.eq_widths[n] - 
        #                                       self.tmj_mod_interp[n][ijk])/
        #                                      self.width_errs[n])**2 for n in names])/
        #                          self.tmj_free_param)
            
        
        eq_arr = np.asarray([self.eq_widths[n] for n in names], dtype=float).T
        err_arr = np.asarray([self.width_errs[n] for n in names], dtype=float).T
        
        with self.err_log_file.open("a") as f:
            f.write("Generated index arrays: {}".format(
                time.time() - self.start_time
            )+"\n")

        # mod_arr = np.array([self.tmj_mod_interp[n] for n in names]).T
        # # mod_err_arr = np.array([self.tmj_err_interp[n] for n in names]).T
        if self.mod_arr is None:
            self.mod_arr = np.asarray([self.tmj_mod_interp[n] for n in names], dtype=float).T
        
        with self.err_log_file.open("a") as f:
            f.write("Generated models array: {}".format(
                time.time() - self.start_time
            )+"\n")
    
        self.tmj_chi = (bn.nansum(
            (
                (eq_arr - self.mod_arr)/err_arr#np.sqrt(err_arr**2+mod_err_arr**2)
            )**2,
            axis=-1,
        )/self.tmj_free_param).T
        
        with self.err_log_file.open("a") as f:
            f.write("Created chi^2 array: {}".format(
                time.time() - self.start_time
            )+"\n")
            
        self.tmj_chi_idx = np.unravel_index(bn.nanargmin(self.tmj_chi, 
                                axis=None), self.tmj_chi.shape)
        
        with self.err_log_file.open("a") as f:
            f.write("Found minimum: {}".format(time.time() - self.start_time)+"\n")
        
        self.tmj_chi_min = self.tmj_chi[self.tmj_chi_idx]
        
        self.ssp_age_tmj = self.tmj_age_interp[self.tmj_chi_idx[0]]
        self.ssp_Z_tmj = self.tmj_Z_interp[self.tmj_chi_idx[1]]        
        self.ssp_alpha_tmj = self.tmj_alpha_interp[self.tmj_chi_idx[2]]
        
        return
    
###################################################################################################
    
    def tmj_check(self):
        
        self.tmj_idx_check = dict()
        for name in self.tmj_names:
            tmj_max = bn.nanmax(self.tmj_mods[name],axis=0)
            tmj_min = np.nanmin(self.tmj_mods[name],axis=0)
            
            if self.eq_widths[name] - np.abs(self.width_errs[name]) > tmj_max:
                self.tmj_idx_check[name] = False
            elif self.eq_widths[name] + np.abs(self.width_errs[name]) < tmj_min:
                self.tmj_idx_check[name] = False
            elif not np.isfinite(self.eq_widths[name]):
                self.tmj_idx_check[name] = False
            else:
                self.tmj_idx_check[name] = True
    
###################################################################################################
    
    def tmj_measure(self, iterate = False):
        
        '''A method to calculate the most likely SSP parameters from observed lick indices, using 
        the models from Thomas, Maraston and Johansson (2011). This part calculates which indices 
        are more than one sigma from the edge of the model predicted index space, and flags these.
        
        Parameters
        ----------
        iterate (bool, optional):       If iterate is True, then an iterative search will be 
                                        performed until the result is statistically significant. 
                                        Defaults to False.
                                    
        Attributes
        ----------
        tmj_idx_check (1d bool array):  The array values show whether the index should be included
                                        in the fit.
        
        '''
        
        self.tmj_check()
            
        ### Must have at least 5 indices in the model range
        if np.sum([val for val in self.tmj_idx_check.values()]) < 5:
            self.ssp_age_tmj = np.nan
            self.ssp_Z_tmj = np.nan            
            self.ssp_alpha_tmj = np.nan
            
        ### Must have at least one Balmer line
        elif (("Hdelta_A" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Hdelta_A"] == False) and 
              ("Hdelta_F" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Hdelta_F"] == False) and
              ("Hgamma_A" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Hgamma_A"] == False) and
              ("Hgamma_F" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Hgamma_F"] == False) and
              ("Hbeta" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Hbeta"] == False)):
            
            self.ssp_age_tmj = np.nan
            self.ssp_Z_tmj = np.nan            
            self.ssp_alpha_tmj = np.nan
            
        ### Must have at least one Fe index
        elif (("Fe4383" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Fe4383"] == False) and 
              ("Fe4531" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Fe4531"] == False) and
              ("Fe4668" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Fe4668"] == False) and
              ("Fe5015" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Fe5015"] == False) and
              ("Fe5270" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Fe5270"] == False) and
              ("Fe5335" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Fe5335"] == False) and
              ("Fe5406" in self.tmj_idx_check == False or 
               self.tmj_idx_check["Fe5406"] == False)):
            
            self.ssp_age_tmj = np.nan
            self.ssp_Z_tmj = np.nan            
            self.ssp_alpha_tmj = np.nan
            
        else:
            self.tmj_measure_sub()
                
        # if iterate:
            
        # # Following code is an implementation of an iterative search for 
        # # the minimum, rejecting the greatest outlier at the minimum if
        # # the result is not significant enough
            
        #     print ('')
        #     print ('Minimum Chi^2')
        #     print (self.tmj_chi[self.tmj_chi_idx])
        #     print ('')
            
        #     current_min = scipy_chi2.sf(self.tmj_chi[self.tmj_chi_idx], 
        #                                 self.tmj_free_param)
            
        #     while current_min <= 0.95:
            
        #         chi_min = ma.array([((d-m)/e)**2 for d,e,m in zip(self.tmj_lick_dat, 
        #                                               self.tmj_lick_err,
        #                                               self.tmj_mod_interp[self.tmj_chi_idx])],
        #                             mask = np.invert(self.tmj_idx_check))
                
        #         to_remove = bn.nanmax(chi_min[~chi_min.mask])
        #         for i, chi in enumerate(chi_min):
        #             if chi == to_remove:
        #                 self.tmj_idx_check[i] = False
        
        #         self.tmj_measure_sub()
                
        #         print ('')
        #         print ('Minimum Chi^2')
        #         print (self.tmj_chi[self.tmj_chi_idx])
        #         print ('')
                
        #         current_min = scipy_chi2.sf(self.tmj_chi[self.tmj_chi_idx], 
        #                                     self.tmj_free_param)
                
        #         print ('')
        #         print ('Probability')
        #         print (current_min)
        #         print ('')
                
        return
    
# ###################################################################################################
    
#     def sch_chi_sum(self, ijk, names):
        
#         '''A method to calculate the chi^2 sum for a given set of observed indices, model 
#         predictions, and errors. Will first check to see if the model predictions exist (since the 
#         Schiavon models do not have uniform coverage in the parameter space), returning NaN if not.
        
#         Parameters
#         ----------
#         obs (1d array):     The observed index measurements.
        
#         err (1d array):     The error on the observed indices.
            
#         mod (1d array):     The predicted index measurements from the model.
            
        
#         Output
#         ------
#         chi^2:          Will return either NaN or the sum of chi^2 contributions from each index.
        
#         '''
        
#         if np.any(self.sch_mod_interp[names[0]][ijk] == np.nan):
#             return np.nan
#         else:
#             return (bn.nansum([((self.eq_widths[n] - self.sch_mod_interp[n][ijk])/
#                                 self.width_errs[n])**2 for n in names])/self.sch_free_param)
    
# ###################################################################################################
    
#     def sch_measure_sub(self):
        
#         '''A method to calculate the chi^2 array for all possible parameter values, given 
#         sch_idx_check exists, using the models from Schiavon (2007). Finds the minimum chi^2 value 
#         and the corresponding SSP parameters.
        
#         Attributes
#         ----------
#         sch_chi (3d array):         The reduced chi^2 values corresponding to the fit of the model 
#                                     to the observations.
        
#         sch_chi_idx (1d array):     The index of the minimum value in the chi^2 array.
        
#         ssp_age_sch (float):        The most likely value for the age of the stellar population.
                                    
#         ssp_Z_sch (float):          The most likely value for the metallicity, Z.
        
#         ssp_alpha_sch (float):      The most likely value for [alpha/Fe].
        
#         '''
        
        
#         names = [n for n in self.sch_names if self.sch_idx_check[n]]
        
#         self.sch_chi = np.empty_like(self.sch_mod_interp[names[0]])
        
#         for ijk in np.ndindex(self.sch_chi.shape):
#             self.sch_chi[ijk] = self.sch_chi_sum(ijk, names)
        
#         self.sch_chi = bn.replace(self.sch_chi, 0, np.nan)
        
#         self.sch_chi_idx = np.unravel_index(bn.nanargmin(self.sch_chi, 
#                                 axis=None), self.sch_chi.shape)
        
#         self.sch_chi_min = self.sch_chi[self.sch_chi_idx]
        
#         self.ssp_age_sch = self.sch_age_interp[self.sch_chi_idx[0]]
#         self.ssp_Z_sch = self.sch_Z_interp[self.sch_chi_idx[1]]        
#         self.ssp_alpha_sch = self.sch_alpha_interp[self.sch_chi_idx[2]]
        
#         return
    
# ###################################################################################################
    
#     def sch_measure(self, iterate = False):
        
#         '''A method to calculate the most likely SSP parameters from observed lick indices, using 
#         the models from Schiavon (2007). This part calculates which indices are more than one sigma
#         from the edge of the model predicted index space, and flags these.
        
#         Parameters
#         ----------
#         iterate (bool, optional):       If iterate is True, then an iterative search will be 
#                                         performed until the result is statistically significant. 
#                                         Defaults to False.
                                    
#         Attributes
#         ----------
#         sch_idx_check (1d bool array):  The array values show whether the index should be included 
#                                         in the fit.
        
#         '''
        
#         self.sch_idx_check = dict()
        
#         for name in self.sch_names:
#             sch_max = bn.nanmax(self.sch_mods[name],axis=0)
#             sch_min = np.nanmin(self.sch_mods[name],axis=0)
            
#             if self.eq_widths[name] - np.abs(self.width_errs[name]) > sch_max:
#                 self.sch_idx_check[name] = False
#             elif self.eq_widths[name] + np.abs(self.width_errs[name]) < sch_min:
#                 self.sch_idx_check[name] = False
#             else:
#                 self.sch_idx_check[name] = True
            
#         ### Must have at least 5 indices in the model range
#         if np.sum([val for val in self.sch_idx_check.values()]) < 5:
#             self.ssp_age_sch = np.nan
#             self.ssp_Z_sch = np.nan            
#             self.ssp_alpha_sch = np.nan
            
#         ### Must have at least one Balmer line
#         elif (("Hdelta_A" in self.sch_idx_check == False or 
#                self.sch_idx_check["Hdelta_A"] == False) and 
#               ("Hdelta_F" in self.sch_idx_check == False or 
#                self.sch_idx_check["Hdelta_F"] == False) and
#               ("Hgamma_A" in self.sch_idx_check == False or 
#                self.sch_idx_check["Hgamma_A"] == False) and
#               ("Hgamma_F" in self.sch_idx_check == False or 
#                self.sch_idx_check["Hgamma_F"] == False) and
#               ("Hbeta" in self.sch_idx_check == False or 
#                self.sch_idx_check["Hbeta"] == False)):
            
#             self.ssp_age_sch = np.nan
#             self.ssp_Z_sch = np.nan            
#             self.ssp_alpha_sch = np.nan
            
#         ### Must have at least one Fe index
#         elif (("Fe4383" in self.sch_idx_check == False or 
#                self.sch_idx_check["Fe4383"] == False) and 
#               ("Fe4531" in self.sch_idx_check == False or 
#                self.sch_idx_check["Fe4531"] == False) and
#               ("Fe4668" in self.sch_idx_check == False or 
#                self.sch_idx_check["Fe4668"] == False) and
#               ("Fe5015" in self.sch_idx_check == False or 
#                self.sch_idx_check["Fe5015"] == False) and
#               ("Fe5270" in self.sch_idx_check == False or 
#                self.sch_idx_check["Fe5270"] == False) and
#               ("Fe5335" in self.sch_idx_check == False or 
#                self.sch_idx_check["Fe5335"] == False) and
#               ("Fe5406" in self.sch_idx_check == False or 
#                self.sch_idx_check["Fe5406"] == False)):
            
#             self.ssp_age_sch = np.nan
#             self.ssp_Z_sch = np.nan            
#             self.ssp_alpha_sch = np.nan
            
#         else:
#             self.sch_measure_sub()
        
            
#         # if iterate:
            
#         # # Following code is an implementation of an iterative search for 
#         # # the minimum, rejecting the greatest outlier at the minimum if
#         # # the result is not significant enough
        
#         #     print ('')
#         #     print ('Minimum Chi^2')
#         #     print (self.sch_chi[self.sch_chi_idx])
#         #     print ('')
            
#         #     current_min = scipy_chi2.sf(self.sch_chi[self.sch_chi_idx], 
#         #                                 self.sch_free_param)
            
#         #     while current_min <= 0.95:
            
#         #         chi_min = ma.array([((d-m)/e)**2 for d,e,m in zip(self.sch_lick_dat, 
#         #                                               self.sch_lick_err,
#         #                                               self.sch_mod_interp[self.sch_chi_idx])],
#         #                             mask = np.invert(self.sch_idx_check))
                
#         #         to_remove = bn.nanmax(chi_min[~chi_min.mask])
#         #         for i, chi in enumerate(chi_min):
#         #             if chi == to_remove:
#         #                 self.sch_idx_check[i] = False
        
#         #         self.sch_measure_sub()
                
#         #         print ('')
#         #         print ('Minimum Chi^2')
#         #         print (self.sch_chi[self.sch_chi_idx])
#         #         print ('')
                
#         #         current_min = scipy_chi2.sf(self.sch_chi[self.sch_chi_idx], 
#         #                                     self.sch_free_param)
                
#         #         print ('')
#         #         print ('Probability')
#         #         print (current_min)
#         #         print ('')
        
            
#         return
    
###################################################################################################
                                        
    def chi_plot(self):
        '''A method to plot the output from the chi^2 minimisation. Returns 6 plots, one for each 
        2 axis representation of a slice through the minimum, for both models.
        '''
        
        if hasattr(self, 'tmj_chi'):
            
            # cols = ["0.6", "k", "red"]
            # cols = ["red", "k", "0.6"]
            # cmap1 = colors.LinearSegmentedColormap.from_list("mycmap", cols)
            
            plt.figure(figsize=(15,10))
            Z, alpha = np.meshgrid(self.tmj_Z_interp, self.tmj_alpha_interp, indexing='ij')
            plt.pcolormesh(
                Z, 
                alpha, 
                self.tmj_chi[self.tmj_chi_idx[0],:,:],
                norm=colors.LogNorm(vmin=1, vmax=1000), 
                cmap=plt.cm.jet,
                # cmap=cmap1
            )
            
            colorbar = plt.colorbar()
            colorbar.set_label(r"$\chi^2_{\nu}$", 
                               fontsize=14, rotation=0, y=1.05, labelpad=-40)
            
            plt.contour(Z, alpha, self.tmj_chi[self.tmj_chi_idx[0], :, :], 
                        levels=[self.tmj_chi[self.tmj_chi_idx] + 1,
                                self.tmj_chi[self.tmj_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, origin = 'lower')
            
            plt.scatter(self.tmj_Z_interp[self.tmj_chi_idx[1]],
                        self.tmj_alpha_interp[self.tmj_chi_idx[2]], color='r')
            plt.xlabel(r'$[Z/H]$', fontsize=14)
            plt.ylabel(r'$[\alpha/\rm{Fe}]$', fontsize=14)
            plt.show()
            
            plt.figure(figsize=(15,10))
            age, Z = np.meshgrid(self.tmj_age_interp, self.tmj_Z_interp, indexing='ij')
            plt.pcolormesh(age, Z, self.tmj_chi[:, :, self.tmj_chi_idx[2]],
                           norm=colors.LogNorm(vmin=1, vmax=1000), 
                           cmap=plt.cm.jet)
            
            colorbar = plt.colorbar()
            colorbar.set_label(r"$\chi^2_{\nu}$", 
                               fontsize=14, rotation=0, y=1.05, labelpad=-40)
            
            plt.contour(age, Z, self.tmj_chi[:, :, self.tmj_chi_idx[2]], 
                        levels=[self.tmj_chi[self.tmj_chi_idx] + 1,
                                self.tmj_chi[self.tmj_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, 
                        origin = 'lower')
    
    
            plt.scatter(self.tmj_age_interp[self.tmj_chi_idx[0]],
                        self.tmj_Z_interp[self.tmj_chi_idx[1]],color='r')
            plt.xlabel(r'$\rm{Age\ (Gyr)}$', fontsize=14)
            plt.ylabel(r'$\rm{[Z/H]}$', fontsize=14)
            plt.show()
            
            plt.figure(figsize=(15,10))
            alpha, age = np.meshgrid(self.tmj_alpha_interp, self.tmj_age_interp)
            plt.pcolormesh(alpha, age, self.tmj_chi[:, self.tmj_chi_idx[1], :],
                           norm=colors.LogNorm(vmin=1, vmax=1000),
                           cmap=plt.cm.jet)
            
            colorbar = plt.colorbar()
            colorbar.set_label(r"$\chi^2_{\nu}$", 
                               fontsize=14, rotation=0, y=1.05, labelpad=-40)
            
            plt.contour(alpha, age, self.tmj_chi[:, self.tmj_chi_idx[1], :], 
                        levels=[self.tmj_chi[self.tmj_chi_idx] + 1,
                                self.tmj_chi[self.tmj_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, origin = 'lower')
    
            plt.scatter(self.tmj_alpha_interp[self.tmj_chi_idx[2]],
                        self.tmj_age_interp[self.tmj_chi_idx[0]], color='r')
            plt.ylabel(r'$\rm{Age\ (Gyr)}$', fontsize=14)
            plt.xlabel(r'$[\alpha/\rm{Fe}]$', fontsize=14)
            plt.show()
        
        if hasattr(self, 'sch_chi'):
                                
            plt.figure(figsize=(20,10))
            plt.imshow(self.sch_chi[self.sch_chi_idx[0], :, :], 
                           norm=colors.LogNorm(vmin=1, vmax=1000),
                           cmap=plt.cm.jet, 
                       extent = [self.sch_alpha_interp[0], 
                                 self.sch_alpha_interp[-1],
                                 self.sch_Z_interp[0], self.sch_Z_interp[-1]], 
                       origin = 'lower', aspect = 1/3.65)
            plt.colorbar()
            
            plt.contour(self.sch_chi[self.sch_chi_idx[0], :, :], 
                        levels=[self.sch_chi[self.sch_chi_idx] + 1,
                                self.sch_chi[self.sch_chi_idx] + 3.5], 
                        colors='white', alpha=1.0,
                        extent = [self.sch_alpha_interp[0], 
                                  self.sch_alpha_interp[-1], self.sch_Z_interp[0], 
                                  self.sch_Z_interp[-1]], origin = 'lower')
            
            plt.scatter(self.sch_alpha_interp[self.sch_chi_idx[2]],
                        self.sch_Z_interp[self.sch_chi_idx[1]], color='r')
            plt.xlabel(r'$[\alpha/\rm{Fe}]$', fontsize=14)
            plt.ylabel(r'$Z$', fontsize=14)
            plt.show()
    
            plt.figure(figsize=(15,10))
        
            age, Z = np.meshgrid(self.sch_age_interp, self.sch_Z_interp, indexing='ij')
            plt.pcolormesh(age,Z, self.sch_chi[:, :, self.sch_chi_idx[2]], 
                           norm=colors.LogNorm(vmin=1, vmax=1000),
                           cmap=plt.cm.jet)
            plt.colorbar()
            
            plt.contour(age, Z, self.sch_chi[:, :, self.sch_chi_idx[2]], 
                        levels=[self.sch_chi[self.sch_chi_idx] + 1,
                                self.sch_chi[self.sch_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, 
                        origin = 'lower')
    
    
            plt.scatter(self.sch_age_interp[self.sch_chi_idx[0]],
                        self.sch_Z_interp[self.sch_chi_idx[1]],color='r')
            plt.xlabel(r'$\rm{Age\ (Gyr)}$', fontsize=14)
            plt.ylabel(r'$\rm{Z}$', fontsize=14)
            plt.show()
            
            plt.figure(figsize=(15,10))
            alpha, age = np.meshgrid(self.sch_alpha_interp, self.sch_age_interp)
            plt.pcolormesh(alpha,age, self.sch_chi[:, self.sch_chi_idx[1], :],
                           norm=colors.LogNorm(vmin=1, vmax=1000),
                           cmap=plt.cm.jet)
            
            plt.colorbar()
            
            plt.contour(alpha, age, self.sch_chi[:, self.sch_chi_idx[1], :], 
                        levels=[self.sch_chi[self.sch_chi_idx] + 1,
                                self.sch_chi[self.sch_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, origin = 'lower')
    
            plt.scatter(self.sch_alpha_interp[self.sch_chi_idx[2]],
                        self.sch_age_interp[self.sch_chi_idx[0]], color='r')
            plt.ylabel(r'$\rm{Age\ (Gyr)}$', fontsize=14)
            plt.xlabel(r'$[\alpha/\rm{Fe}]$', fontsize=14)
            plt.show()        
    
###################################################################################################
            
    def ssp_errors(self):
        
        '''A method to estimate the errors on the SSP population parameters from the 1-sigma 
        confidence intervals (CI). Will log errors in the unlikely case that this CI
        is less than the grid size used. Depending on which models were used, some of these 
        attributes may not be available.

        Attributes
        ----------
        ssp_age_tmj_bounds (1d array):      The upper and lower bounds of the 1-sigma CI on 
                                            age using the TMJ models.
                                            
        ssp_Z_tmj_bounds (1d array):        The upper and lower bounds of the 1-sigma CI on Z using
                                            the TMJ models.
                                            
        ssp_alpha_tmj_bounds (1d array):    The upper and lower bounds of the 1-sigma CI on 
                                            [alpha/Fe] using the TMJ models.
                                            
        ssp_age_sch_bounds (1d array):      The upper and lower bounds of the 1-sigma CI on 
                                            age using the Schiavon models.
                                            
        ssp_Z_sch_bounds (1d array):        The upper and lower bounds of the 1-sigma CI on Z using
                                            the Schiavon models.
                                            
        ssp_alpha_sch_bounds (1d array):    The upper and lower bounds of the 1-sigma CI on 
                                            [alpha/Fe] using the Schiavon models.                
        '''
        
        if hasattr(self, 'tmj_chi'):
            
            err_msg_tmj = ("\nError in TMJ SSP error estimation. \n"
                           "Assume results unreliable. \n\n")
            
            tmj_age_slice = np.nanmin(self.tmj_chi, axis = (1,2))
            tmj_age_region = self.tmj_age_interp[np.where(tmj_age_slice <=
                                                          (self.tmj_chi[self.tmj_chi_idx] + 1))]
            try:
                self.ssp_age_tmj_bounds = [tmj_age_region[0], tmj_age_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_tmj)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_tmj)
                
                self.ssp_age_tmj_bounds = [np.nan, np.nan]
                
            tmj_Z_slice = np.nanmin(self.tmj_chi, axis = (0,2))
            tmj_Z_region = self.tmj_Z_interp[np.where(tmj_Z_slice <= 
                                                      (self.tmj_chi[self.tmj_chi_idx] + 1))]
            try:
                self.ssp_Z_tmj_bounds = [tmj_Z_region[0], tmj_Z_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_tmj)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_tmj)
                
                self.ssp_z_tmj_bounds = [np.nan, np.nan]
            
            tmj_alpha_slice = np.nanmin(self.tmj_chi, axis = (0,1))
            tmj_alpha_region = self.tmj_alpha_interp[np.where(tmj_alpha_slice <= 
                                                          (self.tmj_chi[self.tmj_chi_idx] + 1))]
            try:
                self.ssp_alpha_tmj_bounds = [tmj_alpha_region[0], tmj_alpha_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_tmj)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_tmj)
                
                self.ssp_alpha_tmj_bounds = [np.nan, np.nan]
            
        if hasattr(self, 'sch_chi'):
            
            err_msg_sch = ("\nError in Schiavon SSP error estimation. \n"
                           "Assume results unreliable. \n\n")
            
            sch_age_slice = np.nanmin(self.sch_chi, axis = (1,2))
            sch_age_region = self.sch_age_interp[np.where(sch_age_slice <=
                                                          (self.sch_chi[self.sch_chi_idx] + 1))]
            try:
                self.ssp_age_sch_bounds = [sch_age_region[0], sch_age_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_sch)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_sch)
                
                self.ssp_age_sch_bounds = [np.nan, np.nan]
                
            sch_Z_slice = np.nanmin(self.sch_chi, axis = (0,2))
            sch_Z_region = self.sch_Z_interp[np.where(sch_Z_slice <= 
                                                      (self.sch_chi[self.sch_chi_idx] + 1))]
            try:
                self.ssp_Z_sch_bounds = [sch_Z_region[0], sch_Z_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_tmj)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_sch)
                
                self.ssp_z_sch_bounds = [np.nan, np.nan]
            
            sch_alpha_slice = np.nanmin(self.sch_chi, axis = (0,1))
            sch_alpha_region = self.sch_alpha_interp[np.where(sch_alpha_slice <= 
                                                          (self.sch_chi[self.sch_chi_idx] + 1))]
            try:
                self.ssp_alpha_sch_bounds = [sch_alpha_region[0], sch_alpha_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_tmj)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_sch)
                
                self.ssp_alpha_sch_bounds = [np.nan, np.nan]
                
    
###################################################################################################
               
                
@numba.njit(nogil=True)
def _any_nans(a):
    for x in a:
        if np.isnan(x): return True
    return False


##################################################################################################


@numba.jit
def any_nans(a):
    if not a.dtype.kind=='f': return False
    return _any_nans(a.flat)


###################################################################################################
