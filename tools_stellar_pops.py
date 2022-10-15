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
    reg_mods:   An array containing the models used, in the format given by
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
            reg_mods = None, 
            reg_errs = None,
            irreg_mods = None, 
            reg_names = None, 
            irreg_names = None,
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
        
        self.reg_mods = reg_mods
        self.reg_errs = reg_errs
        self.irreg_mods = irreg_mods
        
        if self.reg_mods is None:# and self.irreg_mods is None:
            err_msg_init = "Requires at least one model as input."
            raise Exception(err_msg_init)
            
        self.reg_names = reg_names
        self.irreg_names = irreg_names
            
        if self.irreg_mods is not None:
            # try:
            #     with open(name_conversion) as infile:
            #         self.LI_to_sch = json.load(infile)
                    
            # except Exception:
            #     err_conv_dict_sch = ("Could not load file to translate index "+
            #                          "names from Schiavon templates.\n")
            #     raise Exception(err_conv_dict_sch)
            
            if self.irreg_names is None:
                self.irreg_names = ['Hdelta_F', 'Hdelta_A', 'Hgamma_F', 
                                  'Hgamma_A', 'Fe4383','H_beta','Fe5015',
                                  'Mg_b','Mg_2','Fe5270','Fe5335',
                                   "CN_1", "CN_2",
                                                     "Ca4227", "G4300",
                                                     "Fe4668"
                                  ]
            
            
        if self.reg_mods is not None:
            # try:
            #     with open(name_conversion) as infile:
            #         self.LI_to_reg = json.load(infile)
                    
            # except Exception:
            #     err_conv_dict_reg = ("Could not load file to translate index "+
            #                          "names from reg templates.\n")
            #     raise Exception(err_conv_dict_reg)
                
                
        ### Do everything up to calculating chi^2 using individual indices
        ### Store the extrapolated/interpolated models per index
        ### Check for existence of models before running all these checks
                
                
            if self.reg_names is None:
                self.reg_names = ["Hdelta_A", "Hdelta_F", "CN_1", "CN_2",
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
        
        
        self.start_time = time.time()
            
        if self.irreg_mods is not None:
            
            with self.err_log_file.open("a") as f:
                f.write("Beginning irregular SSP methods: {}".format(
                        time.time() - self.start_time
                )+"\n")
                
            self.irreg_interp()
        #     if any(self.sig_flags.values()) == 1:
        #         print ("Beginning correction")
        #         self.irreg_disp_correction()
        #         print ("Ending correction")
        #     else:
            self.irreg_measure()
            
        if self.reg_mods is not None:
            
            
            with self.err_log_file.open("a") as f:
                f.write("Beginning regular SSP methods: {}".format(
                        time.time() - self.start_time
                )+"\n")
            
            self.reg_interp()
            
            with self.err_log_file.open("a") as f:
                f.write("Loaded models: {}".format(
                    time.time() - self.start_time
                )+"\n")
            
            if any(self.sig_flags.values()) == 1:
                self.reg_disp_correction()
                
                with self.err_log_file.open("a") as f:
                    f.write("Finished analysis: {}".format(
                        time.time() - self.start_time
                    )+"\n")
            else:
                self.reg_measure()
        
        # self.irreg_disp_correction()
        # self.reg_disp_correction()
        
        # if self.flag:
        #     print ("")
        #     print ("Needs correction")
        #     print ("")
        #     self.irreg_disp_correction()
        #     print ("")
        #     print ("Schiavon corrected")
        #     print ("")
        #     self.reg_disp_correction()
        #     print ("")
        #     print ("reg corrected")
        #     print ("")
        # else:
        #     self.irreg_measure()
        #     self.reg_measure()
            
        self.ssp_errors()
            
###################################################################################################
    
    def reg_interp(self):
        
        '''Interpolates the reg models to a given resolution. If hi_res is True, then the parameter
        spacing is 0.01 in [alpha/Fe], and 0.02 in log age and metallicity.
        
        Attributes
        ----------
        reg_free_param (float):         The number of free parameters in the model.
        
        reg_age_interp (1d array):      The log interpolated values for the age of the stellar 
                                        population.
                                    
        reg_Z_interp (1d array):        The linearly interpolated values for the metallicity, Z.
        
        reg_alpha_interp (1d array):    The linearly interpolated values for [alpha/Fe].
        
        reg_mod_interp (4d array):      The first 3 axes are the parameter values used, and the 4th
                                        axis the corresponding index measurements from the model.
        
        '''
        
        reg_age = np.unique(self.reg_mods["age"])
        reg_Z = np.unique(self.reg_mods["[Z/H]"])
        reg_alpha = np.unique(self.reg_mods["[alpha/Fe]"])
        
        self.reg_free_param = len(self.reg_names) - 3
        
        self.reg_mod_arr = None
        
        if self.hi_res == True:
            
            self.reg_age_interp = np.geomspace(bn.nanmin(reg_age),
                                              bn.nanmax(reg_age), 220)
            self.reg_Z_interp = np.linspace(bn.nanmin(reg_Z),
                                              bn.nanmax(reg_Z), 141)
            self.reg_alpha_interp = np.linspace(bn.nanmin(reg_alpha),
                                              bn.nanmax(reg_alpha), 81)
            
            self.reg_mod_interp = dict()
            
            for name in self.reg_names:
                try:
                    self.reg_mod_interp[name] = np.load(
                        self.temp_dir.joinpath(
                            "reg_interpolated_hi_res/{}.npy".format(name)))
                    
                except:
                    self.reg_mod_interp[name] = self.reg_create_interp(
                        name, 
                        self.reg_mods, 
                        "reg_interpolated_hi_res",
                    )
                
            if self.reg_errs is not None:
                self.reg_err_interp = dict()
                for name in self.reg_names:
                    try:
                        self.reg_err_interp[name] = np.load(
                            self.temp_dir.joinpath(
                                "reg_err_interpolated_hi_res/{}.npy".format(name)))
                        
                    except:
                        self.reg_err_interp[name] = self.reg_create_interp(
                            name, 
                            self.reg_errs, 
                            "reg_err_interpolated_hi_res",
                        )
                
        else:
            
            self.reg_age_interp = np.geomspace(bn.nanmin(reg_age),
                                              bn.nanmax(reg_age), 12)
            self.reg_Z_interp = np.linspace(bn.nanmin(reg_Z),
                                              bn.nanmax(reg_Z), 15)
            self.reg_alpha_interp = np.linspace(bn.nanmin(reg_alpha),
                                              bn.nanmax(reg_alpha), 9)
            
            self.reg_mod_interp = dict()
            
            for name in self.reg_names:
                try:
                    self.reg_mod_interp[name] = np.load(
                        self.temp_dir.joinpath(
                            "reg_interpolated_lo_res/{}.npy".format(name)))
                    
                except:
                    self.reg_mod_interp[name] = self.reg_create_interp(
                        name, 
                        self.reg_mods, 
                        "reg_interpolated_lo_res",
                    )
                
            if self.reg_errs is not None:
                self.reg_err_interp = dict()
                for name in self.reg_names:
                    try:
                        self.reg_err_interp[name] = np.load(
                            self.temp_dir.joinpath(
                                "reg_err_interpolated_lo_res/{}.npy".format(name)))
                        
                    except:
                        self.reg_err_interp[name] = self.reg_create_interp(
                            name, 
                            self.reg_errs, 
                            "reg_err_interpolated_lo_res",
                        )
        
        return
    
###################################################################################################
   
    def reg_create_interp(self, name, data, out_path):
        
        '''Generates the interpolated reg models. 
        Saves the output to a file for future use.
        
        Attributes
        ----------
        reg_mod_interp:     4D array. The first 3 axes are the parameter values used, and the 4th
                            axis the corresponding index measurements from the model.
        '''
        
        if np.nanmin(data["[alpha/Fe]"])==np.nanmax(data["[alpha/Fe]"]):
            
            X, Y = np.meshgrid(self.reg_age_interp, self.reg_Z_interp,
                               indexing="ij")
            
            interp = griddata(
                (data["age"],data["[Z/H]"]),
                data[name], 
                (X,Y),
                method = 'linear',
            )
            
            interp = np.repeat(
                interp[:, :, np.newaxis], 
                len(self.reg_alpha_interp), 
                axis=2,
            )
            
            del X,Y
            
        else:
        
            X, Y, Z = np.meshgrid(self.reg_age_interp, self.reg_Z_interp, 
                                  self.reg_alpha_interp, indexing='ij')
        
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
        #     folder = self.temp_dir.joinpath("reg_interpolated_hi_res")
        #     folder.mkdir(parents=True, exist_ok=True)
        # else:            
        #     folder = self.temp_dir.joinpath("reg_interpolated_lo_res")
        #     folder.mkdir(parents=True, exist_ok=True)
            
        np.save(folder.joinpath("{}.npy".format(name)), interp)
        
        return interp
    
###################################################################################################
    
    def irreg_interp(self):

        '''Interpolates the Schiavon models to a given resolution. 
        If hi_res is True, then the parameter spacing is 0.01 in [alpha/Fe], and 0.02 in log age 
        and metallicity. If no existing template file is found, a new one will be generated and 
        saved for future use.
        
        Attributes
        ----------
        irreg_free_param (float):         The number of free parameters in the model.
        
        irreg_age_interp (1d array):      The log interpolated values for the age of the stellar 
                                        population.
                                    
        irreg_Z_interp (1d array):        The linearly interpolated values for the metallicity, Z.
        
        irreg_alpha_interp (1d array):    The linearly interpolated values for [alpha/Fe].
        
        irreg_mod_interp (4d array):      The first 3 axes are the parameter values used, and the 4th
                                        axis the corresponding index measurements from the model.
        
        '''
        
        irreg_age = np.unique(self.irreg_mods["age"])
        irreg_Z = np.unique(self.irreg_mods["[Z/H]"])
        irreg_alpha = np.unique(self.irreg_mods["[alpha/Fe]"])
        
        self.irreg_free_param = len(self.irreg_names) - 3
        
        self.irreg_mod_arr = None
        
        if self.hi_res == True:
            
            self.irreg_age_interp = np.geomspace(bn.nanmin(irreg_age),
                                              bn.nanmax(irreg_age), 220)
            self.irreg_Z_interp = np.linspace(bn.nanmin(irreg_Z),
                                              bn.nanmax(irreg_Z), 90)
            self.irreg_alpha_interp = np.linspace(bn.nanmin(irreg_alpha),
                                              bn.nanmax(irreg_alpha), 43)
            
            self.irreg_mod_interp = dict()
            
            for name in self.irreg_names:
                try:
                    self.irreg_mod_interp[name] = np.load(
                        self.temp_dir.joinpath(
                            "irreg_interpolated_hi_res/{}.npy".format(name)))
                    
                except:
                    
                    self.irreg_create_interp(name)
                
        else:
            
            self.irreg_age_interp = np.geomspace(bn.nanmin(irreg_age),
                                              bn.nanmax(irreg_age), 12)
            self.irreg_Z_interp = np.linspace(bn.nanmin(irreg_Z),
                                              bn.nanmax(irreg_Z), 15)
            self.irreg_alpha_interp = np.linspace(bn.nanmin(irreg_alpha),
                                              bn.nanmax(irreg_alpha), 9)
            
            self.irreg_mod_interp = dict()
            
            for name in self.irreg_names:
                try:
                    self.irreg_mod_interp[name] = np.load(
                        self.temp_dir.joinpath(
                            "irreg_interpolated_lo_res/{}.npy".format(name)))
                    
                except:
                    
                    self.irreg_create_interp(name)
        
        return
    
###################################################################################################
   
    def irreg_create_interp(self, name):
        
        '''Generates the interpolated sch models. 
        Saves the output to a file for future use.
        
        Attributes
        ----------
        irreg_mod_interp:     4D array. The first 3 axes are the parameter values used, and the 4th
                            axis the corresponding index measurements from the model.
        '''
        
        X, Y, Z = np.meshgrid(self.irreg_age_interp, self.irreg_Z_interp, 
                              self.irreg_alpha_interp, indexing='ij')
        
        
        interp = griddata((self.irreg_mods["age"],self.irreg_mods["[Z/H]"],
                                              self.irreg_mods["[alpha/Fe]"]), 
                                              self.irreg_mods[name], (X,Y,Z),
                                              method = 'linear')
        
        self.irreg_mod_interp[name] = interp
        
        if self.hi_res == True:
            folder = self.temp_dir.joinpath("irreg_interpolated_hi_res")
            folder.mkdir(parents=True, exist_ok=True)
        else:            
            folder = self.temp_dir.joinpath("irreg_interpolated_lo_res")
            folder.mkdir(parents=True, exist_ok=True)
            
        np.save(folder.joinpath("{}.npy".format(name)), interp)
        
        del X,Y,Z, interp
        return
    
###################################################################################################
    
#     def irreg_interp(self):
        
#         '''Interpolates the Schiavon models to a given resolution. 
#         If hi_res is True, then the parameter spacing is 0.01 in [alpha/Fe], and 0.02 in log age 
#         and metallicity. If no existing template file is found, a new one will be generated and 
#         saved for future use.
        
#         Attributes
#         ----------
#         irreg_free_param (float):         The number of free parameters in the model.
        
#         irreg_age_interp (1d array):      The log interpolated values for the age of the stellar 
#                                         population.
                                    
#         irreg_Z_interp (1d array):        The linearly interpolated values for the metallicity, Z.
        
#         irreg_alpha_interp (1d array):    The linearly interpolated values for [alpha/Fe].
        
#         irreg_mod_interp (4d array):      The first 3 axes are the parameter values used, and the 4th
#                                         axis the corresponding index measurements from the model.
        
#         '''
        
#         irreg_age = np.unique(self.irreg_mods["Age"])
#         irreg_Z = np.unique(self.irreg_mods["[Fe/H]"])
        
#         self.irreg_free_param = len(self.irreg_names) - 3
        
#         if self.hi_res == True:
            
#             self.irreg_age_interp = np.geomspace(bn.nanmin(irreg_age),
#                                               bn.nanmax(irreg_age), 220)
#             self.irreg_Z_interp = np.linspace(bn.nanmin(irreg_Z),
#                                               bn.nanmax(irreg_Z), 90)
#             self.irreg_alpha_interp = np.linspace(-0.3, 0.5, 81)
            
#             self.irreg_mod_interp = dict()
            
#             for name in self.irreg_names:
#                 try:
#                     self.irreg_mod_interp[name] = \
#                         np.load("templates/irreg_interpolated_hi_res/{}.npy".format(name))
                    
#                 except:
                    
#                     self.irreg_create_interp(name)
                
#         else:
            
#             self.irreg_age_interp = np.linspace(bn.nanmin(irreg_age),
#                                               bn.nanmax(irreg_age), 12)
#             self.irreg_Z_interp = np.linspace(bn.nanmin(irreg_Z),
#                                               bn.nanmax(irreg_Z), 9)
#             self.irreg_alpha_interp = np.linspace(-0.3, 0.5, 9)
            
#             self.irreg_mod_interp = dict()
            
#             for name in self.irreg_names:
#                 try:
#                     self.irreg_mod_interp[name] = \
#                         np.load("templates/irreg_interpolated_lo_res/{}.npy".format(name))
                    
#                 except:
                    
#                     self.irreg_create_interp(name)

#         return
    
# ###################################################################################################
    
#     def irreg_create_interp(self, name):
        
#         '''Generates the interpolated Schiavon models. 
#         The models are first interpolated to a regular grid within the confines of the model. The 
#         [alpha/Fe] axis is then extrapolated to the range [-0.3, 0.5].
#         Saves the output to a file for future use.
        
#         Attributes
#         ----------
#         irreg_mod_interp:     4D array. The first 3 axes are the parameter values used, and the 4th
#                             axis the corresponding index measurements from the model.
#         '''
        
#         irreg_alpha = np.unique(self.irreg_mods["[alpha/Fe]"])
        
#         if self.hi_res == True:
#             old_alpha_interp = np.linspace(bn.nanmin(irreg_alpha),
#                                               bn.nanmax(irreg_alpha), 43)
#         else:
#             old_alpha_interp = np.linspace(bn.nanmin(irreg_alpha),
#                                        bn.nanmax(irreg_alpha), 5)
        
#         X, Y, Z = np.meshgrid(self.irreg_age_interp, self.irreg_Z_interp, 
#                               old_alpha_interp, indexing='ij')
        
#         interp = griddata((self.irreg_mods["Age"],self.irreg_mods["[Fe/H]"],
#                                               self.irreg_mods["[alpha/Fe]"]), 
#                                              self.irreg_mods[self.LI_to_sch[name]], (X,Y,Z),
#                                              method = 'linear')
        
#         new_array = np.full((len(self.irreg_age_interp), len(self.irreg_Z_interp), 
#                              len(self.irreg_alpha_interp)), np.nan)
        
#         for i,age in enumerate(self.irreg_age_interp):
#             for j,Z in enumerate(self.irreg_Z_interp):
                
#                 alpha = interp[i,j,:]
                
#                 if bn.anynan(alpha) == False:
                    
#                     spl = InterpolatedUnivariateSpline(old_alpha_interp, alpha, k=1)
#                     alpha_extrap = spl(self.irreg_alpha_interp)
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
                    
#                     lo_idx = np.searchsorted(self.irreg_alpha_interp, lower_lim)
#                     hi_idx = np.searchsorted(self.irreg_alpha_interp, upper_lim, 
#                                              side = 'right')
                    
#                     alpha_extrap = spl(self.irreg_alpha_interp[lo_idx:hi_idx])
                    
#                     new_array[i,j,lo_idx:hi_idx] = alpha_extrap
                    
#                     continue
                
#                 else:
                    
#                     val = old_alpha_interp[idx_check]
                    
#                     idx = np.searchsorted(self.irreg_alpha_interp, val)
                    
#                     new_array[i,j,idx] = alpha[idx_check]
                    
#         self.irreg_mod_interp[name] = new_array
    
#         if self.hi_res == True:
#             if not os.path.exists("templates/irreg_interpolated_hi_res/"):
#                 os.mkdir("templates/irreg_interpolated_hi_res/")
#             np.save("templates/irreg_interpolated_hi_res/{}.npy".format(name), new_array)
#         else:            
#             if not os.path.exists("templates/irreg_interpolated_lo_res/"):
#                 os.mkdir("templates/irreg_interpolated_lo_res/")
#             np.save("templates/irreg_interpolated_lo_res/{}.npy".format(name), new_array)
                
#         del X,Y,Z, interp
#         return
    
# ###################################################################################################

#     def irreg_disp_correction(self):
        
#         '''Dispersion correction for the Schiavon models.
        
#         '''
        
#         print ("Beginning correction")
        
#         self.irreg_copy = copy.deepcopy(self.eq_widths)
        
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
            
#             end = self.irreg_corr_sub(p, a)
            
#             if end:
#                 break
            
            
#         self.eq_widths, self.irreg_copy = self.irreg_copy, self.eq_widths
        
        
#         return
        
        
#     def irreg_corr_sub(self, tab_path, age):
        
#         with open(tab_path, 'rb') as outfile:
#             corr_tab = pickle.load(outfile)
        
#         sig_idx = np.searchsorted(corr_tab["sigma"], self.sigma)
        
        
#         for n in self.irreg_names:
#             if not np.isfinite(corr_tab[n][sig_idx]):
#                 self.eq_widths[n] = self.irreg_copy[n]
#             elif corr_tab[n+"_units"] == 1:
#                 self.eq_widths[n] = self.irreg_copy[n] + corr_tab[n][sig_idx]
#             else:
#                 self.eq_widths[n] = self.irreg_copy[n] * corr_tab[n][sig_idx]
                
#         self.irreg_measure(iterate = False)
#         # self.chi_plot()
        
#         if self.ssp_age_sch <= age:
#             print ("Successfully converged to a solution, with age below "
#                    "{} Gyr.\n".format(age))
#             return True
#         else:
#             print ("No solution found yet.\n")
#             return False 
    
###################################################################################################

    def reg_disp_correction(self):
        
        '''Dispersion correction for the reg models.
        
        THIS NEEDS A SERIOUS REWRITE.
        
        To do:
            
        - This function needs to be rewritten. Should not be linear progression, but looped with
          relevant function calls.
        - Write an equivalent set for reg - or combine the two into one set of corrections.
        - Should be able to run automatically, and choose if indices need correction for dispersion
        - Output should be as tuple, 
        
        IMPORTANT -- USE SIGMA_FLAG FROM INITIAL TABLE TO DECIDE IF THIS METHOD IS RUN!
            
            
        '''
        
        '''Dispersion correction for the Schiavon models.
        
        '''
        
        self.reg_copy = copy.deepcopy(self.eq_widths)
        self.reg_errs_copy = copy.deepcopy(self.width_errs)
        
        
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
        
        self.reg_check()
        
        with self.err_log_file.open("a") as f:
            f.write("Checked indices: {}".format(
                time.time() - self.start_time
            )+"\n")
        
        
        for p, a in zip(templates, ages):
            
            end = self.reg_corr_sub(p, a)
            
            if end:
                break
        
        # print (self.eq_widths)
        # self.corr_widths = copy.deepcopy(self.eq_widths)
        # print (self.corr_widths)
        # print (self.reg_copy)
        # self.eq_widths = copy.deepcopy(self.reg_copy)
        # print (self.eq_widths)
        # print (self.corr_widths)
            
        self.eq_widths, self.reg_copy = self.reg_copy, self.eq_widths
        self.width_errs, self.reg_errs_copy = self.reg_errs_copy, self.width_errs
        
        self.corr_widths = self.reg_copy
        self.corr_width_errs = self.reg_errs_copy
        
            
        return
        
        
    def reg_corr_sub(self, tab_path, age):
        
        with open(tab_path, 'rb') as outfile:
            corr_tab = pickle.load(outfile)
            
        with self.err_log_file.open("a") as f:
            f.write("Loaded correction table: {}".format(
                time.time() - self.start_time
            )+"\n")
        
        sig_idx = np.searchsorted(corr_tab["sigma"], self.sigma)
        
        
        for n in self.reg_names:
            sig_idx = np.clip(
                sig_idx, 
                a_min=0, 
                a_max=len(corr_tab[n])-1,
            )
            if not np.isfinite(corr_tab[n][sig_idx]):
                self.eq_widths[n] = self.reg_copy[n]
            elif corr_tab[n+"_units"] == 1:
                self.eq_widths[n] = self.reg_copy[n] + corr_tab[n][sig_idx]
                self.width_errs[n] = self.reg_errs_copy[n] + corr_tab[n][sig_idx]
            else:
                self.eq_widths[n] = self.reg_copy[n] * corr_tab[n][sig_idx]
                self.width_errs[n] = self.reg_errs_copy[n] * corr_tab[n][sig_idx]
                
        with self.err_log_file.open("a") as f:
            f.write("Applied corrections to indices: {}".format(
                time.time() - self.start_time
            )+"\n")
        
        self.reg_measure_sub()
        
        if self.ssp_age_reg <= age:
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
        
    def reg_measure_sub(self):
        
        '''A method to calculate the chi^2 array for all possible parameter values, given 
        reg_idx_check exists, using the models from Thomas, Maraston and Johansson (2011). Finds 
        the minimum chi^2 value and the corresponding SSP parameters.
        
        Attributes
        ----------
        reg_chi (3d array):         The reduced chi^2 values corresponding to the fit of the model 
                                    to the observations.
        
        reg_chi_idx (1d array):     The index of the minimum value in the chi^2 array.
        
        ssp_age_reg (float):        The most likely value for the age of the stellar population.
                                    
        ssp_Z_reg (float):          The most likely value for the metallicity, Z.
        
        ssp_alpha_reg (float):      The most likely value for [alpha/Fe].
        
        '''
        
        names = [n for n in self.reg_names if self.reg_idx_check[n]]
        
        # self.reg_chi = np.empty_like(self.reg_mod_interp[names[0]])
        
        self.reg_free_param = len(names) - 3
        
        # for ijk in np.ndindex(self.reg_chi.shape):
        #     self.reg_chi[ijk] = (bn.nansum([((self.eq_widths[n] - 
        #                                       self.reg_mod_interp[n][ijk])/
        #                                      self.width_errs[n])**2 for n in names])/
        #                          self.reg_free_param)
            
        
        eq_arr = np.asarray([self.eq_widths[n] for n in names], dtype=float).T
        err_arr = np.asarray([self.width_errs[n] for n in names], dtype=float).T
        
        with self.err_log_file.open("a") as f:
            f.write("Generated index arrays: {}".format(
                time.time() - self.start_time
            )+"\n")

        # reg_mod_arr = np.array([self.reg_mod_interp[n] for n in names]).T
        # # mod_err_arr = np.array([self.reg_err_interp[n] for n in names]).T
        if self.reg_mod_arr is None:
            self.reg_mod_arr = np.asarray([self.reg_mod_interp[n] for n in names], dtype=float).T
        
        with self.err_log_file.open("a") as f:
            f.write("Generated models array: {}".format(
                time.time() - self.start_time
            )+"\n")
            
        print (eq_arr.dtype, flush=True)
    
        self.reg_chi = (bn.nansum(
            (
                (eq_arr - self.reg_mod_arr)/err_arr#np.sqrt(err_arr**2+mod_err_arr**2)
            )**2,
            axis=-1,
        )/self.reg_free_param).T
        
        with self.err_log_file.open("a") as f:
            f.write("Created chi^2 array: {}".format(
                time.time() - self.start_time
            )+"\n")
            
        self.reg_chi_idx = np.unravel_index(bn.nanargmin(self.reg_chi, 
                                axis=None), self.reg_chi.shape)
        
        with self.err_log_file.open("a") as f:
            f.write("Found minimum: {}".format(time.time() - self.start_time)+"\n")
        
        self.reg_chi_min = self.reg_chi[self.reg_chi_idx]
        
        self.ssp_age_reg = self.reg_age_interp[self.reg_chi_idx[0]]
        self.ssp_Z_reg = self.reg_Z_interp[self.reg_chi_idx[1]]        
        self.ssp_alpha_reg = self.reg_alpha_interp[self.reg_chi_idx[2]]
        
        return
    
###################################################################################################
    
    def reg_check(self):
        
        self.reg_idx_check = dict()
        for name in self.reg_names:
            reg_max = bn.nanmax(self.reg_mods[name],axis=0)
            reg_min = np.nanmin(self.reg_mods[name],axis=0)
            
            if self.eq_widths[name] - np.abs(self.width_errs[name]) > reg_max:
                self.reg_idx_check[name] = False
            elif self.eq_widths[name] + np.abs(self.width_errs[name]) < reg_min:
                self.reg_idx_check[name] = False
            elif not np.isfinite(self.eq_widths[name]):
                self.reg_idx_check[name] = False
            else:
                self.reg_idx_check[name] = True
    
###################################################################################################
    
    def reg_measure(self, iterate = False):
        
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
        reg_idx_check (1d bool array):  The array values show whether the index should be included
                                        in the fit.
        
        '''
        
        self.reg_check()
            
        ### Must have at least 5 indices in the model range
        if np.sum([val for val in self.reg_idx_check.values()]) < 5:
            self.ssp_age_reg = np.nan
            self.ssp_Z_reg = np.nan            
            self.ssp_alpha_reg = np.nan
            
        ### Must have at least one Balmer line
        elif (("Hdelta_A" in self.reg_idx_check == False or 
               self.reg_idx_check["Hdelta_A"] == False) and 
              ("Hdelta_F" in self.reg_idx_check == False or 
               self.reg_idx_check["Hdelta_F"] == False) and
              ("Hgamma_A" in self.reg_idx_check == False or 
               self.reg_idx_check["Hgamma_A"] == False) and
              ("Hgamma_F" in self.reg_idx_check == False or 
               self.reg_idx_check["Hgamma_F"] == False) and
              ("Hbeta" in self.reg_idx_check == False or 
               self.reg_idx_check["Hbeta"] == False)):
            
            self.ssp_age_reg = np.nan
            self.ssp_Z_reg = np.nan            
            self.ssp_alpha_reg = np.nan
            
        ### Must have at least one Fe index
        elif (("Fe4383" in self.reg_idx_check == False or 
               self.reg_idx_check["Fe4383"] == False) and 
              ("Fe4531" in self.reg_idx_check == False or 
               self.reg_idx_check["Fe4531"] == False) and
              ("Fe4668" in self.reg_idx_check == False or 
               self.reg_idx_check["Fe4668"] == False) and
              ("Fe5015" in self.reg_idx_check == False or 
               self.reg_idx_check["Fe5015"] == False) and
              ("Fe5270" in self.reg_idx_check == False or 
               self.reg_idx_check["Fe5270"] == False) and
              ("Fe5335" in self.reg_idx_check == False or 
               self.reg_idx_check["Fe5335"] == False) and
              ("Fe5406" in self.reg_idx_check == False or 
               self.reg_idx_check["Fe5406"] == False)):
            
            self.ssp_age_reg = np.nan
            self.ssp_Z_reg = np.nan            
            self.ssp_alpha_reg = np.nan
            
        else:
            self.reg_measure_sub()
                
        # if iterate:
            
        # # Following code is an implementation of an iterative search for 
        # # the minimum, rejecting the greatest outlier at the minimum if
        # # the result is not significant enough
            
        #     print ('')
        #     print ('Minimum Chi^2')
        #     print (self.reg_chi[self.reg_chi_idx])
        #     print ('')
            
        #     current_min = scipy_chi2.sf(self.reg_chi[self.reg_chi_idx], 
        #                                 self.reg_free_param)
            
        #     while current_min <= 0.95:
            
        #         chi_min = ma.array([((d-m)/e)**2 for d,e,m in zip(self.reg_lick_dat, 
        #                                               self.reg_lick_err,
        #                                               self.reg_mod_interp[self.reg_chi_idx])],
        #                             mask = np.invert(self.reg_idx_check))
                
        #         to_remove = bn.nanmax(chi_min[~chi_min.mask])
        #         for i, chi in enumerate(chi_min):
        #             if chi == to_remove:
        #                 self.reg_idx_check[i] = False
        
        #         self.reg_measure_sub()
                
        #         print ('')
        #         print ('Minimum Chi^2')
        #         print (self.reg_chi[self.reg_chi_idx])
        #         print ('')
                
        #         current_min = scipy_chi2.sf(self.reg_chi[self.reg_chi_idx], 
        #                                     self.reg_free_param)
                
        #         print ('')
        #         print ('Probability')
        #         print (current_min)
        #         print ('')
                
        return
    
##################################################################################################
    
    def irreg_measure_sub(self):
        
        '''A method to calculate the chi^2 array for all possible parameter values, given 
        irreg_idx_check exists, using the models from Thomas, Maraston and Johansson (2011). Finds 
        the minimum chi^2 value and the corresponding SSP parameters.
        
        Attributes
        ----------
        irreg_chi (3d array):         The reduced chi^2 values corresponding to the fit of the model 
                                    to the observations.
        
        irreg_chi_idx (1d array):     The index of the minimum value in the chi^2 array.
        
        ssp_age_reg (float):        The most likely value for the age of the stellar population.
                                    
        ssp_Z_reg (float):          The most likely value for the metallicity, Z.
        
        ssp_alpha_reg (float):      The most likely value for [alpha/Fe].
        
        '''
        
        names = [n for n in self.irreg_names if self.irreg_idx_check[n]]
        
        # self.irreg_chi = np.empty_like(self.irreg_mod_interp[names[0]])
        
        self.irreg_free_param = len(names) - 3
        
        # for ijk in np.ndindex(self.irreg_chi.shape):
        #     self.irreg_chi[ijk] = (bn.nansum([((self.eq_widths[n] - 
        #                                       self.irreg_mod_interp[n][ijk])/
        #                                      self.width_errs[n])**2 for n in names])/
        #                          self.irreg_free_param)
            
        
        eq_arr = np.asarray([self.eq_widths[n] for n in names], dtype=float).T
        err_arr = np.asarray([self.width_errs[n] for n in names], dtype=float).T
        
        with self.err_log_file.open("a") as f:
            f.write("Generated index arrays: {}".format(
                time.time() - self.start_time
            )+"\n")

        # reg_mod_arr = np.array([self.irreg_mod_interp[n] for n in names]).T
        # # mod_err_arr = np.array([self.irreg_err_interp[n] for n in names]).T
        if self.irreg_mod_arr is None:
            self.irreg_mod_arr = np.asarray([self.irreg_mod_interp[n] for n in names], dtype=float).T
        
        with self.err_log_file.open("a") as f:
            f.write("Generated models array: {}".format(
                time.time() - self.start_time
            )+"\n")
            
        print (eq_arr.dtype, flush=True)
    
        self.irreg_chi = (
            (eq_arr - self.irreg_mod_arr)/err_arr#np.sqrt(err_arr**2+mod_err_arr**2)
        )**2
            
        
        _is_nan = np.isnan(self.irreg_chi)
        nan_idx = np.argwhere(_is_nan.all(axis=-1))
        
        print (_is_nan.shape)
        
        print (self.irreg_free_param)
        
        self.irreg_chi = bn.nansum(self.irreg_chi, axis=-1)/self.irreg_free_param
        
        print (self.irreg_chi.shape)
        
        self.irreg_chi[nan_idx[:,0], nan_idx[:,1],nan_idx[:,2]] = np.nan
        
        self.irreg_chi = self.irreg_chi.T
        
        with self.err_log_file.open("a") as f:
            f.write("Created chi^2 array: {}".format(
                time.time() - self.start_time
            )+"\n")
            
        self.irreg_chi_idx = np.unravel_index(bn.nanargmin(self.irreg_chi, 
                                axis=None), self.irreg_chi.shape)
        
        with self.err_log_file.open("a") as f:
            f.write("Found minimum: {}".format(time.time() - self.start_time)+"\n")
        
        self.irreg_chi_min = self.irreg_chi[self.irreg_chi_idx]
        
        self.ssp_age_reg = self.irreg_age_interp[self.irreg_chi_idx[0]]
        self.ssp_Z_reg = self.irreg_Z_interp[self.irreg_chi_idx[1]]        
        self.ssp_alpha_reg = self.irreg_alpha_interp[self.irreg_chi_idx[2]]
        
        return
    
###################################################################################################
    
    def irreg_check(self):
        
        self.irreg_idx_check = dict()
        for name in self.irreg_names:
            irreg_max = bn.nanmax(self.irreg_mods[name],axis=0)
            irreg_min = np.nanmin(self.irreg_mods[name],axis=0)
            
            if self.eq_widths[name] - np.abs(self.width_errs[name]) > irreg_max:
                self.irreg_idx_check[name] = False
            elif self.eq_widths[name] + np.abs(self.width_errs[name]) < irreg_min:
                self.irreg_idx_check[name] = False
            elif not np.isfinite(self.eq_widths[name]):
                self.irreg_idx_check[name] = False
            else:
                self.irreg_idx_check[name] = True
    
###################################################################################################
    
    def irreg_measure(self, iterate = False):
        
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
        irreg_idx_check (1d bool array):  The array values show whether the index should be included
                                        in the fit.
        
        '''
        
        self.irreg_check()
            
        ### Must have at least 5 indices in the model range
        if np.sum([val for val in self.irreg_idx_check.values()]) < 5:
            self.ssp_age_reg = np.nan
            self.ssp_Z_reg = np.nan            
            self.ssp_alpha_reg = np.nan
            
        ### Must have at least one Balmer line
        elif (("Hdelta_A" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Hdelta_A"] == False) and 
              ("Hdelta_F" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Hdelta_F"] == False) and
              ("Hgamma_A" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Hgamma_A"] == False) and
              ("Hgamma_F" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Hgamma_F"] == False) and
              ("Hbeta" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Hbeta"] == False)):
            
            self.ssp_age_reg = np.nan
            self.ssp_Z_reg = np.nan            
            self.ssp_alpha_reg = np.nan
            
        ### Must have at least one Fe index
        elif (("Fe4383" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Fe4383"] == False) and 
              ("Fe4531" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Fe4531"] == False) and
              ("Fe4668" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Fe4668"] == False) and
              ("Fe5015" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Fe5015"] == False) and
              ("Fe5270" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Fe5270"] == False) and
              ("Fe5335" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Fe5335"] == False) and
              ("Fe5406" in self.irreg_idx_check == False or 
               self.irreg_idx_check["Fe5406"] == False)):
            
            self.ssp_age_reg = np.nan
            self.ssp_Z_reg = np.nan            
            self.ssp_alpha_reg = np.nan
            
        else:
            self.irreg_measure_sub()
                
        # if iterate:
            
        # # Following code is an implementation of an iterative search for 
        # # the minimum, rejecting the greatest outlier at the minimum if
        # # the result is not significant enough
            
        #     print ('')
        #     print ('Minimum Chi^2')
        #     print (self.irreg_chi[self.irreg_chi_idx])
        #     print ('')
            
        #     current_min = scipy_chi2.sf(self.irreg_chi[self.irreg_chi_idx], 
        #                                 self.irreg_free_param)
            
        #     while current_min <= 0.95:
            
        #         chi_min = ma.array([((d-m)/e)**2 for d,e,m in zip(self.irreg_lick_dat, 
        #                                               self.irreg_lick_err,
        #                                               self.irreg_mod_interp[self.irreg_chi_idx])],
        #                             mask = np.invert(self.irreg_idx_check))
                
        #         to_remove = bn.nanmax(chi_min[~chi_min.mask])
        #         for i, chi in enumerate(chi_min):
        #             if chi == to_remove:
        #                 self.irreg_idx_check[i] = False
        
        #         self.irreg_measure_sub()
                
        #         print ('')
        #         print ('Minimum Chi^2')
        #         print (self.irreg_chi[self.irreg_chi_idx])
        #         print ('')
                
        #         current_min = scipy_chi2.sf(self.irreg_chi[self.irreg_chi_idx], 
        #                                     self.irreg_free_param)
                
        #         print ('')
        #         print ('Probability')
        #         print (current_min)
        #         print ('')
                
        return
    
# ###################################################################################################
    
#     def irreg_chi_sum(self, ijk, names):
        
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
        
#         if np.any(self.irreg_mod_interp[names[0]][ijk] == np.nan):
#             return np.nan
#         else:
#             return (bn.nansum([((self.eq_widths[n] - self.irreg_mod_interp[n][ijk])/
#                                 self.width_errs[n])**2 for n in names])/self.irreg_free_param)
    
# ###################################################################################################
    
    # def irreg_measure_sub(self):
        
    #     '''A method to calculate the chi^2 array for all possible parameter values, given 
    #     irreg_idx_check exists, using the models from Schiavon (2007). Finds the minimum chi^2 value 
    #     and the corresponding SSP parameters.
        
    #     Attributes
    #     ----------
    #     irreg_chi (3d array):         The reduced chi^2 values corresponding to the fit of the model 
    #                                 to the observations.
        
    #     irreg_chi_idx (1d array):     The index of the minimum value in the chi^2 array.
        
    #     ssp_age_sch (float):        The most likely value for the age of the stellar population.
                                    
    #     ssp_Z_sch (float):          The most likely value for the metallicity, Z.
        
    #     ssp_alpha_sch (float):      The most likely value for [alpha/Fe].
        
    #     '''
        
        
    #     names = [n for n in self.irreg_names if self.irreg_idx_check[n]]
        
    #     self.irreg_chi = np.empty_like(self.irreg_mod_interp[names[0]])
        
    #     for ijk in np.ndindex(self.irreg_chi.shape):
    #         self.irreg_chi[ijk] = self.irreg_chi_sum(ijk, names)
        
    #     self.irreg_chi = bn.replace(self.irreg_chi, 0, np.nan)
        
    #     self.irreg_chi_idx = np.unravel_index(bn.nanargmin(self.irreg_chi, 
    #                             axis=None), self.irreg_chi.shape)
        
    #     self.irreg_chi_min = self.irreg_chi[self.irreg_chi_idx]
        
    #     self.ssp_age_sch = self.irreg_age_interp[self.irreg_chi_idx[0]]
    #     self.ssp_Z_sch = self.irreg_Z_interp[self.irreg_chi_idx[1]]        
    #     self.ssp_alpha_sch = self.irreg_alpha_interp[self.irreg_chi_idx[2]]
        
    #     return
    
# ###################################################################################################
    
#     def irreg_measure(self, iterate = False):
        
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
#         irreg_idx_check (1d bool array):  The array values show whether the index should be included 
#                                         in the fit.
        
#         '''
        
#         self.irreg_idx_check = dict()
        
#         for name in self.irreg_names:
#             irreg_max = bn.nanmax(self.irreg_mods[name],axis=0)
#             irreg_min = np.nanmin(self.irreg_mods[name],axis=0)
            
#             if self.eq_widths[name] - np.abs(self.width_errs[name]) > irreg_max:
#                 self.irreg_idx_check[name] = False
#             elif self.eq_widths[name] + np.abs(self.width_errs[name]) < irreg_min:
#                 self.irreg_idx_check[name] = False
#             else:
#                 self.irreg_idx_check[name] = True
            
#         ### Must have at least 5 indices in the model range
#         if np.sum([val for val in self.irreg_idx_check.values()]) < 5:
#             self.ssp_age_sch = np.nan
#             self.ssp_Z_sch = np.nan            
#             self.ssp_alpha_sch = np.nan
            
#         ### Must have at least one Balmer line
#         elif (("Hdelta_A" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Hdelta_A"] == False) and 
#               ("Hdelta_F" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Hdelta_F"] == False) and
#               ("Hgamma_A" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Hgamma_A"] == False) and
#               ("Hgamma_F" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Hgamma_F"] == False) and
#               ("Hbeta" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Hbeta"] == False)):
            
#             self.ssp_age_sch = np.nan
#             self.ssp_Z_sch = np.nan            
#             self.ssp_alpha_sch = np.nan
            
#         ### Must have at least one Fe index
#         elif (("Fe4383" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Fe4383"] == False) and 
#               ("Fe4531" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Fe4531"] == False) and
#               ("Fe4668" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Fe4668"] == False) and
#               ("Fe5015" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Fe5015"] == False) and
#               ("Fe5270" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Fe5270"] == False) and
#               ("Fe5335" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Fe5335"] == False) and
#               ("Fe5406" in self.irreg_idx_check == False or 
#                self.irreg_idx_check["Fe5406"] == False)):
            
#             self.ssp_age_sch = np.nan
#             self.ssp_Z_sch = np.nan            
#             self.ssp_alpha_sch = np.nan
            
#         else:
#             self.irreg_measure_sub()
        
            
#         # if iterate:
            
#         # # Following code is an implementation of an iterative search for 
#         # # the minimum, rejecting the greatest outlier at the minimum if
#         # # the result is not significant enough
        
#         #     print ('')
#         #     print ('Minimum Chi^2')
#         #     print (self.irreg_chi[self.irreg_chi_idx])
#         #     print ('')
            
#         #     current_min = scipy_chi2.sf(self.irreg_chi[self.irreg_chi_idx], 
#         #                                 self.irreg_free_param)
            
#         #     while current_min <= 0.95:
            
#         #         chi_min = ma.array([((d-m)/e)**2 for d,e,m in zip(self.irreg_lick_dat, 
#         #                                               self.irreg_lick_err,
#         #                                               self.irreg_mod_interp[self.irreg_chi_idx])],
#         #                             mask = np.invert(self.irreg_idx_check))
                
#         #         to_remove = bn.nanmax(chi_min[~chi_min.mask])
#         #         for i, chi in enumerate(chi_min):
#         #             if chi == to_remove:
#         #                 self.irreg_idx_check[i] = False
        
#         #         self.irreg_measure_sub()
                
#         #         print ('')
#         #         print ('Minimum Chi^2')
#         #         print (self.irreg_chi[self.irreg_chi_idx])
#         #         print ('')
                
#         #         current_min = scipy_chi2.sf(self.irreg_chi[self.irreg_chi_idx], 
#         #                                     self.irreg_free_param)
                
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
        
        if hasattr(self, 'reg_chi'):
            
            # cols = ["0.6", "k", "red"]
            # cols = ["red", "k", "0.6"]
            # cmap1 = colors.LinearSegmentedColormap.from_list("mycmap", cols)
            
            plt.figure(figsize=(15,10))
            Z, alpha = np.meshgrid(self.reg_Z_interp, self.reg_alpha_interp, indexing='ij')
            plt.pcolormesh(
                Z, 
                alpha, 
                self.reg_chi[self.reg_chi_idx[0],:,:],
                norm=colors.LogNorm(vmin=1, vmax=1000), 
                cmap=plt.cm.jet,
                # cmap=cmap1
            )
            
            colorbar = plt.colorbar()
            colorbar.set_label(r"$\chi^2_{\nu}$", 
                               fontsize=14, rotation=0, y=1.05, labelpad=-40)
            
            plt.contour(Z, alpha, self.reg_chi[self.reg_chi_idx[0], :, :], 
                        levels=[self.reg_chi[self.reg_chi_idx] + 1,
                                self.reg_chi[self.reg_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, origin = 'lower')
            
            plt.scatter(self.reg_Z_interp[self.reg_chi_idx[1]],
                        self.reg_alpha_interp[self.reg_chi_idx[2]], color='r')
            plt.xlabel(r'$[Z/H]$', fontsize=14)
            plt.ylabel(r'$[\alpha/\rm{Fe}]$', fontsize=14)
            plt.show()
            
            plt.figure(figsize=(15,10))
            age, Z = np.meshgrid(self.reg_age_interp, self.reg_Z_interp, indexing='ij')
            plt.pcolormesh(age, Z, self.reg_chi[:, :, self.reg_chi_idx[2]],
                           norm=colors.LogNorm(vmin=1, vmax=1000), 
                           cmap=plt.cm.jet)
            
            colorbar = plt.colorbar()
            colorbar.set_label(r"$\chi^2_{\nu}$", 
                               fontsize=14, rotation=0, y=1.05, labelpad=-40)
            
            plt.contour(age, Z, self.reg_chi[:, :, self.reg_chi_idx[2]], 
                        levels=[self.reg_chi[self.reg_chi_idx] + 1,
                                self.reg_chi[self.reg_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, 
                        origin = 'lower')
    
    
            plt.scatter(self.reg_age_interp[self.reg_chi_idx[0]],
                        self.reg_Z_interp[self.reg_chi_idx[1]],color='r')
            plt.xlabel(r'$\rm{Age\ (Gyr)}$', fontsize=14)
            plt.ylabel(r'$\rm{[Z/H]}$', fontsize=14)
            plt.show()
            
            plt.figure(figsize=(15,10))
            alpha, age = np.meshgrid(self.reg_alpha_interp, self.reg_age_interp)
            plt.pcolormesh(alpha, age, self.reg_chi[:, self.reg_chi_idx[1], :],
                           norm=colors.LogNorm(vmin=1, vmax=1000),
                           cmap=plt.cm.jet)
            
            colorbar = plt.colorbar()
            colorbar.set_label(r"$\chi^2_{\nu}$", 
                               fontsize=14, rotation=0, y=1.05, labelpad=-40)
            
            plt.contour(alpha, age, self.reg_chi[:, self.reg_chi_idx[1], :], 
                        levels=[self.reg_chi[self.reg_chi_idx] + 1,
                                self.reg_chi[self.reg_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, origin = 'lower')
    
            plt.scatter(self.reg_alpha_interp[self.reg_chi_idx[2]],
                        self.reg_age_interp[self.reg_chi_idx[0]], color='r')
            plt.ylabel(r'$\rm{Age\ (Gyr)}$', fontsize=14)
            plt.xlabel(r'$[\alpha/\rm{Fe}]$', fontsize=14)
            plt.show()
        
        if hasattr(self, 'irreg_chi'):
                                
            plt.figure(figsize=(20,10))
            plt.imshow(self.irreg_chi[self.irreg_chi_idx[0], :, :], 
                           norm=colors.LogNorm(vmin=1, vmax=1000),
                           cmap=plt.cm.jet, 
                       extent = [self.irreg_alpha_interp[0], 
                                 self.irreg_alpha_interp[-1],
                                 self.irreg_Z_interp[0], self.irreg_Z_interp[-1]], 
                       origin = 'lower', aspect = 1/3.65)
            plt.colorbar()
            
            plt.contour(self.irreg_chi[self.irreg_chi_idx[0], :, :], 
                        levels=[self.irreg_chi[self.irreg_chi_idx] + 1,
                                self.irreg_chi[self.irreg_chi_idx] + 3.5], 
                        colors='white', alpha=1.0,
                        extent = [self.irreg_alpha_interp[0], 
                                  self.irreg_alpha_interp[-1], self.irreg_Z_interp[0], 
                                  self.irreg_Z_interp[-1]], origin = 'lower')
            
            plt.scatter(self.irreg_alpha_interp[self.irreg_chi_idx[2]],
                        self.irreg_Z_interp[self.irreg_chi_idx[1]], color='r')
            plt.xlabel(r'$[\alpha/\rm{Fe}]$', fontsize=14)
            plt.ylabel(r'$Z$', fontsize=14)
            plt.show()
    
            plt.figure(figsize=(15,10))
        
            age, Z = np.meshgrid(self.irreg_age_interp, self.irreg_Z_interp, indexing='ij')
            plt.pcolormesh(age,Z, self.irreg_chi[:, :, self.irreg_chi_idx[2]], 
                           norm=colors.LogNorm(vmin=1, vmax=1000),
                           cmap=plt.cm.jet)
            plt.colorbar()
            
            plt.contour(age, Z, self.irreg_chi[:, :, self.irreg_chi_idx[2]], 
                        levels=[self.irreg_chi[self.irreg_chi_idx] + 1,
                                self.irreg_chi[self.irreg_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, 
                        origin = 'lower')
    
    
            plt.scatter(self.irreg_age_interp[self.irreg_chi_idx[0]],
                        self.irreg_Z_interp[self.irreg_chi_idx[1]],color='r')
            plt.xlabel(r'$\rm{Age\ (Gyr)}$', fontsize=14)
            plt.ylabel(r'$\rm{Z}$', fontsize=14)
            plt.show()
            
            plt.figure(figsize=(15,10))
            alpha, age = np.meshgrid(self.irreg_alpha_interp, self.irreg_age_interp)
            plt.pcolormesh(alpha,age, self.irreg_chi[:, self.irreg_chi_idx[1], :],
                           norm=colors.LogNorm(vmin=1, vmax=1000),
                           cmap=plt.cm.jet)
            
            plt.colorbar()
            
            plt.contour(alpha, age, self.irreg_chi[:, self.irreg_chi_idx[1], :], 
                        levels=[self.irreg_chi[self.irreg_chi_idx] + 1,
                                self.irreg_chi[self.irreg_chi_idx] + 3.5], 
                        colors='white', alpha=1.0, origin = 'lower')
    
            plt.scatter(self.irreg_alpha_interp[self.irreg_chi_idx[2]],
                        self.irreg_age_interp[self.irreg_chi_idx[0]], color='r')
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
        ssp_age_reg_bounds (1d array):      The upper and lower bounds of the 1-sigma CI on 
                                            age using the reg models.
                                            
        ssp_Z_reg_bounds (1d array):        The upper and lower bounds of the 1-sigma CI on Z using
                                            the reg models.
                                            
        ssp_alpha_reg_bounds (1d array):    The upper and lower bounds of the 1-sigma CI on 
                                            [alpha/Fe] using the reg models.
                                            
        ssp_age_irreg_bounds (1d array):      The upper and lower bounds of the 1-sigma CI on 
                                            age using the Schiavon models.
                                            
        ssp_Z_irreg_bounds (1d array):        The upper and lower bounds of the 1-sigma CI on Z using
                                            the Schiavon models.
                                            
        ssp_alpha_irreg_bounds (1d array):    The upper and lower bounds of the 1-sigma CI on 
                                            [alpha/Fe] using the Schiavon models.                
        '''
        
        if hasattr(self, 'reg_chi'):
            
            err_msg_reg = ("\nError in reg SSP error estimation. \n"
                           "Assume results unreliable. \n\n")
            
            reg_age_slice = np.nanmin(self.reg_chi, axis = (1,2))
            reg_age_region = self.reg_age_interp[np.where(reg_age_slice <=
                                                          (self.reg_chi[self.reg_chi_idx] + 1))]
            try:
                self.ssp_age_reg_bounds = [reg_age_region[0], reg_age_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_reg)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_reg)
                
                self.ssp_age_reg_bounds = [np.nan, np.nan]
                
            reg_Z_slice = np.nanmin(self.reg_chi, axis = (0,2))
            reg_Z_region = self.reg_Z_interp[np.where(reg_Z_slice <= 
                                                      (self.reg_chi[self.reg_chi_idx] + 1))]
            try:
                self.ssp_Z_reg_bounds = [reg_Z_region[0], reg_Z_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_reg)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_reg)
                
                self.ssp_z_reg_bounds = [np.nan, np.nan]
            
            reg_alpha_slice = np.nanmin(self.reg_chi, axis = (0,1))
            reg_alpha_region = self.reg_alpha_interp[np.where(reg_alpha_slice <= 
                                                          (self.reg_chi[self.reg_chi_idx] + 1))]
            try:
                self.ssp_alpha_reg_bounds = [reg_alpha_region[0], reg_alpha_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_reg)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_reg)
                
                self.ssp_alpha_reg_bounds = [np.nan, np.nan]
            
        if hasattr(self, 'irreg_chi'):
            
            err_msg_sch = ("\nError in Schiavon SSP error estimation. \n"
                           "Assume results unreliable. \n\n")
            
            irreg_age_slice = np.nanmin(self.irreg_chi, axis = (1,2))
            irreg_age_region = self.irreg_age_interp[np.where(irreg_age_slice <=
                                                          (self.irreg_chi[self.irreg_chi_idx] + 1))]
            try:
                self.ssp_age_irreg_bounds = [irreg_age_region[0], irreg_age_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_sch)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_sch)
                
                self.ssp_age_irreg_bounds = [np.nan, np.nan]
                
            irreg_Z_slice = np.nanmin(self.irreg_chi, axis = (0,2))
            irreg_Z_region = self.irreg_Z_interp[np.where(irreg_Z_slice <= 
                                                      (self.irreg_chi[self.irreg_chi_idx] + 1))]
            try:
                self.ssp_Z_irreg_bounds = [irreg_Z_region[0], irreg_Z_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_reg)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_sch)
                
                self.ssp_z_irreg_bounds = [np.nan, np.nan]
            
            irreg_alpha_slice = np.nanmin(self.irreg_chi, axis = (0,1))
            irreg_alpha_region = self.irreg_alpha_interp[np.where(irreg_alpha_slice <= 
                                                          (self.irreg_chi[self.irreg_chi_idx] + 1))]
            try:
                self.ssp_alpha_irreg_bounds = [irreg_alpha_region[0], irreg_alpha_region[-1]]
                
            except:                    
                with self.err_log_file.open("a") as f:
                    f.write(err_msg_reg)
                    traceback.print_exc(file = f)
                logging.exception (err_msg_sch)
                
                self.ssp_alpha_irreg_bounds = [np.nan, np.nan]
                
    
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

# @numba.jit
def nansumwrapper(a, **kwargs):
    if np.isnan(a).all(axes=-1):
    # if bn.reduce.allnan(a, **kwargs):
        return np.nan
    else:
        return bn.nansum(a, **kwargs)