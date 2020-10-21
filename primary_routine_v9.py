# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:31:02 2020

@author: pcadmin
"""

import glob, logging, traceback, time
import numpy as np
from astropy.table import Table
import pickle

from tools_stellar_pops import SSP_Params
from tools_eq_width import Index_Measure, Spectrum_Cut
from tools_spec_fit_class import Spectrum_Fit, spec_prep
from tools_convolution import Convolutions


def main_routine(cat_data, lam_temp, flux_temp, LI, tmj_models, sch_models, directory):   
    
    catid = np.int64(cat_data[2])
    
    print (catid)
    
    start_time = time.strftime("%Y_%m_%dT%H%M%SZ")
    err_log_file = ("sample_data/{0}/logs/{1}_".format(directory, cat_data[4])+start_time+".log")
    
    # try:
        
    #     Table.read('sample_data/{0}/{1}.csv'.format(directory, catid), format = 'ascii.csv')
    #     return
    
    # except:
    #     pass
    
    try:
    
        
        
        z = cat_data[3]
        # r_e = cat_data["r_e"]
        # PA = cat_data["PA"]
        # ellip = cat_data["ellip"]
        
        blue_name = glob.glob("v0.12/aperture_spectra/{0}".format(cat_data[0]))
        blue = blue_name[0]
        
        red_name = glob.glob("v0.12/aperture_spectra/{0}".format(cat_data[1]))
        red = red_name[0]
        
        spec, var, wave = spec_prep(blue, red)
        
        SPC = Spectrum_Fit(spec, var, wave, flux_temp, lam_temp,
                            z, log_file_name=err_log_file,
                            time_output=False)
        
        gal_rep, residuals, lam, vel, sigma, SN, gpixels, vel_err, sig_err = SPC.output
        
        with open('sample_data/{0}/gals/{1}.pkl'.format(directory, cat_data[4]), 'wb') as outfile:
            pickle.dump(gal_rep, outfile)       
            
        with open('sample_data/{0}/lams/{1}.pkl'.format(directory, cat_data[4]), 'wb') as outfile:
            pickle.dump(lam, outfile)
            
        with open('sample_data/{0}/res/{1}.pkl'.format(directory, cat_data[4]), 'wb') as outfile:
            pickle.dump(residuals, outfile)
            
        with open('sample_data/{0}/gpix/{1}.pkl'.format(directory, cat_data[4]), 'wb') as outfile:
            pickle.dump(gpixels, outfile)
            
        # directory_prev = "glamdring_v10"
        
        # prev_data = Table.read('sample_data/{0}/{1}.csv'.format(directory_prev, cat_data[4]), 
        #                         format = 'ascii.csv')
        
        # vel = prev_data["vel_measure"]
        # vel_err = prev_data["vel_measure_err"]
        # sigma = prev_data["sig_measure"]
        # sig_err = prev_data["sig_measure_err"]
        
        # SN = prev_data["SN"]
        
        # with open('sample_data/{0}/gals/{1}.pkl'.format(directory_prev, cat_data[4]),
        #           'rb') as outfile:
        #     gal_rep = pickle.load(outfile)       
            
        # with open('sample_data/{0}/lams/{1}.pkl'.format(directory_prev, cat_data[4]), 
        #           'rb') as outfile:
        #     lam = pickle.load(outfile)
            
        # with open('sample_data/{0}/res/{1}.pkl'.format(directory_prev, cat_data[4]),
        #           'rb') as outfile:
        #     residuals = pickle.load(outfile)
            
        # with open('sample_data/{0}/gpix/{1}.pkl'.format(directory_prev, cat_data[4]),
        #           'rb') as outfile:
        #     gpixels = pickle.load(outfile)
            
            
        c = 299792.458
        
        z_new = z + np.sqrt((c+vel)/(c-vel)) - 1
        
        # Shift the wavelength range to match
        lam_new = np.exp(lam)/(1+np.sqrt((c+vel)/(c-vel)) - 1)
        
        
        ### Here is where my new code needs to go
        ### 1) Create dict of spectral slices
        ### 2) Feed each slice through largely existing code
        
        ### 3 dicts:
        ### 1) Spectrum
        ### 2) Wavelengths
        ### 3) Convolution flag
        
        ### Cut out +/- 20 pixels around
        
        ### IMPORTANT - need to implement checking for skyline replacement
        
        ### Rewrite Convolutions so it takes dict inputs for specs, lams, res
        ### Requires bands as input (or None)
        ### Then needs to return dict outputs
        
        SC = Spectrum_Cut(gal_rep, lam_new, bands = LI, spec_err = residuals, goodpixels=gpixels)
        specs, lams, res, gpix = SC.results
        
        ### Convolve the galaxy spectrum to the Lick/IDS resolution
        conv_obj = Convolutions(lams, specs, vel, sigma, z_new, 
                                system='IDS', flux_err=res)
        specs_conv, res_conv, flags = conv_obj.output
        
        # plt.figure(figsize = (20,10))
        # for name in LI["Name"]:
        #     plt.plot(lams[name], specs_conv[name])
        # plt.show()
        
        # plt.figure(figsize = (20,10))
        # for name in LI["Name"]:
        #     plt.plot(lams[name], specs[name])
        # plt.show()
        
        ### Measure the equivalent widths of the indices
        IM = Index_Measure(lams, specs_conv, res_conv, bands=LI, goodpixels=gpix,
                           variance_weight=True,
                           plot=False)
        equivalent_widths, width_errs = IM.results
        
        ### Determine the SSP parameters
        # SSP = SSP_Params(equivalent_widths, width_errs, sig_flags = flags, sigma = sigma, 
        #                  tmj_mods = tmj_models, sch_mods = sch_models, hi_res = True)
        
        SSP = SSP_Params(equivalent_widths, width_errs, sig_flags = flags, sigma = sigma, 
                         tmj_mods = tmj_models, hi_res = True, )
                         # tmj_names=["Hdelta_A", "Hdelta_F", "CN_1", "CN_2",
                         #          "Ca4227", "G4300", "Hgamma_A", "Hgamma_F",
                         #          "Fe4383", "Ca4455", "Fe4531", "Fe4668",
                         #          "H_beta", "Fe5015", "Mg_2", 
                         #          "Fe5270", "Fe5335", "Fe5406"])
        
        # !! Convolution of error spectrum
        # !! Weighting of fitting - Cenarro 2001
        
        
        t = Table(names = ('catid', 'z_in', 'z_new', 'vel_measure', "vel_measure_err",
                           'sig_measure', "sig_measure_err","SN", 
                           "tmj_chi_min"),#, "sch_chi_min"), 
                  dtype=('int', 'f8','f8','f8','f8','f8','f8','f8','f8'))#,'f8'))
        
        t.add_row((0,0.,0.,0.,0.,0.,0.,0., 0.))#, 0.))
        
        for name in LI['Name']:
            t["measured_"+name] = 0.
            t["measured_"+name+'_err'] = 0.
            t[name+"_sig_flag"] = 0.
            t["corr_"+name] = 0.
        
        t['measured_age_tmj'] = 0.
        t['measured_age_tmj_err_low'] = 0.
        t['measured_age_tmj_err_high'] = 0.
        t['measured_Z_tmj'] = 0.
        t['measured_Z_tmj_err_low'] = 0.
        t['measured_Z_tmj_err_high'] = 0.
        t['measured_alpha_tmj'] = 0.
        t['measured_alpha_tmj_err_low'] = 0.
        t['measured_alpha_tmj_err_high'] = 0.
        # t['measured_age_sch'] = 0.
        # t['measured_age_sch_err_low'] = 0.
        # t['measured_age_sch_err_high'] = 0.
        # t['measured_Z_sch'] = 0.
        # t['measured_Z_sch_err_low'] = 0.
        # t['measured_Z_sch_err_high'] = 0.
        # t['measured_alpha_sch'] = 0.
        # t['measured_alpha_sch_err_low'] = 0.
        # t['measured_alpha_sch_err_high'] = 0.
        
        tab_row = [catid, z, z_new, vel, vel_err, sigma, sig_err, SN, SSP.tmj_chi_min]#, SSP.sch_chi_min]
        
        for name in LI["Name"]:
            tab_row.append(equivalent_widths[name])
            # print (equivalent_widths[name])
            tab_row.append(width_errs[name])
            tab_row.append(flags[name])
            try:
                tab_row.append(SSP.corr_widths[name])
                # print ((SSP.corr_widths[name]))
            except:
                tab_row.append(0.)
            
        
        tab_row.extend([SSP.ssp_age_tmj, SSP.ssp_age_tmj_bounds[0], SSP.ssp_age_tmj_bounds[1],
        SSP.ssp_Z_tmj, SSP.ssp_Z_tmj_bounds[0], SSP.ssp_Z_tmj_bounds[1],
        SSP.ssp_alpha_tmj, SSP.ssp_alpha_tmj_bounds[0], SSP.ssp_alpha_tmj_bounds[1]])#,
        # SSP.ssp_age_sch, SSP.ssp_age_sch_bounds[0], SSP.ssp_age_sch_bounds[1],
        # SSP.ssp_Z_sch, SSP.ssp_Z_sch_bounds[0], SSP.ssp_Z_sch_bounds[1],
        # SSP.ssp_alpha_sch, SSP.ssp_alpha_sch_bounds[0], SSP.ssp_alpha_sch_bounds[1]])
        
        t.add_row((tab_row))
        tab = t[1:]
        
        # try:
        
        #     Table.read('sample_data/{0}/{1}.csv'.format(directory, catid), format = 'ascii.csv')
            
        # except:
            
        #     tab.write('sample_data/{0}/{1}.csv'.format(directory, catid), 
        #               format = 'ascii.csv', overwrite = True)
            
        # else:
            
        tab.write('sample_data/{0}/{1}.csv'.format(directory, cat_data[4]), 
                  format = 'ascii.csv', overwrite = True)
            
    except:
        err_msg = "Something went wrong here. Maybe run individually later?\n"
        with open(err_log_file, 'a') as logfile:
            logfile.write(err_msg)
            traceback.print_exc(file = logfile)
        logging.exception (err_msg)