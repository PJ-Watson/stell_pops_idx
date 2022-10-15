# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:22:21 2020

@author: peter
"""

###################################################################################################

import numpy as np
import bottleneck as bn

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util

from astropy.convolution import convolve_fft,Gaussian1DKernel
from astropy.stats import sigma_clip, biweight_location

from itertools import groupby

from operator import itemgetter

from scipy import ndimage

import time

import copy

import random

##################################################################################################

class Spectrum_Fit():
    
    def __init__(self, 
             gal_spec, 
             gal_var,
             gal_lam,
             temp_array_spec,
             temp_array_lam,
             z, 
             FWHM_temp=2.50, 
             use_residuals=False,
             FWHM_gal_spec=2.6554,
             p_order=12,
             log_file_path=None, 
             time_output=False,
             run_auto=True
             ):
        
        
        start_time = time.strftime("%Y_%m_%dT%H%M%SZ")
        if log_file_path==None:
            self.err_log_file = "log_files/SSP_error_log_"+start_time+".log"
        else:
            self.err_log_file = log_file_path

        self.gal_spec = gal_spec
        self.gal_var = gal_var
        self.gal_lam = gal_lam
        self.temp_array_spec = temp_array_spec
        self.temp_array_lam = temp_array_lam
       
        self.z = z
        self.c = 299792.458 # speed of light in km/s
        
        self.FWHM_temp = FWHM_temp
        self.FWHM_gal_spec = FWHM_gal_spec
        self.p_order = p_order
        
        self.use_residuals = use_residuals
        self.time_output = time_output
        self.t = time.time()
        
        if run_auto == True:
            
            try:
                self.spec_prepare()
            except:
                err_msg_prep = "Failure in preparing spectrum for ppxf fit."
                raise Exception(err_msg_prep)
            
            # (km/s), starting guess for [V,sigma]
            self.init_guess = [0.0, 100.0] 
            
            try:
                self.initial_fit()
            except:
                err_msg_first = "Failure in first ppxf fit."
                raise Exception(err_msg_first)
                
            try:
                self.second_fit()
            except:
                err_msg_second = "Failure in second ppxf fit."
                raise Exception(err_msg_second)
                
            try:
                self.third_fit_mc_errors()
            except:
                err_msg_third = "Failure in third ppxf fit."
                raise Exception(err_msg_third)
                
            
###################################################################################################            
            
    def print_time(self, position):
        
        current_time = time.time() - self.t
        
        print ("")
        print ("Time at position {0}: {1}".format(position, current_time))
        print ("")
        
        return
  
###################################################################################################
    
    def find_nearest(arr, val):
        """ Find the pixel index in an array with the values closest to some chosen value.
    
        INPUTS:
        arr - the array you wish to search
        val - the value whose closest index you wish to find in arr
    
        OUTPUTS:
        idx - the index of arr where the value is closest to val
    
        """
        
        # Finds the index of the array element in arr nearest to val
        idx=(np.abs(arr-val)).argmin()
        return idx

###################################################################################################
    
    def determine_goodpixels(self, lam, spec, var, 
                             skyline=False, mask_lines=False):
    
        goodpixel_mask = np.zeros_like(lam)
        goodpixel_mask[np.where(spec == 0.0)] = 1.0
        goodpixel_mask[np.where(var <= 0.0)] = 1.0
        goodpixel_mask[np.where(np.isfinite(var) == False)] = 1.0
        goodpixel_mask[np.where(np.isfinite(spec) == False)] = 1.0
        goodpixel_mask[-5:-1] = 1.0
        
        if skyline:
            goodpixel_mask[np.where((np.exp(lam) >= (5572./(1+self.z))) & 
                                    (np.exp(lam) <= (5582./(1+self.z))) )] = 1
        
        if mask_lines:
            # Edited by JvdS. I am in favour of always masking these lines.
            #                 -----[OII]-----    Hdelta   Hgamma   Hbeta    
            lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33,
            #                  -----[OIII]-----   [OI]    -----[NII]----- 
                              4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 
            #                  Halpha   -----[SII]-----  
                              6562.80, 6716.47, 6730.85])
            
            for line in lines:
                idx_line = self.find_nearest(np.exp(lam), line)
        
                goodpixel_mask[idx_line-8:idx_line+9]=1.0
        
        goodpixels = np.where(goodpixel_mask==0.0)
        
        badpixels_pp2 = [i for i in range(len(lam)) if i not in np.squeeze(goodpixels)]
        new_pix_all = []
        for k, g in groupby(enumerate(badpixels_pp2), lambda ix : ix[0] - ix[1]):
            tmp = np.fromiter(map(itemgetter(1), g), dtype = np.int)
            if len(tmp) == 1:
                new_pix = np.array([tmp[0] - 1, tmp[0], tmp[0] + 1])
                new_pix_all = np.concatenate((new_pix_all, new_pix))
            else:
                range_increase = int(np.ceil(len(tmp) * 1.1))
                new_pix = np.arange(tmp[0], tmp[0] + range_increase)
                
                shift = int((len(tmp) - range_increase) / 2)
                
                new_pix += shift
                new_pix_all = np.concatenate((new_pix_all, new_pix))
                
        goodpix_expand = np.asarray([i for i in range(len(lam)) 
                                          if i not in new_pix_all])
        
        goodpixels = np.squeeze(goodpix_expand)

        return goodpixels
    
###################################################################################################    
            
    def spec_prepare(self):
        
        if self.time_output:
            self.print_time(0)
        
        ### Shift the galaxy wavelengths to the rest frame
        gal_lam_rest = self.gal_lam/(1.0+self.z) 

        ### Adjust the FWHM as well
        FWHM_gal = self.FWHM_gal_spec/(1.0+self.z)

        gal_lam_range = np.array([gal_lam_rest[0], gal_lam_rest[-1]])
        
        ### First rebin to get the velocity scale only
        g_s_l, g_l_l, self.velscale = util.log_rebin(gal_lam_range, self.gal_spec)
        
        ### Rebin the data using velscale
        self.log_noise_spec, log_noise_lam, vs = util.log_rebin(gal_lam_range,
                                                                np.sqrt(self.gal_var), 
                                                                velscale=self.velscale)
        self.log_gal_spec, self.log_gal_lam, vs = util.log_rebin(gal_lam_range,
                                                                 self.gal_spec,
                                                                 velscale=self.velscale)
        
        # Smooth the templates to data resolution, and log rebin them.
        log_temp_lam, self.template_array = prepare_templates(
            self.temp_array_lam, 
            self.temp_array_spec, 
            self.FWHM_temp,
            FWHM_gal, 
            self.velscale,
            log_file_name=self.err_log_file,
        )

        # Now make a guess at the systemic velocity of the galaxy (km/s)
        self.dv = (log_temp_lam[0] - self.log_gal_lam[0])*self.c
        
        if self.time_output:
            self.print_time(1)

        # Use sami ppxf utils to determine the goodpixels - masking the emission lines
        self.goodpixels = self.determine_goodpixels(self.log_gal_lam, self.log_gal_spec, 
                                                            self.log_noise_spec)

        # Then mask the arrays
        self.log_gal_spec[np.where(np.isfinite(self.log_gal_spec) == False)] = 0.0
        self.log_noise_spec[np.where(np.isfinite(self.log_noise_spec) == False)] = bn.nanmedian(
                                                            self.log_noise_spec[self.goodpixels])

        # If not enough goodpixels, skip this galaxy.
        if np.shape(self.goodpixels)[0] <= 1.0:
            err_msg_init = "Not enough usable pixels in spectrum."
            raise Exception(err_msg_init)
            return
            
        else:
            return
        
###################################################################################################
        
    def noise_clip(self):
                
        if self.use_residuals == False:
            
            noise_med = bn.nanmedian(self.log_noise_spec[self.goodpixels])
            
            noise_copy = copy.deepcopy(self.log_noise_spec)
            noise_masked = sigma_clip(noise_copy)
            noise_new = noise_masked.filled(bn.nanmax(noise_masked[~noise_masked.mask]))
            
        else:
            noise_new = np.abs(self.residuals)

            badpixels = [i for i in range(len(self.log_gal_spec)) 
                         if i not in self.goodpixels]
            noise_copy = np.copy(noise_new[self.goodpixels])
            noise_med = bn.nanmedian(noise_copy)
            noise_new[badpixels] = noise_med
            noise_clip = sigma_clip(noise_copy)
            
            res_sigma = 5.
            gauss = Gaussian1DKernel(res_sigma)
            noise_smooth = convolve_fft(noise_clip, gauss)
            noise_new[self.goodpixels] = noise_smooth
        
        noise_new[np.where(noise_new <= 0)] = np.abs(noise_med)
        
        gp_cut = self.goodpixels[np.where(self.goodpixels>500)]
        
        noise_orig = biweight_location(noise_new[gp_cut])
        noise_est = robust_sigma(self.residuals[gp_cut])  #FIX THIS
        noise_new = noise_new*(noise_est/noise_orig)
        
        return noise_new
        
        
###################################################################################################
        
    def initial_fit(self):
        
        if self.time_output:
            self.print_time(2)
            
        # Make a fake noise array, full of 1s.
        fake_noise = np.full_like(self.log_gal_spec, 1.0)
        
        # 1) Run ppxf with fake noise - to get optimal template and noise es
        pp = ppxf(self.template_array, self.log_gal_spec, 
                  fake_noise, self.velscale, self.init_guess, 
                  goodpixels=self.goodpixels, plot=False, moments=2, 
                  degree=self.p_order, quiet=True, vsyst=self.dv)
        
        # Update the goodpixels array
        self.goodpixels = pp.goodpixels
        
        self.residuals = pp.galaxy - pp.bestfit
        
        self.noise_new = self.noise_clip()
        
        return
        
###################################################################################################
        
    def second_fit(self):
        
        if self.time_output:
            self.print_time(3)

        # 2) Run ppxf with better noise using clean to get goodpixels
        pp = ppxf(self.template_array, self.log_gal_spec, 
                  self.noise_new, self.velscale, self.init_guess, 
                  clean=True, goodpixels=self.goodpixels, plot=False,
                  moments=2, degree=self.p_order, quiet=True, vsyst=self.dv)
        
        # Update the goodpixels array again
        self.goodpixels = pp.goodpixels
        
        self.residuals = pp.galaxy - pp.bestfit
        
        self.noise_new = self.noise_clip()
        
        self.gp_mask = np.zeros_like(pp.galaxy, dtype = int)
        self.gp_mask[self.goodpixels] = 1
        

        badpixels_pp2 = [i for i in range(len(self.log_gal_spec)) if i not in self.goodpixels]
        new_pix_all = []
        for k, g in groupby(enumerate(badpixels_pp2), lambda ix : ix[0] - ix[1]):
            tmp = np.fromiter(map(itemgetter(1), g), dtype = np.int)
            if 2450 in tmp and 2750 in tmp:
                new_pix_all = np.concatenate((new_pix_all, tmp))
            elif len(tmp) == 1:
                new_pix = np.array([tmp[0] - 1, tmp[0], tmp[0] + 1])
                new_pix_all = np.concatenate((new_pix_all, new_pix))
            else:
                range_increase = int(np.ceil(len(tmp) * 1.25))
                new_pix = np.arange(tmp[0], tmp[0] + range_increase)
                
                shift = int((len(tmp) - range_increase) / 2)
                
                new_pix += shift
                new_pix_all = np.concatenate((new_pix_all, new_pix))
                
        self.goodpix_expand = np.asarray([i for i in range(len(self.log_gal_spec)) 
                                          if i not in new_pix_all])
        
        return
        
###################################################################################################
        
    def third_fit_mc_errors(self):
        
        if self.time_output:
            self.print_time(4)
            
        # 3) Run ppxf with even better noise estimate, and the correct bias, 
        # proper goodpixels, no clean
        pp = ppxf(self.template_array, self.log_gal_spec, 
                  self.noise_new, self.velscale, self.init_guess, 
                  goodpixels=self.goodpix_expand, plot=False, moments=2, 
                  quiet=True, degree=self.p_order, vsyst=self.dv)
        
        self.goodpixels = pp.goodpixels
        
        self.residuals = pp.galaxy - pp.bestfit
        
        self.gal_replaced = np.copy(self.log_gal_spec)
        badpixels = [i for i in range(len(self.log_gal_spec)) 
                     if i not in self.goodpixels]
        self.gal_replaced[badpixels] = pp.bestfit[badpixels]
        
        
        # pix_res = np.abs(self.log_gal_spec - pp.bestfit)
        
        # # res_std = bn.nanmedian(pix_res[pp.goodpixels])
        # res_med = bn.nanmedian(pix_res[pp.goodpixels])
        # # res_max = bn.nanmax(pix_res[pp.goodpixels])
        # pix_res[badpixels] = res_med
        
        
        
        
        # SN_ap = bn.nanmedian(pp.galaxy[pp.goodpixels]/noise_est)
        
        # return gal_replaced, pix_res, self.log_gal_lam, pp.sol[0], pp.sol[1], SN_ap, gp_mask
        
        # if residuals:
        #     pix_res = np.abs(self.log_gal_spec - pp.bestfit)
            
        #     res_std = bn.nanmedian(pix_res[pp.goodpixels])
        #     pix_res[badpixels] = 3*res_std
            
        #     SN_ap = bn.nanmedian(pp.galaxy[pp.goodpixels]/noise_est)
        
        #     return gal_replaced, pix_res, self.log_gal_lam, pp.sol[0], pp.sol[1], SN_ap, pp.goodpixels
    
        # else:
            
        #     # noise_std = bn.nanstd(noise_new_aperture[pp.goodpixels])
        #     # noise_new_aperture[badpixels] = 3*noise_std
        #     # noise_new_aperture[badpixels] *= 3
            
        #     # noise_new_aperture *= (noise_est/noise_orig)
            
        noise_median = bn.nanmedian(self.noise_new[self.goodpixels])
        
        self.noise_new[badpixels] = noise_median
        
        # noise_max = bn.nanmax(noise_new_aperture[pp.goodpixels])
        # noise_new_aperture[badpixels] = noise_max
        

        SN_ap = bn.nanmedian(pp.galaxy[np.where((np.exp(self.log_gal_lam) >= 4600) &
                                                (np.exp(self.log_gal_lam) <= 4800))]/
                             self.noise_new[np.where((np.exp(self.log_gal_lam) >= 4600) &
                                                (np.exp(self.log_gal_lam) <= 4800))])
        
        if self.time_output:
            self.print_time(5)
        
        temp_arr = np.sum(pp.weights*self.template_array, axis=1)
        
        n_repeats=100
        
        vs = np.zeros(n_repeats)
        sigs = np.copy(vs)
        
        # Do loop over repeats
        for n in range(n_repeats):
                                                
            #make new array for simulated spectrum
            gal_new = np.copy(pp.bestfit)
            noise_n = np.copy(self.noise_new)
            spec_len = np.shape(gal_new)[0]
            
            # make best-fit spectrum with the polynomial subtracted
            # removing the polynomial will make the simulated fit a lot faster!
            #
            tmp_x = np.linspace(-1, 1, spec_len)
            tmp_apoly = np.polynomial.legendre.legval(tmp_x, pp.polyweights)
            gal_new = gal_new - tmp_apoly
                
            # determine sectors in which we are going to reshuffle noise
            # We pick 8 sectors, which should be enough, to avoid noise and less noise getting mixed up
            # This number is a little arbritrary
            nsectors = 10
            pix_vals_min = np.arange(nsectors)*np.divide(spec_len,nsectors).astype(int)
            # print (np.int(pix_vals_min))
            pix_vals_max = np.arange(1,nsectors+1)*np.divide(spec_len,nsectors).astype(int)
            # print (pix_vals_max)
            pix_vals_max[-1] = spec_len+1
            
            
            #Go through 14 sectors and reshuffle noise
            for i in range(nsectors):
                ww = np.where((self.goodpixels >= pix_vals_min[i]) & 
                              (self.goodpixels < pix_vals_max[i]))
                # we need at least 10 spectrum pixels to shuffle noise
                if len(ww[0]) > 10:
                    #make array integer array for sorting
                    index_arr = np.linspace(0, len(ww[0])-1, len(ww[0]), dtype=np.int)
                    #do random shuffle of those indices
                    random.shuffle(index_arr)
                                                
                    #get the residual
                    resids_temp = self.residuals[self.goodpixels[ww]]
                    noise_temp = self.noise_new[self.goodpixels[ww]]
                    #reshuffle the resid_temp array with index_arr
                    gal_new[self.goodpixels[ww]]+=resids_temp[index_arr]
                    noise_n[self.goodpixels[ww]]=noise_temp[index_arr]
                    
                    
                                                
                else:
                    gal_new[pix_vals_min[i]:pix_vals_max[i]] = \
                        pp.galaxy[pix_vals_min[i]:pix_vals_max[i]]
                    noise_n[pix_vals_min[i]:pix_vals_max[i]] = \
                        self.noise_new[pix_vals_min[i]:pix_vals_max[i]]
        
            try:
                pp_temp=ppxf(temp_arr, gal_new, noise_n, self.velscale, self.init_guess,
                             quiet=True, goodpixels=self.goodpixels, plot=False,
                             moments=2, degree=3, vsyst=self.dv)
                
            except:
                err_msg_init = "Crashed trying to estimate errors."
                raise Exception(err_msg_init)
                                            
            vs[n] = pp_temp.sol[0]
            sigs[n] = pp_temp.sol[1]
        
        
        vel_err = robust_sigma(vs)
        sig_err = robust_sigma(sigs)
        
        if self.time_output:
            self.print_time(6)
            
        self.gp_mask_exp = np.zeros_like(pp.galaxy, dtype = int)
        self.gp_mask_exp[self.goodpixels] = 1
        
        # self.output = (self.gal_replaced, self.noise_new, self.log_gal_lam, 
        #         pp.sol[0], pp.sol[1], SN_ap, self.gp_mask, mverr, mserr)
        self.results = {
            "flux_replaced_pixels":self.gal_replaced, 
            "flux_err":self.noise_new, 
            "wavelength":self.log_gal_lam, 
            "good_pixel_mask":self.gp_mask_exp,
            "S/N":SN_ap,
            "velocity":pp.sol[0],
            "velocity_err":vel_err,
            "sigma":pp.sol[1],
            "sigma_err":sig_err,
            }
                # pp.sol[0], pp.sol[1], SN_ap, self.gp_mask_exp, vel_err, sig_err}
                
                
###################################################################################################

# DEF20150706 start.
def prepare_templates(lamtemplatearr, templatearr, FWHM_tem, FWHM_data, velscale,
                      log_file_name='template.log'):
    """ Read in any list of templates, convolve to the data FWHM, log rebin, and return the results in an array.


    INPUTS:
    lamtemplatearr - array containing wavelength range of spectra of size [length of template]
    templatearr - array containing template spectra of size [length of template, number of templates]

    FWHM_tem - the FWHM of the templates
    FWHM_data - the FWHM of the data you wish to fit with the templates

    log_file_name: float, optional
        name of the log file for the templates.

    OUTPUTS:
    template_array - an array of prepared (smoothed and rebinned) templates of size [length of log template, number of templates]

    """

    # log_buff is the buffer for log messages.
    # Lisa's favourite banner.
    banny = "-" * 62 + "\n"
    log_buff = banny
    log_buff += "Preparing the templates\n"

    n_templates=np.shape(templatearr)[1]

    log_buff += "You have supplied %5i templates\n"%(n_templates)
    log_buff += "The FWHM of the templates is %6.4f\n"%(FWHM_tem)
    log_buff += "The FWHM of the data is %6.4f\n"%(FWHM_data)
    # DEF20150706 end.

    # Isolate a single template.
    lambda_tem_lin=lamtemplatearr[:,0]
    tem_lin=templatearr[:,0]

    # DEF20150706 start.
    log_buff += "Checking the wavelength coverage\n"
    log_buff += "lambda_0,0=%7.8f;  lambda_0,N=%7.8f\n"%(lamtemplatearr[0,0], lamtemplatearr[-1,0])
    log_buff += "lambda_N,0=%7.8f;  lambda_N,N=%7.8f\n"%(lamtemplatearr[0,0], lamtemplatearr[-1,0])
    # DEF20150706 end.

    #py.plot(tem_single_tab['template_lambda'], tem_single_tab['template'])

    # Find the difference in resolution between the galaxy and the templates.
    FWHM_diff=np.sqrt(FWHM_data**2.0 - FWHM_tem**2.0)

    if np.isfinite(FWHM_diff):
        sigma_diff=FWHM_diff/2.355/(lambda_tem_lin[1]-lambda_tem_lin[0])
        
    else:
        sigma_diff=0.0001
        

    # DEF20150706 start.
    log_buff += "Sigma diff: %7.3f\n"%(sigma_diff)
    # DEF20150706 end.

    # Log rebin the one template
    lambda_range_tem=np.array([lambda_tem_lin[0], lambda_tem_lin[-1]])
    # DEF20150706 start.
    log_buff += "Lambda range template: [%7.3f; %7.3f]\n"%(lambda_tem_lin[0], lambda_tem_lin[-1])
    # DEF20150706 end.

    tem_log, lambda_tem_log, velscale = util.log_rebin(lambda_range_tem, tem_lin, velscale=velscale)

    # DEF20150706 start.
    log_buff += "\nLambda tem log: [%7.8f; %7.8f]\n"%(lambda_tem_log[0], lambda_tem_log[-1])
    log_buff += "Velscale is: %7.3f\n"%(velscale)
    # DEF20150706 end.

    # Make an array to hold the rebinned templates that is the correct size.
    template_array_final=np.empty((tem_log.size, n_templates))

    # DEF20150706 start.
    log_buff += "\nSmoothing and rebinning all templates.\n"
    # DEF20150706 end.

    # Now read in all of the templates, smooth them and put into the array of templates.
    for i in range(n_templates):

        # go through the arrays
        lambda_tem_lin=lamtemplatearr[:,i]
        tem_lin=templatearr[:,i]

        # Smooth the template with a Gaussian filter.
        tem_lin_smooth=ndimage.gaussian_filter1d(tem_lin, sigma_diff)

        # Log rebin the template
        tem_log, lambda_tem_log, velscale = util.log_rebin(lambda_range_tem, tem_lin_smooth, velscale=velscale)
        template_array_final[:,i]=tem_log/np.median(tem_log)

    
    # DEF20150706 start.
    log_buff += "\nFinished preparing templates.\n"
    log_buff += banny
    log_file = open(log_file_name, 'a')
    log_file.writelines(log_buff)
    log_file.close()
    # DEF20150706 end.

    return lambda_tem_log, template_array_final