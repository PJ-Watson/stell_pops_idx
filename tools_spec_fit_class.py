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

import time

import sami_ppxf_utils_pjw

# import matplotlib.pyplot as plt

import copy

import random

##################################################################################################

def spec_prep(inbluespec, inredspec):

    try:
        spec, var, hdr = sami_ppxf_utils_pjw.combine_blue_red_aperFITS(inbluespec, inredspec)
    except Exception as e:
        print ("Something's wrong.")
        raise e
        
    gal_name=hdr['NAME']



    # # --------------------------------------------------------------
    # Pull out the wavelength range of the galaxy
    x=np.arange(hdr['NAXIS1'])+1
    L0=hdr['CRVAL1']-hdr['CRPIX1']*hdr['CDELT1'] #Lc-pix*dL

    lambda_galaxy=L0+x*hdr['CDELT1']

    return spec, var, lambda_galaxy


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
             log_file_name=None, 
             time_output=False,
             run_auto=True
             ):
        
        
        start_time = time.strftime("%Y_%m_%dT%H%M%SZ")
        if log_file_name==None:
            self.err_log_file = "log_files/SSP_error_log_"+start_time+".log"
        else:
            self.err_log_file = log_file_name

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
            
            self.spec_prepare()
            
            # (km/s), starting guess for [V,sigma]
            self.init_guess = [0.0, 100.0] 
            
            self.initial_fit()
            self.second_fit()
            self.third_fit_mc_errors()
            
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
        
        goodpixels = np.squeeze(goodpixels)

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
        log_temp_lam, self.template_array = sami_ppxf_utils_pjw.prepare_templates(
            self.temp_array_lam, self.temp_array_spec, self.FWHM_temp, FWHM_gal, self.velscale)

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
            with open(self.err_log_file, 'a') as myfile:
                myfile.write(err_msg_init)
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
                with open(self.err_log_file, 'a') as myfile:
                    myfile.write(err_msg_init)
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
        self.output =  (self.gal_replaced, self.noise_new, self.log_gal_lam, 
                pp.sol[0], pp.sol[1], SN_ap, self.gp_mask_exp, vel_err, sig_err)