# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:21:24 2020

@author: peter
"""

import traceback, time, pathlib

import numpy as np

from astropy.table import Table

from stell_pops_idx.tools_stellar_pops import SSP_Params
from stell_pops_idx.tools_eq_width import Index_Measure, Spectrum_Cut
from stell_pops_idx.tools_spec_fit_class import Spectrum_Fit
from stell_pops_idx.tools_convolution import Convolutions


def run_all(
        output_directory,
        output_id,
        flux_spectrum,
        variance_spectrum,
        wavelength_spectrum,
<<<<<<< HEAD
        template_directory,
=======
>>>>>>> ce01334... New
        flux_array_templates,
        wavelength_array_templates,
        redshift_estimate,
        index_definitions,
        reg_SSP_models=None,
        irreg_SSP_models=None,
        log_time=True,
        save_intermediate_files=True
<<<<<<< HEAD
    ):   
    
    # out_dir = pathlib.Path(output_directory)
    out_subdir = pathlib.Path(output_directory).joinpath("{}".format(output_id))
    temp_dir = pathlib.Path(template_directory)
    
    # print (out_subdir.exists())
    out_subdir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    err_log_file = out_subdir.joinpath("Logfile_"+time.strftime("%Y_%m_%dT%H%M%SZ")+".log")
=======
        ):   
    
    out_dir = pathlib.Path(output_directory)
    print (out_dir.is_dir())
    
    start_time = time.time()
    err_log_file = out_dir.joinpath("errors.log")
>>>>>>> ce01334... New
    init_msg = "{} - analysis started.\n\n".format(time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    with err_log_file.open("w") as f:
            f.write(init_msg)
    
    #### if error - save id to file in errors folder
    
    
    
    try:
        
        SF = Spectrum_Fit(
            flux_spectrum,
            variance_spectrum,
            wavelength_spectrum,
            flux_array_templates,
            wavelength_array_templates,
            redshift_estimate,
            log_file_path=err_log_file,
            time_output=False,
            )
        
        if save_intermediate_files:
            log_rebinned_data = Table()
            log_rebinned_data["flux_replaced_pixels"] = SF.results["flux_replaced_pixels"]
            log_rebinned_data["flux_err"] = SF.results["flux_err"]
            log_rebinned_data["wavelength"] = SF.results["wavelength"]
            log_rebinned_data["good_pixels"] = SF.results["good_pixel_mask"]
            
<<<<<<< HEAD
            lrd_table_path = out_subdir.joinpath("{}_log_rebinned_data.fits".format(output_id))
=======
            lrd_table_path = out_dir.joinpath("{}_log_rebinned_data.fits".format(output_id))
>>>>>>> ce01334... New
            log_rebinned_data.write(lrd_table_path, overwrite = True)            
            
        c = 299792.458
        
        z_new = redshift_estimate + np.sqrt(
            (c+SF.results["velocity"])/(c-SF.results["velocity"])) - 1
        
        # Shift the wavelength range to match
        lam_new = np.exp(SF.results["wavelength"])/(1+np.sqrt(
            (c+SF.results["velocity"])/(c-SF.results["velocity"])) - 1)
        
    except:
        err_msg = "{:.2f} - Failure in spectral fitting module.\n\n".format(
            time.time() - start_time)
<<<<<<< HEAD
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
=======
        with open(err_log_file, 'a') as logfile:
            logfile.write(err_msg)
            traceback.print_exc(file = logfile)
>>>>>>> ce01334... New
        raise Exception(err_msg)
        
    try:
        
        SC = Spectrum_Cut(
            SF.results["flux_replaced_pixels"],
            lam_new, 
            bands=index_definitions,
            spec_err = SF.results["flux_err"], 
            goodpixels=SF.results["good_pixel_mask"],
            )
        
        ### Convolve the galaxy spectrum to the Lick/IDS resolution
        conv_obj = Convolutions(
            SC.output["fluxes"],
            SC.output["wavelengths"], 
            SF.results["velocity"],
            SF.results["sigma"],
            z_new, 
            system='IDS',
            flux_err=SC.output["fluxes_err"],
            )
        
    except:
        err_msg = "{:.2f} - Failure in spectral cutting and convolution module.\n\n".format(
            time.time() - start_time)
<<<<<<< HEAD
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
=======
        with open(err_log_file, 'a') as logfile:
            logfile.write(err_msg)
            traceback.print_exc(file = logfile)
>>>>>>> ce01334... New
        raise Exception(err_msg)
        
    try:
        ### Measure the equivalent widths of the indices
        IM = Index_Measure(
            conv_obj.output["fluxes_convolved"],
            SC.output["wavelengths"],
            conv_obj.output["fluxes_convolved_err"],
            bands=index_definitions,
            goodpixels=SC.output["good_pixels"],
            variance_weight=True,
            plot=False,
            )
        
    except:
        err_msg = "{:.2f} - Failed to measure line indices.\n\n".format(
            time.time() - start_time)
<<<<<<< HEAD
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
        raise Exception(err_msg)
        
    try:
        
        # template_dir = out_dir.joinpath("models_used")
        
=======
        with open(err_log_file, 'a') as logfile:
            logfile.write(err_msg)
            traceback.print_exc(file = logfile)
        raise Exception(err_msg)
        
    try:
>>>>>>> ce01334... New
        ### Determine the SSP parameters        
        SSP = SSP_Params(
            IM.results["equivalent_widths"],
            IM.results["equivalent_widths_err"],
<<<<<<< HEAD
            temp_dir,
            err_log_file,
            sig_flags=conv_obj.output["convolution_flags"],
            sigma=SF.results["sigma"],
            tmj_mods = reg_SSP_models, 
            hi_res = True, 
=======
            sig_flags=conv_obj.output["convolution_flags"],
            sigma=SF.results["sigma"],
            tmj_mods = reg_SSP_models, 
            hi_res = False, 
>>>>>>> ce01334... New
            )
        
    except:
        err_msg = "{:.2f} - Failed to determine optimal SSP parameters.\n\n".format(
            time.time() - start_time)
<<<<<<< HEAD
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
=======
        with open(err_log_file, 'a') as logfile:
            logfile.write(err_msg)
            traceback.print_exc(file = logfile)
>>>>>>> ce01334... New
        raise Exception(err_msg)
        
    try:
        
        results_table = Table()
        results_table["CUBEID"] = [output_id]
        results_table["z_in"] = redshift_estimate
        results_table["z_out"] = z_new
        results_table["velocity"] = SF.results["velocity"]
        results_table["velocity_err"] = SF.results["velocity_err"]
        results_table["sigma"] = SF.results["sigma"]
        results_table["sigma_err"] = SF.results["sigma_err"]
        results_table["S/N"] = SF.results["S/N"]
        
        for name in index_definitions['Name']:
            results_table["eq_width_"+name] = IM.results["equivalent_widths"][name]
            results_table["eq_width_"+name+'_err'] = IM.results["equivalent_widths_err"][name]
            results_table[name+"_sig_flag"] = conv_obj.output["convolution_flags"][name]
            try:
                results_table[name+"_corrected"] = SSP.corr_widths[name]
            except:
                results_table[name+"_corrected"] = 0.
                
        results_table["SSP_chi_min"] = SSP.tmj_chi_min
        results_table["SSP_age"] = SSP.ssp_age_tmj
        results_table["SSP_age_err_low"] = SSP.ssp_age_tmj_bounds[0]
        results_table["SSP_age_err_high"] = SSP.ssp_age_tmj_bounds[1]
        results_table["SSP_Z"] = SSP.ssp_Z_tmj
        results_table["SSP_Z_err_low"] = SSP.ssp_Z_tmj_bounds[0]
        results_table["SSP_Z_err_high"] = SSP.ssp_Z_tmj_bounds[1]
        results_table["SSP_alpha"] = SSP.ssp_alpha_tmj
        results_table["SSP_alpha_err_low"] = SSP.ssp_alpha_tmj_bounds[0]
        results_table["SSP_alpha_err_high"] = SSP.ssp_alpha_tmj_bounds[1]
            
        # out_table_path = out_dir.joinpath("{}_output.fits".format(output_id))
<<<<<<< HEAD
        out_table_path = out_subdir.joinpath("output.fits")
        results_table.write(
            out_table_path, 
            overwrite=True,
            )
        final_msg = "{:.2f} - Analysis finished. Results written to disk.\n\n".format(
            time.time() - start_time)
        with err_log_file.open("a") as f:
            f.write(final_msg)
        
    except:
        err_msg = "{:.2f} - Failed to write results to disk.\n\n".format(
            time.time() - start_time)
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
        raise Exception(err_msg)
        

def run_all_previous_results(
        previous_directory,
        output_directory,
        output_id,
        # flux_spectrum,
        # variance_spectrum,
        # wavelength_spectrum,
        template_directory,
        # flux_array_templates,
        # wavelength_array_templates,
        redshift_estimate,
        index_definitions,
        reg_SSP_models=None,
        irreg_SSP_models=None,
        log_time=True,
        ):   
    
    # out_dir = pathlib.Path(output_directory)
    out_subdir = pathlib.Path(output_directory).joinpath("{}".format(output_id))
    
    # in_dir = pathlib.Path(previous_directory)
    in_subdir = pathlib.Path(previous_directory).joinpath("{}".format(output_id))
    
    temp_dir = pathlib.Path(template_directory)
    
    # print (out_subdir.exists())
    out_subdir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    err_log_file = out_subdir.joinpath("Logfile_"+time.strftime("%Y_%m_%dT%H%M%SZ")+".log")
    init_msg = "{} - analysis started.\n\n".format(time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    with err_log_file.open("w") as f:
            f.write(init_msg)
    
    #### if error - save id to file in errors folder
    
    
    
    try:
        
        lrd_table_path = in_subdir.joinpath("{}_log_rebinned_data.fits".format(output_id))
        
        in_table_path = in_subdir.joinpath("output.fits")
        
        log_rebinned_data = Table.read(lrd_table_path, format="fits")     
          
        in_table = Table.read(in_table_path, format="fits")
            
        c = 299792.458
        
        z_new = redshift_estimate + np.sqrt(
            (c+in_table["velocity"])/(c-in_table["velocity"])) - 1
        
        # Shift the wavelength range to match
        lam_new = np.exp(log_rebinned_data["wavelength"])/(1+np.sqrt(
            (c+in_table["velocity"])/(c-in_table["velocity"])) - 1)
        
    except:
        err_msg = "{:.2f} - Failed to read previous data.\n\n".format(
            time.time() - start_time)
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
        raise Exception(err_msg)
        
    try:
        
        SC = Spectrum_Cut(
            log_rebinned_data["flux_replaced_pixels"],
            lam_new, 
            bands=index_definitions,
            spec_err = log_rebinned_data["flux_err"], 
            goodpixels=log_rebinned_data["good_pixels"],
            )
        
        ### Convolve the galaxy spectrum to the Lick/IDS resolution
        conv_obj = Convolutions(
            SC.output["fluxes"],
            SC.output["wavelengths"], 
            in_table["velocity"],
            in_table["sigma"],
            z_new, 
            system='IDS',
            flux_err=SC.output["fluxes_err"],
            )
        
    except:
        err_msg = "{:.2f} - Failure in spectral cutting and convolution module.\n\n".format(
            time.time() - start_time)
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
        raise Exception(err_msg)
        
    try:
        ### Measure the equivalent widths of the indices
        IM = Index_Measure(
            conv_obj.output["fluxes_convolved"],
            SC.output["wavelengths"],
            conv_obj.output["fluxes_convolved_err"],
            bands=index_definitions,
            goodpixels=SC.output["good_pixels"],
            variance_weight=True,
            plot=False,
            )
        
    except:
        err_msg = "{:.2f} - Failed to measure line indices.\n\n".format(
            time.time() - start_time)
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
        raise Exception(err_msg)
        
    try:
        
        # template_dir = out_dir.joinpath("models_used")
        
        ### Determine the SSP parameters        
        SSP = SSP_Params(
            IM.results["equivalent_widths"],
            IM.results["equivalent_widths_err"],
            temp_dir,
            err_log_file,
            sig_flags=conv_obj.output["convolution_flags"],
            sigma=in_table["sigma"],
            tmj_mods = reg_SSP_models, 
            hi_res = True, 
            )
        
    except:
        err_msg = "{:.2f} - Failed to determine optimal SSP parameters.\n\n".format(
            time.time() - start_time)
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
        raise Exception(err_msg)
        
    try:
        
        results_table = Table()
        results_table["CUBEID"] = [output_id]
        results_table["z_in"] = redshift_estimate
        results_table["z_out"] = z_new
        results_table["velocity"] = in_table["velocity"]
        results_table["velocity_err"] = in_table["velocity_err"]
        results_table["sigma"] = in_table["sigma"]
        results_table["sigma_err"] = in_table["sigma_err"]
        results_table["S/N"] = in_table["S/N"]
        
        for name in index_definitions['Name']:
            results_table["eq_width_"+name] = IM.results["equivalent_widths"][name]
            results_table["eq_width_"+name+'_err'] = IM.results["equivalent_widths_err"][name]
            results_table[name+"_sig_flag"] = conv_obj.output["convolution_flags"][name]
            try:
                results_table[name+"_corrected"] = SSP.corr_widths[name]
            except:
                results_table[name+"_corrected"] = 0.
                
        results_table["SSP_chi_min"] = SSP.tmj_chi_min
        results_table["SSP_age"] = SSP.ssp_age_tmj
        results_table["SSP_age_err_low"] = SSP.ssp_age_tmj_bounds[0]
        results_table["SSP_age_err_high"] = SSP.ssp_age_tmj_bounds[1]
        results_table["SSP_Z"] = SSP.ssp_Z_tmj
        results_table["SSP_Z_err_low"] = SSP.ssp_Z_tmj_bounds[0]
        results_table["SSP_Z_err_high"] = SSP.ssp_Z_tmj_bounds[1]
        results_table["SSP_alpha"] = SSP.ssp_alpha_tmj
        results_table["SSP_alpha_err_low"] = SSP.ssp_alpha_tmj_bounds[0]
        results_table["SSP_alpha_err_high"] = SSP.ssp_alpha_tmj_bounds[1]
            
        # out_table_path = out_dir.joinpath("{}_output.fits".format(output_id))
        out_table_path = out_subdir.joinpath("output.fits".format(output_id))
=======
        out_table_path = out_dir.joinpath("0_output.fits")
>>>>>>> ce01334... New
        results_table.write(
            out_table_path, 
            overwrite=True,
            )
<<<<<<< HEAD
        final_msg = "{:.2f} - Analysis finished. Results written to disk.\n\n".format(
            time.time() - start_time)
        with err_log_file.open("a") as f:
            f.write(final_msg)
=======
>>>>>>> ce01334... New
        
    except:
        err_msg = "{:.2f} - Failed to write results to disk.\n\n".format(
            time.time() - start_time)
<<<<<<< HEAD
        with err_log_file.open("a") as f:
            f.write(err_msg)
            traceback.print_exc(file = f)
=======
        with open(err_log_file, 'a') as logfile:
            logfile.write(err_msg)
            traceback.print_exc(file = logfile)
>>>>>>> ce01334... New
        raise Exception(err_msg)