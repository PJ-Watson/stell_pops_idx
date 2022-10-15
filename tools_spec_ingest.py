# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:43:59 2020

@author: peter
"""

import numpy as np

import astropy.io.fits as pf

from scipy import ndimage
from scipy import interpolate

def binned_data_wrapper(inbluecube, inredcube, bin_ext="RE_BINS"):
    
    hdulist_blue=pf.open(inbluecube)
   
    hdr_blue=hdulist_blue['BINNED_FLUX_{}'.format(bin_ext)].header
    name1=hdr_blue['NAME']
    # hdr_blue2=hdulist_blue['BINNED_FLUX_SN25'].header
    
    # print ("yes")
    
    bin_mask = hdulist_blue['BIN_MASK_{}'.format(bin_ext)].data #.flatten()
    
    # print (np.nansum(bin_mask))
    
    
    ### NEED TO CHECK THAT BINS ARE COMBINED CORRECTLY!
    bin_nums, bin_indices = ndix_unique(bin_mask)
    
    # print (bin_nums)
    # print (bin_indices)
    
    # Pull out the wavelength range, BLUE
    x_blue=np.arange(hdr_blue['NAXIS3'])+1
    L0_blue=hdr_blue['CRVAL3']-hdr_blue['CRPIX3']*hdr_blue['CDELT3'] #Lc-pix*dL

    lambda_galaxy_blue=L0_blue+x_blue*hdr_blue['CDELT3']   
    
    
    hdulist_red=pf.open(inredcube)

    hdr_red=hdulist_red['BINNED_FLUX_{}'.format(bin_ext)].header
    name2=hdr_red['NAME']
    # hdr_red2=hdulist_red['RE_MGE'].header
    
    #hdulist_red.close()

    # Pull out the wavelength range, RED
    x_red=np.arange(hdr_red['NAXIS3'])+1
    L0_red=hdr_red['CRVAL3']-hdr_red['CRPIX3']*hdr_red['CDELT3'] #Lc-pix*dL

    lambda_galaxy_red=L0_red+x_red*hdr_red['CDELT3']
    
    
    if name1 != name2:
        
        raise Exception("Spectra are not from the same galaxy.\n")
    
    else:
        
        output = dict()
        
        for num, idx in zip(bin_nums, bin_indices):
            
            # print (hdulist_red['BINNED_FLUX_SN25'].data.shape)
                
            # print (idx)
            # # print (np.nansum())
            # print (np.nansum(hdulist_blue['BINNED_FLUX_SN25'].data[:,idx[0],idx[1]]))
            # print (np.nansum(hdulist_blue['BINNED_FLUX_SN25'].data[:,(8,24)]))
            # print (hdulist_blue['BINNED_FLUX_SN25'].data.shape)
            
            comb_dict=combine_blue_red(
                lambda_galaxy_blue, 
                hdulist_blue['BINNED_FLUX_{}'.format(bin_ext)].data[:,idx[0],idx[1]], 
                lambda_galaxy_red, 
                hdulist_red['BINNED_FLUX_{}'.format(bin_ext)].data[:,idx[0],idx[1]], 
                var_blue=hdulist_blue['BINNED_VARIANCE_{}'.format(bin_ext)].data[:,idx[0],idx[1]], 
                var_red=hdulist_red['BINNED_VARIANCE_{}'.format(bin_ext)].data[:,idx[0],idx[1]],
            )
                    
            x=np.arange(len(comb_dict['spectrum']))+1
            L0=hdr_blue['CRVAL3']-hdr_blue['CRPIX3']*hdr_blue['CDELT3'] #Lc-pix*dL
        
            lambda_galaxy=L0+x*hdr_blue['CDELT3']
            
            output[int(num)] = (comb_dict['spectrum'], comb_dict['variance'], lambda_galaxy )
            
        return output
    
    

def spec_prep(inbluespec, inredspec, overwrite=False):
    """ Take a blue and red SAMI aperture spectra and combine them, resampling the red. A combined FITS file is produced.
    """

    # --------------------------------------------------------------
    # Read in the BLUE data aper
    hdulist_blue=pf.open(inbluespec)
   
    hdr_blue=hdulist_blue['PRIMARY'].header
    name1=hdr_blue['NAME']
    hdr_blue2=hdulist_blue['RE_MGE'].header

    #hdulist_blue.close()

    # Pull out the wavelength range, BLUE
    x_blue=np.arange(hdr_blue2['NAXIS1'])+1
    L0_blue=hdr_blue2['CRVAL1']-hdr_blue2['CRPIX1']*hdr_blue2['CDELT1'] #Lc-pix*dL

    lambda_galaxy_blue=L0_blue+x_blue*hdr_blue2['CDELT1']

    # --------------------------------------------------------------
    # Now read in the RED data aper
    hdulist_red=pf.open(inredspec)

    hdr_red=hdulist_red['PRIMARY'].header
    name2=hdr_red['NAME']
    hdr_red2=hdulist_red['RE_MGE'].header
    
    #hdulist_red.close()

    # Pull out the wavelength range, RED
    x_red=np.arange(hdr_red2['NAXIS1'])+1
    L0_red=hdr_red2['CRVAL1']-hdr_red2['CRPIX1']*hdr_red2['CDELT1'] #Lc-pix*dL

    lambda_galaxy_red=L0_red+x_red*hdr_red2['CDELT1']

    # --------------------------------------------------------------
    # First check that the two apers are from the same galaxy

    # Check that the two data apers are from the same galaxy
    if name1 != name2:
        
        raise Exception("Spectra are not from the same galaxy.\n")
    
    else:
        
        comb_dict=combine_blue_red(lambda_galaxy_blue, hdulist_blue['RE_MGE'].data, lambda_galaxy_red, hdulist_red['RE_MGE'].data, 
          var_blue=hdulist_blue['RE_MGE_VAR'].data, var_red=hdulist_red['RE_MGE_VAR'].data)
                
        x=np.arange(len(comb_dict['spectrum']))+1
        L0=hdr_blue2['CRVAL1']-hdr_blue2['CRPIX1']*hdr_blue2['CDELT1'] #Lc-pix*dL
    
        lambda_galaxy=L0+x*hdr_blue2['CDELT1']
        
        return comb_dict['spectrum'], comb_dict['variance'], lambda_galaxy
        
        
        
def combine_blue_red(lam_blue, spec_blue, lam_red, spec_red, FWHM_blue=2.6554, FWHM_red=1.5916, var_blue=None, var_red=None):
    """ A function to combine red and blue SAMI spectra for ppxf fitting. Based on Nic Scott's stellat pops code.

    spec_blue, var_blue, spec_red, var_red, FWHM_blue, FWHM_red, lam_blue, lam_red
    
    FWHM values are from van de Sande et al. 2017, and updated for v0.12
    """

    # ---------------------------------------------------------------------------------
    # Define a common wavelength sampling based on the wavelength sampling in the blue
    samp_blue=lam_blue[1]-lam_blue[0]
    samp_red=lam_red[1]-lam_red[0]

    lam_range_full=lam_red[-1]-lam_blue[0]
    nrange=np.ceil(lam_range_full/samp_blue)+1
    #print nran

    lam_all=np.arange(nrange+1)*samp_blue+lam_blue[0]
    #print lam_all

    # Ensure the array sizes are ok for future steps
    tmp = lam_all - np.min(lam_red)
    tmp[tmp < 0] = 1e10
    red_lower = tmp.argmin()
    tmp = lam_all - np.max(lam_red)
    tmp[tmp < 0] = 1e10
    red_upper = tmp.argmin()
    del tmp

    # The red wavelength range but resampled
    lam_resamp_red=lam_all[red_lower]+np.arange(red_upper-red_lower)*samp_blue
    #print lam_resamp_red
    
    # Find the convolution parameters for the red
    FWHM_dif=np.sqrt(FWHM_blue**2 - FWHM_red**2)
    sigma_conv=FWHM_dif/2.355/samp_red
    
    # print (sigma_conv)
    
    # print (samp_red)
    # print (samp_blue)

    # Smooth the red spectrum
    spec_red_smooth=ndimage.gaussian_filter1d(spec_red, sigma_conv)
    
    # print (spec_red_smooth.shape)
    # print (lam_red.shape)

    # Interpolate red spectrum and variance onto new wavelength sampling
    f_spec=interpolate.interp1d(lam_red, spec_red_smooth, kind='linear')
    red_spec_resamp=f_spec(lam_resamp_red)

    spectrum = np.ones((len(lam_all)))*np.nan
    spectrum[0:len(spec_blue)] = spec_blue
    spectrum[red_lower:red_upper] = red_spec_resamp

    # Now do exactly the same to the variance if you have the spectra
    if np.all(var_red is not None) and np.all(var_blue is not None):
        # Convolve
        var_red_smooth=ndimage.gaussian_filter1d(var_red, sigma_conv)

        # Interpolate the smoothed red variance spectrum onto the new wavelength sampling
        f_var=interpolate.interp1d(lam_red, var_red_smooth, kind='linear')
        red_var_resamp=f_var(lam_resamp_red)
    
        variance = np.ones((len(lam_all)))*np.nan
        variance[0:len(var_blue)] = var_blue
        variance[red_lower:red_upper] = red_var_resamp

        # Make a dictionary including the variance spectrum
        dict_out={'spectrum': spectrum, 'variance': variance, 'wavelength': lam_all}

    else:
        # Make a dictionary without the variance
        dict_out={'spectrum': spectrum, 'wavelength': lam_all}

    # Return the dictionary
    return dict_out



def read_templates_FITS(intemplates):
    """ A function to read a set of templates in FITS format.

    Parameters
    ----------

    intemplates: string
        name of a file containing a list of templates file names, i.e. a .txt or
    .list file, with one file name per line.

    Return
    ------
    template_names: array_like
        an array of template names
    lamtemplate_array_orig: array_like
        wavelengths
    template_array_orig: array_like
        template spectra (not rebinned)

    """

    # --------------------------------------------------------------
    # Read in the names of the templates
    templates_all=[]
    # templates_names=[]

    templates_all = intemplates

    # Number of templates in the list
    n_templates=len(templates_all)
    #print templates_names

    # Get the size of the templates
    list1=pf.open(templates_all[0])
    hdr=list1['PRIMARY'].header
    data=np.squeeze(list1['PRIMARY'].data)
    list1.close()

    # This is to deal with one template without CRPIX1
    # ONLY valid for the MILES templates
    #try:
    #    pix=hdr['CRPIX1']
    #except:
    #    pix=1.0

    # Pull out the wavelength range
    x=np.arange(hdr['NAXIS1'])+1
    L0=hdr['CRVAL1']-hdr['CRPIX1']*hdr['CDELT1'] #Lc-pix*dL

    lambda_tem=L0+x*hdr['CDELT1']

    # Set up the output arrays
    template_array_orig=np.empty((lambda_tem.size, n_templates))
    lamtemplate_array_orig=np.empty((data.size, n_templates))

    for i, template in enumerate(templates_all):
        
        list1=pf.open(template)
        hdr=list1['PRIMARY'].header
        data=np.squeeze(list1['PRIMARY'].data)
        list1.close()

        # This is to deal with one template without CRPIX1
        try:
            pix=hdr['CRPIX1']
        except:
            pix=1.0

        # Pull out the wavelength range
        x=np.arange(hdr['NAXIS1'])+1
        L0=hdr['CRVAL1']-pix*hdr['CDELT1'] #Lc-pix*dL

        lambda_tem=L0+x*hdr['CDELT1']

        template_array_orig[:,i]=data
        lamtemplate_array_orig[:,i]=lambda_tem

    return lamtemplate_array_orig, template_array_orig


def ndix_unique(x):
    """
    Returns an N-dimensional array of indices
    of the unique values in x
    ----------
    x: np.array
        Array with arbitrary dimensions
    Returns
    -------
    - 1D-array of sorted unique values
    - Array of arrays. Each array contains the indices where a
      given value in x is found
    """
    x_flat = x.ravel()
    # ix_flat = np.argsort(x_flat)
    # u, ix_u = np.unique(x_flat[ix_flat], return_index=True)
    u, ix_u = np.unique(x_flat, return_index=True)
    # print (ix_u)
    ix_ndim = []
    for i in ix_u:
        ix_ndim.append(np.unravel_index(i, x.shape))
    # ix_ndim = np.unravel_index(ix_u, x.shape)
    # ix_ndim = np.c_[ix_ndim] if x.ndim > 1 else ix_flat
    # return u, np.split(ix_ndim, ix_u[1:])
    return u, ix_ndim