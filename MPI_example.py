#!/usr/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:49:32 2020

@author: peter
"""

import stell_pops_idx

import glob
from astropy.table import Table
from astropy.io import ascii

from pathlib import Path

from stell_pops_idx.tools_disp_corr import Dispersion_Correction, get_files, corr_fit
import numpy as np

from stell_pops_idx.tools_spec_ingest import read_templates_FITS, binned_data_wrapper

import argparse

from mpi4py import MPI
    

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser(
        description="Extract stellar population parameters from SAMI data."
    )
    
    ap.add_argument(
        "-i",
        "--Lick_indices", 
        default=Path(stell_pops_idx.__path__[0]) / "Lick_Indices_def.txt",
        help=("Wavelength definitions of Lick indices to be included. "
              "This must match the format of the default table."),
    ) 
    
    ap.add_argument(
        "--disp_corr_age_list",
        default=[1.5,3.5,8,14],
        help=("Comma delimited ages to use when calculating velocity dispersion corrections "
              "to the indices."),
        type=lambda s: [int(item) for item in s.split(',')],
    )
    
    ap.add_argument(
        "--disp_corr_temp_select",
        # default=[
        #     "Mbi1.30Zm0.25*",
        #     "Mbi1.30Zp0.06*",
        #     "Mbi1.30Zp0.15*",
        #     "Mbi1.30Zp0.26*",
        # ],
        help=("Pattern to match when selecting templates for the dispersion correction."),
        type=lambda s: [str(item) for item in s.split(',')],
    )
    
    ap.add_argument(
        "-l",
        "--spec_temp_lib_loc",
        required=True,
        help=("Location of spectral template library to use during kinematic fitting. "
              "Can include filename pattern to match."),
    )
    
    ap.add_argument(
        "-s",
        "--SSP_temp_loc",
        required=True,
        help=("Location of SSP templates to use during stellar population fitting. "
              "Assumed to match the format of tmj.dat (Thomas, Maraston, Johansson 2011)."),
    )
    
    ap.add_argument(
        "-t",
        "--temp_loc",
        required=True,
        help="Directory to save new templates in.",
    )
    
    ap.add_argument(
        "-c",
        "--input_catalogue_loc",
        required=True,
        help=("Location of the .fits input catalogue. "
              "Must contain both CATID and z_in columns."),
    )
    
    ap.add_argument(
        "-d",
        "--input_data_loc",
        required=True,
        help=("Location of the input data cubes. "
              "Assumed to match the format of SAMI DR3 "
              "(e.g. CATID_A_cube_blue.fits)."),
    )    
    
    ap.add_argument(
        "-o",
        "--output_data_loc",
        required=True,
        help=("Location of the output data."),
    )
    
    ap.add_argument(
        "--previous_data_loc",
        help=("Location of the previous data."),
    )
    
    args = ap.parse_args()
    
    LI = Table.read(args.Lick_indices, format = 'ascii')
    
    template_directory = Path(args.temp_loc)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank==0:
    
        ### Generate the dispersion correction files
    
        if not glob.glob((
                template_directory / "vel_disp_corrs" / "corr_*_gyr.pkl"
            ).__str__()):
            try:
                corr_temps = get_files(args.disp_corr_temp_select)
                
                assert len(corr_temps) >=1
                
            except Exception:
                disp_corr_files = ("Could not load dispersion correction input templates.")
                raise Exception(disp_corr_files)
            
            Dispersion_Correction(
                temp_dir = corr_temps,
                out_dir = (template_directory / "vel_disp_corrs" / "base_measurements"),
                re_run=False,
                output_ages=args.disp_corr_age_list, 
                bands=LI,
            )
            
            corr_fit(
                args.disp_corr_age_list, 
                template_directory / "vel_disp_corrs" / "base_measurements",
                template_directory / "vel_disp_corrs",
                bands=LI,
            )
    
    lam_temp, flux_temp = read_templates_FITS(
        glob.glob(args.spec_temp_lib_loc)
    )    
    
    SSP_models = ascii.read(
        args.SSP_temp_loc, 
        Reader = ascii.CommentedHeader, 
        header_start = -3,
    )
    
    input_cat = Table.read(args.input_catalogue_loc)
    
    if rank == 0:
        idx_chunks = np.array_split(np.arange(len(input_cat)), size)
    else:
        idx_chunks = None
            
    idx_chunks = comm.scatter(idx_chunks, root=0)
    
    print (rank, idx_chunks, flush=True)
    
    for row in input_cat[idx_chunks]:
        
        data_location = Path(args.input_data_loc)
    
        try:
            bin_dict = binned_data_wrapper(
                data_location / "{}_A_cube_blue.fits.gz".format(row[0]),
                data_location / "{}_A_cube_red.fits.gz".format(row[0]),
                bin_ext="SN_25",
            )
        except:
            continue
        
        try:
            bin_dict = binned_data_wrapper(
                data_location / "{}_A_cube_blue.fits.gz".format(row[0]),
                data_location / "{}_A_cube_red.fits.gz".format(row[0]),
                bin_ext="RE_LOG_4", #Change this to the extension you want to use
            )
        except:
            continue
        
        for bin_num, (spec, var, wave) in bin_dict.items():
            if bin_num == 0:
                pass
            else:
                
                out_location = Path(args.output_data_loc) / "{}".format(row[0])
                
                out_location.mkdir(parents=True, exist_ok=True)
                
                if not (out_location / "{}".format(bin_num) / "output.fits").is_file():
                    
                    print (
                        "Beginning analysis on core {0}.\n"
                        "Using galaxy {1}, bin {2}.\n".format(
                            rank,
                            row[0],
                            bin_num,
                        )
                    )
                    
                    out_location = Path(args.output_data_loc) / "{}".format(row[0])
                    
                    out_location.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        stell_pops_idx.run_all(
                            out_location,
                            "{}".format(bin_num),
                            spec, 
                            var, 
                            wave,
                            template_directory,
                            flux_temp,
                            lam_temp,
                            row[1],
                            LI,
                            reg_SSP_models=SSP_models,
                        )
                    except Exception as e:
                        print (e)
                    
                    print (
                        "Finished analysis on core {0}, for "
                        "galaxy {1}, bin {2}.\n".format(
                            rank,
                            row[0],
                            bin_num,
                        )
                    )
                
    print ("Completed all galaxies on core {}".format(rank), flush=True)
                    