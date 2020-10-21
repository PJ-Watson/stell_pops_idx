#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:44:21 2020

@author: peter
"""


# ### Folder location to save all data
# directory = "glamdring_v1_residuals"

# ### Location of templates for full spectrum fit (including pattern matching)
# template_loc = "templates/MILES_library_v9.1_FITS/*"

# ### Location of Lick Indices definition
# LI_loc = "templates/Lick_Indices.txt"



# ###################################################################################################




# import sami_ppxf_utils_pjw
# import glob
# from astropy.table import Table, vstack, join
# from astropy.io import ascii
# import os
# from multiprocessing import Pool
# import json
# from functools import partial

# from pathlib import Path

# from tools_disp_corr import Dispersion_Correction, get_files, corr_fit
# from primary_routine import main_routine

# # template_list = glob.glob("templates/MILES_subset/Mbi1.30Z*")

# # template_list = glob.glob("templates/MILES_subset/Mbi1.30Zp0.06*")

# lam_temp, flux_temp = sami_ppxf_utils_pjw.read_templates_FITS(glob.glob(template_loc))

# LI = Table.read(LI_loc, format = 'ascii')

# tmj_models = ascii.read("templates/tmj.dat", Reader = ascii.CommentedHeader, header_start = -3)

# LI_to_tmj = {"Hdelta_A"     :"HdA",     "Hdelta_F"      :"HdF",
#              "CN_1"         :"CN1",     "CN_2"          :"CN2",
#              "Ca4227"       :"Ca4227",  "G4300"         :"G4300",
#              "Hgamma_A"     :"HgA",     "Hgamma_F"      :"HgF",
#              "Fe4383"       :"Fe4383",  "Ca4455"        :"Ca4455",
#              "Fe4531"       :"Fe4531",  "Fe4668"        :"C24668",
#              "H_beta"       :"Hb",      "Fe5015"        :"Fe5015",
#              "Mg_1"         :"Mg1",     "Mg_2"          :"Mg2",
#              "Mg_b"         :"Mgb",     "Fe5270"        :"Fe5270",
#              "Fe5335"       :"Fe5335",  "Fe5406"        :"Fe5406"
#              }

# with open("templates/LI_to_tmj.json", 'w') as outfile:
#     json.dump(LI_to_tmj, outfile)

# LI_to_sch = {"Hdelta_A"     :"HdA",     "Hdelta_F"      :"HdF",
#              "CN_1"         :"CN1",     "CN_2"          :"CN2",
#              "Ca4227"       :"Ca4227",  "G4300"         :"G4300",
#              "Hgamma_A"     :"HgA",     "Hgamma_F"      :"HgF",
#              "Fe4383"       :"Fe4383",  "Fe4668"        :"C2-4668",
#              "H_beta"       :"Hb",      "Fe5015"        :"Fe5015",
#              "Mg_b"         :"Mgb",     "Mg_2"          :"Mg2",
#              "Fe5270"       :"Fe5270",  "Fe5335"        :"Fe5335",
#              }

# with open("templates/LI_to_sch.json", 'w') as outfile:
#     json.dump(LI_to_sch, outfile)

# sch_1 = Table.read('templates/datafile31.txt',format = 'ascii')
# sch_1['x'] = 0
# sch_1['x'].name = '[alpha/Fe]'
# sch_1.remove_columns(['CN1','CN2','Ca4227','G4300','C2-4668'])
   
# sch_2 = Table.read('templates/datafile32.txt',format = 'ascii')
# sch_2['x'] = 0.42
# sch_2['x'].name = '[alpha/Fe]'  
# sch_2.remove_columns(['CN1','CN2','Ca4227','G4300','C2-4668'])
   
# # sch_models = rf.structured_to_unstructured(vstack([sch_1,sch_2]).as_array())

# sch_models = vstack([sch_1, sch_2])

# cat_1 = Table.read("cat_SAMI/table_data/Stelkin.fits", format = "fits")
# cat_2 = Table.read("cat_SAMI/table_data/DR2Sample.fits", format = "fits")
# cat = join(cat_1, cat_2, keys = "CATID")

# if not os.path.exists("sample_data/{0}/".format(directory)):
#     os.mkdir("sample_data/{0}/".format(directory))
# if not os.path.exists("sample_data/{0}/logs/".format(directory)):
#     os.mkdir("sample_data/{0}/logs/".format(directory))
# if not os.path.exists("sample_data/{0}/gals/".format(directory)):
#     os.mkdir("sample_data/{0}/gals/".format(directory))
# if not os.path.exists("sample_data/{0}/lams/".format(directory)):
#     os.mkdir("sample_data/{0}/lams/".format(directory))
# if not os.path.exists("sample_data/{0}/res/".format(directory)):
#     os.mkdir("sample_data/{0}/res/".format(directory))


# # age_list = [1.5,3.5,8,14]

# # temp_selection = ["Mbi1.30Zm0.25*",
# #                   "Mbi1.30Zp0.06*",
# #                   "Mbi1.30Zp0.15*",
# #                   "Mbi1.30Zp0.26*"]

# # # temps_alpha_0 = get_files(Path("templates/MILES_SSP/MILES_BASTI_BI_Ep0.00/"),
# # #                           temp_selection)

# # # print (temps_alpha_0)

# # # Dispersion_Correction(temp_dir = temps_alpha_0, 
# # #                       out_dir = os.fspath(Path("templates/vel_disp_corrs/Z_subset__Ep0.00/")),
# # #                       re_run = True, output_ages = age_list)

# # temps_alpha_4 = get_files(Path("templates/MILES_SSP/MILES_BASTI_BI_Ep0.40/"),
# #                           temp_selection)

# # Dispersion_Correction(temp_dir = temps_alpha_4, 
# #                       out_dir = os.fspath(Path("templates/vel_disp_corrs/Z_subset__Ep0.40/")),
# #                       re_run = True, output_ages = age_list)

# # corr_fit(age_list)

# # with open("sample_data/{0}/finished.json".format(directory), 'w') as outfile:
# #     json.dump(LI_to_sch, outfile)
    
# if __name__ == '__main__':       
#     # print ('yes')
#     pool = Pool(os.cpu_count() - 1)
#     pool.map(partial(main_routine, lam_temp = lam_temp, flux_temp = flux_temp,
#                       LI = LI, tmj_models = tmj_models, sch_models = sch_models,
#                       directory = directory), cat[0:])





###################################################################################################




import sami_ppxf_utils_pjw
import glob
from astropy.table import Table, vstack, join
from astropy.io import ascii
import os
from multiprocessing import Pool
import json
from functools import partial

from pathlib import Path

from tools_disp_corr import Dispersion_Correction, get_files, corr_fit
from primary_routine import main_routine
from mpi4py import MPI
import numpy as np

# template_list = glob.glob("templates/MILES_subset/Mbi1.30Z*")

# template_list = glob.glob("templates/MILES_subset/Mbi1.30Zp0.06*")




# age_list = [1.5,3.5,8,14]

# temp_selection = ["Mbi1.30Zm0.25*",
#                   "Mbi1.30Zp0.06*",
#                   "Mbi1.30Zp0.15*",
#                   "Mbi1.30Zp0.26*"]

# # temps_alpha_0 = get_files(Path("templates/MILES_SSP/MILES_BASTI_BI_Ep0.00/"),
# #                           temp_selection)

# # print (temps_alpha_0)

# # Dispersion_Correction(temp_dir = temps_alpha_0, 
# #                       out_dir = os.fspath(Path("templates/vel_disp_corrs/Z_subset__Ep0.00/")),
# #                       re_run = True, output_ages = age_list)

# temps_alpha_4 = get_files(Path("templates/MILES_SSP/MILES_BASTI_BI_Ep0.40/"),
#                           temp_selection)

# Dispersion_Correction(temp_dir = temps_alpha_4, 
#                       out_dir = os.fspath(Path("templates/vel_disp_corrs/Z_subset__Ep0.40/")),
#                       re_run = True, output_ages = age_list)

# corr_fit(age_list)

# with open("sample_data/{0}/finished.json".format(directory), 'w') as outfile:
#     json.dump(LI_to_sch, outfile)
    
if __name__ == '__main__':       
    # print ('yes')
    
    ### Folder location to save all data
    directory = "glamdring_v2_variance_weight"
    
    ### Location of templates for full spectrum fit (including pattern matching)
    template_loc = "templates/MILES_library_v9.1_FITS/*"
    
    ### Location of Lick Indices definition
    LI_loc = "templates/Lick_Indices.txt"

    
    lam_temp, flux_temp = sami_ppxf_utils_pjw.read_templates_FITS(glob.glob(template_loc))

    LI = Table.read(LI_loc, format = 'ascii')
    
    tmj_models = ascii.read("templates/tmj.dat", Reader = ascii.CommentedHeader, header_start = -3)
    
    LI_to_tmj = {"Hdelta_A"     :"HdA",     "Hdelta_F"      :"HdF",
                 "CN_1"         :"CN1",     "CN_2"          :"CN2",
                 "Ca4227"       :"Ca4227",  "G4300"         :"G4300",
                 "Hgamma_A"     :"HgA",     "Hgamma_F"      :"HgF",
                 "Fe4383"       :"Fe4383",  "Ca4455"        :"Ca4455",
                 "Fe4531"       :"Fe4531",  "Fe4668"        :"C24668",
                 "H_beta"       :"Hb",      "Fe5015"        :"Fe5015",
                 "Mg_1"         :"Mg1",     "Mg_2"          :"Mg2",
                 "Mg_b"         :"Mgb",     "Fe5270"        :"Fe5270",
                 "Fe5335"       :"Fe5335",  "Fe5406"        :"Fe5406"
                 }
    
    with open("templates/LI_to_tmj.json", 'w') as outfile:
        json.dump(LI_to_tmj, outfile)
    
    LI_to_sch = {"Hdelta_A"     :"HdA",     "Hdelta_F"      :"HdF",
                 "CN_1"         :"CN1",     "CN_2"          :"CN2",
                 "Ca4227"       :"Ca4227",  "G4300"         :"G4300",
                 "Hgamma_A"     :"HgA",     "Hgamma_F"      :"HgF",
                 "Fe4383"       :"Fe4383",  "Fe4668"        :"C2-4668",
                 "H_beta"       :"Hb",      "Fe5015"        :"Fe5015",
                 "Mg_b"         :"Mgb",     "Mg_2"          :"Mg2",
                 "Fe5270"       :"Fe5270",  "Fe5335"        :"Fe5335",
                 }
    
    with open("templates/LI_to_sch.json", 'w') as outfile:
        json.dump(LI_to_sch, outfile)
    
    sch_1 = Table.read('templates/datafile31.txt',format = 'ascii')
    sch_1['x'] = 0
    sch_1['x'].name = '[alpha/Fe]'
    sch_1.remove_columns(['CN1','CN2','Ca4227','G4300','C2-4668'])
       
    sch_2 = Table.read('templates/datafile32.txt',format = 'ascii')
    sch_2['x'] = 0.42
    sch_2['x'].name = '[alpha/Fe]'  
    sch_2.remove_columns(['CN1','CN2','Ca4227','G4300','C2-4668'])
       
    # sch_models = rf.structured_to_unstructured(vstack([sch_1,sch_2]).as_array())
    
    sch_models = vstack([sch_1, sch_2])
    
    cat_1 = Table.read("cat_SAMI/table_data/Stelkin.fits", format = "fits")
    cat_2 = Table.read("cat_SAMI/table_data/DR2Sample.fits", format = "fits")
    cat = join(cat_1, cat_2, keys = "CATID")
    try:
        if not os.path.exists("sample_data/{0}/".format(directory)):
            os.mkdir("sample_data/{0}/".format(directory))
        if not os.path.exists("sample_data/{0}/logs/".format(directory)):
            os.mkdir("sample_data/{0}/logs/".format(directory))
        if not os.path.exists("sample_data/{0}/gals/".format(directory)):
            os.mkdir("sample_data/{0}/gals/".format(directory))
        if not os.path.exists("sample_data/{0}/lams/".format(directory)):
            os.mkdir("sample_data/{0}/lams/".format(directory))
        if not os.path.exists("sample_data/{0}/res/".format(directory)):
            os.mkdir("sample_data/{0}/res/".format(directory))
    except:
        pass
    
    
    # for row in cat[0:]:
        
    #     main_routine(row, lam_temp, flux_temp, LI, tmj_models, sch_models,
    #                   directory)
    
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    
    print (rank, size)
    
    number = int(np.floor(len(cat[0:])/size))
    
    print (number)
    
    for i in range (number):
        
        idx = int((i*size) + rank)
        
        print (idx)
        
        # main_routine(cat[idx])
        
        main_routine(cat[idx], lam_temp, flux_temp, LI, tmj_models, sch_models,
                      directory)
        
    try:
        
        row = cat[int(number+rank)]
        
        # print (row)
        
        main_routine(row, lam_temp, flux_temp, LI, tmj_models, sch_models,
                      directory)
        
    except:
        
        pass
    
    finally:
        
        print ("Finished on {}".format(rank))
    
    # pool = Pool(os.cpu_count())
    
    # pool.map(partial(main_routine, lam_temp = lam_temp, flux_temp = flux_temp,
    #                   LI = LI, tmj_models = tmj_models, sch_models = sch_models,
    #                   directory = directory), cat[0:])