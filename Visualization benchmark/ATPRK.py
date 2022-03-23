from __future__ import absolute_import, division, print_function

import numpy as np
from utils import *
##############################################
# ATPRK function
#################################################

def ATPRK(ndvi_1km, LST_K_day_1km, ndvi_2km, LST_K_day_2km):
    #FIT
    path_index = ndvi_2km
    path_temperature = LST_K_day_2km
    #path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
    #path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
    min_T = 270
    path_fit = 'Fit_NDBI_T.txt'
    plot =0
    path_plot = ''

    linear_fit_test(path_index, path_temperature, min_T, path_fit, plot, path_plot)
    # index (894, 135)
    # Temp (894, 135)

    #UNMIXING
    path_index = ndvi_1km
    path_temperature = LST_K_day_2km
    path_fit = 'Fit_NDBI_T.txt'
    path_out = ''
    iscale=4
    path_mask=0

    T_unm = linear_unmixing_test(path_index, path_temperature, path_fit, iscale,path_mask, path_out)
    # index (1787, 269)
    # Temp (358, 54)

    #CORR
    path_index = ndvi_2km
    path_fit = 'Fit_NDBI_T.txt'
    path_temperature = LST_K_day_2km
    path_unm = 'T_unm_1km_from2km_jourNDBI.bsq'
    iscale=4
    # scc=2
    scc=32
    block_size = 3 # 3 
    sill=7
    ran=1000
    path_out='Delta_T_1km_from2km_jour_NDBI_atprk.bsq'
    path_out1='T_unm_add_1km_from2km_jour_NDBI_atprk.bsq'
    path_out2='Delta_T_final_1km_from2km_jour_NDBI_atprk.bsq'
    path_out3='T_unmixed_final_1km_from2km_jour_NDBI_atprk.bsq'
    path_plot= 'Gamma_c_NDBI_2km.png'

    prediction , Delta_T_final, TT_unm= correction_ATPRK_test(path_index, path_fit, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot,T_unm)
    return prediction, LST_K_day_1km, Delta_T_final, TT_unm