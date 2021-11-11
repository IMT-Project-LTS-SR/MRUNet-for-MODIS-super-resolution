
'''
######################################## FITS ########################
######################################################################
######################################################################
'''
######### Linear fit ################

from Unmixing_Module_Python import Linear_Fit
path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
min_T = 285
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'
plot =1
path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_vs_T_080628.png'

Linear_Fit.linear_fit(path_index, path_temperature, min_T, path_fit, plot, path_plot)

######### Bilinear fit ################

from Unmixing_Module_Python import Bilinear_Fit
path_index1 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_index2 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/HUTS/Albedo_estimation/080628/albedo_080628_40m.bsq'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
min_T = 285
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'

Bilinear_Fit.bilinear_fit(path_index1, path_index2, path_temperature, min_T, path_fit)


########## HUTS fit ################

from Unmixing_Module_Python import HUTS_Fit
path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_albedo = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/HUTS/Albedo_estimation/080628/albedo_080628_40m.bsq'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
min_T = 285
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'

HUTS_Fit.huts_fit(path_index, path_albedo, path_temperature, min_T, path_fit)

########## linear fit window ################

from Unmixing_Module_Python import Linear_Fit_Window
path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
min_T = 285
b_radius=5
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'
file_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Slope_40m_NDBI.bsq'
file_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Intercept_40m_NDBI.bsq'
plot =1
path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_vs_T_080628.png'
Linear_Fit_Window.linear_fit_window(path_index, path_temperature, min_T, b_radius, file_out1, file_out2, path_fit, plot, path_plot)

########### Fit by class #####################

from Unmixing_Module_Python import Fit_ByClass
path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_class = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/Classif_Unmixing/Index_Based_Classification/Classif_080628/Classif_40m.bsq' 
min_T = 285
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'
plot =1
path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_vs_T_080628.png'

Fit_ByClass.fit_byclass(path_index, path_temperature, path_class, min_T, path_fit, plot, path_plot)


'''
#################################### UNMIXING ########################
######################################################################
######################################################################
'''


########## Linear Unmixing #####################

from Unmixing_Module_Python import Linear_Unmixing

path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628_20m.img'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'
path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI.bsq'
iscale=2
path_mask=0

Linear_Unmixing.linear_unmixing(path_index, path_temperature, path_fit, iscale,path_mask, path_out)

########## Bilinear unmixing ##################

from Unmixing_Module_Python import Bilinear_Unmixing

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
from math import *
from scipy.stats import linregress
from scipy import stats

path_index1 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628_20m.img'
path_index2 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/HUTS/Albedo_estimation/080628/albedo_080628_20m.bsq'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'
iscale=2
path_mask=0
path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI.bsq'

Bilinear_Unmixing.bilinear_unmixing(path_index1, path_index2, path_temperature, path_fit, iscale, path_mask, path_out)

########## HUTS Unmixing #####################

from Unmixing_Module_Python import Huts_Unmixing

path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628_20m.img'
path_albedo = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/HUTS/Albedo_estimation/080628/albedo_080628_20m.bsq'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'
path_mask=0
path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI_huts.bsq'
iscale= 2
Huts_Unmixing.huts_unmixing(path_index, path_albedo, path_temperature, path_fit, iscale, path_mask, path_out)

########### Linear Unmixing By Class ##########

from Unmixing_Module_Python import Linear_Unmixing_ByClass
	
path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628_20m.img'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'
path_mask=0
path_class = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/Classif_Unmixing/Index_Based_Classification/Classif_080628/Classif_20m.bsq' 
path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI_huts.bsq'
iscale= 2	
	
Linear_Unmixing_ByClass.linear_unmixing_byclass(path_index, path_temperature, path_class, path_fit, iscale, path_mask, path_out)

########### AATPRK Unmixing ##########

from Unmixing_Module_Python import AATPRK_Unmixing
	
path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628_20m.img'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'	
path_slope='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Slope_40m_NDBI.bsq'
path_intercept='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Intercept_40m_NDBI.bsq'
iscale=2
path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI_aatprk.bsq'

AATPRK_Unmixing.aatprk_unmixing(path_index, path_temperature, path_slope, path_intercept, iscale, path_out)

'''
##################################### CORRECTION #####################
######################################################################
######################################################################
'''

############## correction ############

from Unmixing_Module_Python import Correction

path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI.bsq'
iscale=2
path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_40m_from20m_jour_NDBI.bsq'
path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_add_40m_from20m_jour_NDBI.bsq'
path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_final_20m_from40m_jour_NDBI.bsq'
path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unmixed_final_20m_from40m_jour_NDBI.bsq'

Correction.correction(path_index, path_temperature, path_unm, iscale, path_out, path_out1, path_out2, path_out3)

############## correction v2 ############

from Unmixing_Module_Python import Correction_v2

path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI.bsq'
iscale=2
path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_40m_from20m_jour_NDBI_v2.bsq'
path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_add_40m_from20m_jour_NDBI_v2.bsq'
path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_final_20m_from40m_jour_NDBI_v2.bsq'
path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unmixed_final_20m_from40m_jour_NDBI_v2.bsq'

Correction_v2.correction_v2(path_index, path_fit, path_temperature, path_unm, iscale, path_out, path_out1, path_out2, path_out3)

############## Quality correction ############

from Unmixing_Module_Python import Quality_Correction

path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI.bsq'
iscale=2
max_thre = None
min_thre = None
path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_40m_from20m_jour_NDBI_qual.bsq'
path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_add_40m_from20m_jour_NDBI_qual.bsq'
path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_final_20m_from40m_jour_NDBI_qual.bsq'
path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unmixed_final_20m_from40m_jour_NDBI_qual.bsq'

Quality_Correction.quality_correction(path_index, path_temperature, path_unm, iscale, max_thre, min_thre, path_out, path_out1, path_out2, path_out3)

############## correction ATPRK ############

from Unmixing_Module_Python import Correction_ATPRK

path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Fit_NDBI_T_080628.txt'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI.bsq'
iscale=2
scc=40
block_size = 5
sill=7
ran=1000
path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_40m_from20m_jour_NDBI_atprk.bsq'
path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_add_40m_from20m_jour_NDBI_atprk.bsq'
path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_final_20m_from40m_jour_NDBI_atprk.bsq'
path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unmixed_final_20m_from40m_jour_NDBI_atprk.bsq'
path_plot= '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Gamma_c_NDBI_40m.png'

Correction_ATPRK.correction_ATPRK(path_index, path_fit, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot)


############## correction AATPRK ############

from Unmixing_Module_Python import Correction_AATPRK

path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/NDBI_080628.img'
path_slope='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Slope_40m_NDBI.bsq'
path_intercept='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Intercept_40m_NDBI.bsq'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_20m_from40m_jourNDBI.bsq'
iscale=2
scc=40
block_size = 5
sill=7
ran=1000
path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_40m_from20m_jour_NDBI_aatprk.bsq'
path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unm_add_40m_from20m_jour_NDBI_aatprk.bsq'
path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Delta_T_final_20m_from40m_jour_NDBI_aatprk.bsq'
path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unmixed_final_20m_from40m_jour_NDBI_aatprk.bsq'
path_plot= '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Gamma_c_NDBI_40m_aatprk.png'

Correction_AATPRK.correction_AATPRK(path_index, path_slope, path_intercept, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot)


'''
##################################### STATS ##########################
######################################################################
######################################################################
'''

from Unmixing_Module_Python import Unm_Stats

path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/T_unmixed_final_20m_from40m_jour_NDBI.bsq'
path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
min_T = 285
path_file = "/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Output.txt"
path_file1 = "/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Output_ttest_.txt"
path_file2 = "/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Output_corr.txt"
plot =1
path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Hist_T20m_080628_jour.png'
path_plot1 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Hist_Tunm_40m_NDBI_080628_jour.png'
path_plot2 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Hist_delta_40m_NDBI_080628_jour.png'

Unm_Stats.unm_stats(path_temperature, path_unm, min_T, path_file, path_file1, path_file2, plot, path_plot, path_plot1, path_plot2)
