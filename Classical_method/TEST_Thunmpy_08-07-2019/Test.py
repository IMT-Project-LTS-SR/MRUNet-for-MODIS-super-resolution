
'''
######################################## FITS ########################
######################################################################
######################################################################
'''
######### Linear fit ################

from Thunmpy import ThunmFit

path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
plot =1
path_plot = 'TEST_Thunmpy_08-07-2019/NDBI_vs_T_080628.png'

ThunmFit.linear_fit(path_index, path_temperature, min_T, path_fit, plot, path_plot)

######### Bilinear fit ################

path_index1 = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
path_index2 = 'TEST_Thunmpy_08-07-2019/Index/albedo_080628_40m.bsq'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
#path_index1 = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
#path_index2 = 'TEST_Thunmpy_08-07-2019/Index/albedo_080628_100m.bsq'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'

ThunmFit.bilinear_fit(path_index1, path_index2, path_temperature, min_T, path_fit)

########## HUTS fit ################

path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
path_albedo = 'TEST_Thunmpy_08-07-2019/Index/albedo_080628_40m.bsq'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
#path_albedo = 'TEST_Thunmpy_08-07-2019/Index/albedo_080628_100m.bsq'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'

ThunmFit.huts_fit(path_index, path_albedo, path_temperature, min_T, path_fit)

########## linear fit window ################

#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
b_radius=2
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
file_out1='TEST_Thunmpy_08-07-2019/Slope_100m_NDBI.bsq'
file_out2='TEST_Thunmpy_08-07-2019/Intercept_100m_NDBI.bsq'
plot =1
path_plot = 'TEST_Thunmpy_08-07-2019/NDBI_vs_T_080628.png'
ThunmFit.linear_fit_window(path_index, path_temperature, min_T, b_radius, file_out1, file_out2, path_fit, plot, path_plot)

########### Fit by class #####################

path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
path_class = 'TEST_Thunmpy_08-07-2019/Class/Classif_40m.bsq' 
min_T = 285
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
plot =1
path_plot = 'TEST_Thunmpy_08-07-2019/NDBI_vs_T_080628.png'

ThunmFit.fit_byclass(path_index, path_temperature, path_class, min_T, path_fit, plot, path_plot)

######### Linear fit Ransac ################

#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
iterations=25
trials=250
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
plot =1
path_plot = 'TEST_Thunmpy_08-07-2019/NDBI_vs_T_080628.png'

ThunmFit.linear_fit_ransac(path_index, path_temperature, min_T, iterations, trials, path_fit, plot, path_plot)

########## linear fit window Ransac ################

#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
b_radius=2
iterations=25
trials=250
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
file_out1='TEST_Thunmpy_08-07-2019/Slope_100m_NDBI_ransac.bsq'
file_out2='TEST_Thunmpy_08-07-2019/Intercept_100m_NDBI_ransac.bsq'

ThunmFit.linear_fit_window_ransac(path_index, path_temperature, min_T, b_radius, iterations, trials, file_out1, file_out2, path_fit)

######### Linear fit Siegel ################
# It doesn't run for the 40m image because of its size (MEMORY ERROR)
# It runs for the 100m image

#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
plot =1
path_plot = 'TEST_Thunmpy_08-07-2019/NDBI_vs_T_080628.png'

ThunmFit.linear_fit_siegel(path_index, path_temperature, min_T, path_fit, plot, path_plot)

########## linear fit window Siegel ################ 

#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
b_radius=2
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
file_out1='TEST_Thunmpy_08-07-2019/Slope_100m_NDBI_Siegel.bsq'
file_out2='TEST_Thunmpy_08-07-2019/Intercept_100m_NDBI_Siegel.bsq'

ThunmFit.linear_fit_window_siegel(path_index, path_temperature, min_T, b_radius, file_out1, file_out2, path_fit)

######### Linear fit Theil Sen ################ 
# It doesn't run for the 40m image because of its size (MEMORY ERROR)
# It runs for the 100m image

#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
plot =1
path_plot = 'TEST_Thunmpy_08-07-2019/NDBI_vs_T_080628.png'

ThunmFit.linear_fit_theilsen(path_index, path_temperature, min_T, path_fit, plot, path_plot)

########## linear fit window Theil Sen ################ 

#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
b_radius=2
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
file_out1='TEST_Thunmpy_08-07-2019/Slope_100m_NDBI_theil.bsq'
file_out2='TEST_Thunmpy_08-07-2019/Intercept_100m_NDBI_theil.bsq'

ThunmFit.linear_fit_window_theilsen(path_index, path_temperature, min_T, b_radius, file_out1, file_out2, path_fit)

######### Linear fit lts ################

#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
plot =1
path_plot = 'TEST_Thunmpy_08-07-2019/NDBI_vs_T_080628.png'
sigx=0.01
sigy=1

ThunmFit.linear_fit_lts(path_index, path_temperature, min_T, sigx, sigy, path_fit, plot, path_plot)

########## linear fit window lts ################ 
# It can give problems in some windows ... However, I have not been able to detect why ... 
# A try-except loop has been included to avoid these errors. Slopes and intercepts in pixels where these errors appear
# are defined as the global slope and intercept obtained with all the pixels.  

#path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
#path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
min_T = 285
b_radius=2
sigx=0.025
sigy=1
min_slope=-300
max_slope=300
min_intercept=270
max_intercept=350
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
file_out1='TEST_Thunmpy_08-07-2019/Slope_100m_NDBI_lts.bsq'
file_out2='TEST_Thunmpy_08-07-2019/Intercept_100m_NDBI_lts.bsq'

ThunmFit.linear_fit_window_lts(path_index, path_temperature, min_T, b_radius, file_out1, file_out2, path_fit, sigx, sigy, min_slope, max_slope, min_intercept, max_intercept)

'''
#################################### UNMIXING ########################
######################################################################
######################################################################
'''

########## Linear Unmixing #####################

from Thunmpy import Thunmixing

path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_20m.img'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
path_out = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from100m_jourNDBI.bsq'
iscale=5
path_mask=0

Thunmixing.linear_unmixing(path_index, path_temperature, path_fit, iscale,path_mask, path_out)

########## Bilinear unmixing ##################

path_index1 = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_20m.img'
path_index2 = 'TEST_Thunmpy_08-07-2019/Index/albedo_080628_20m.bsq'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
iscale=2
path_mask=0
path_out = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from40m_jourNDBI.bsq'

Thunmixing.bilinear_unmixing(path_index1, path_index2, path_temperature, path_fit, iscale, path_mask, path_out)

########## HUTS Unmixing #####################

path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_20m.img'
path_albedo = 'TEST_Thunmpy_08-07-2019/Index/albedo_080628_20m.bsq'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
path_mask=0
path_out = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from40m_jourNDBI_huts.bsq'
iscale= 2

Thunmixing.huts_unmixing(path_index, path_albedo, path_temperature, path_fit, iscale, path_mask, path_out)

########### Linear Unmixing By Class ##########

	
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_20m.img'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
path_mask=0
path_class = 'TEST_Thunmpy_08-07-2019/Class/Classif_20m.bsq' 
path_out = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from40m_jourNDBI_huts.bsq'
iscale= 2	
	
Thunmixing.linear_unmixing_byclass(path_index, path_temperature, path_class, path_fit, iscale, path_mask, path_out)

########### AATPRK Unmixing ##########

	
path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_20m.img'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'	
path_slope='TEST_Thunmpy_08-07-2019/Slope_100m_NDBI_lts.bsq'
path_intercept='TEST_Thunmpy_08-07-2019/Intercept_100m_NDBI_lts.bsq'
iscale=5
path_out = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from100m_jourNDBI_aatprk.bsq'

Thunmixing.aatprk_unmixing(path_index, path_temperature, path_slope, path_intercept, iscale, path_out)


########### AATPRK Unmixing V2 ############

path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_20m.img'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'	
path_slope='TEST_Thunmpy_08-07-2019/Slope_100m_NDBI.bsq'
path_intercept='TEST_Thunmpy_08-07-2019/Intercept_100m_NDBI.bsq'
iscale=5
scc=100
block_size = 5
sill=7
ran=1000
path_out = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from100m_jourNDBI_aatprk.bsq'
path_plot= 'TEST_Thunmpy_08-07-2019/Gamma_c_NDBI_100m_aatprk.png'
path_out_slope='TEST_Thunmpy_08-07-2019/slope_20m_from100m_jour_NDBI_aatprk.bsq'
path_out_intercept='TEST_Thunmpy_08-07-2019/Intercept_20m_from100m_jour_NDBI_aatprk.bsq'


Thunmixing.aatprk_unmixing_v2(path_index, path_temperature, path_slope, path_intercept, iscale,scc, block_size, sill, ran, path_out, path_out_slope, path_out_intercept, path_plot)


'''
##################################### CORRECTION #####################
######################################################################
######################################################################
'''

############## correction ############

from Thunmpy import Thunmcorr

path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
path_unm = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from100m_jourNDBI.bsq'
iscale=2
path_out='TEST_Thunmpy_08-07-2019/Delta_T_100m_from20m_jour_NDBI.bsq'
path_out1='TEST_Thunmpy_08-07-2019/T_unm_add_100m_from20m_jour_NDBI.bsq'
path_out2='TEST_Thunmpy_08-07-2019/Delta_T_final_20m_from100m_jour_NDBI.bsq'
path_out3='TEST_Thunmpy_08-07-2019/T_unmixed_final_20m_from100m_jour_NDBI.bsq'

Thunmcorr.correction(path_index, path_temperature, path_unm, iscale, path_out, path_out1, path_out2, path_out3)

############## correction v2 ############


path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_unm = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from40m_jourNDBI.bsq'
iscale=2
path_out='TEST_Thunmpy_08-07-2019/Delta_T_40m_from20m_jour_NDBI_v2.bsq'
path_out1='TEST_Thunmpy_08-07-2019/T_unm_add_40m_from20m_jour_NDBI_v2.bsq'
path_out2='TEST_Thunmpy_08-07-2019/Delta_T_final_20m_from40m_jour_NDBI_v2.bsq'
path_out3='TEST_Thunmpy_08-07-2019/T_unmixed_final_20m_from40m_jour_NDBI_v2.bsq'

Thunmcorr.correction_v2(path_index, path_fit, path_temperature, path_unm, iscale, path_out, path_out1, path_out2, path_out3)

############## Quality correction ############


path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628.img'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_40m_bruite.bsq'
path_unm = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from40m_jourNDBI_huts.bsq'
iscale=2
max_thre = None
min_thre = None
path_out='TEST_Thunmpy_08-07-2019/Delta_T_40m_from20m_jour_NDBI_qual.bsq'
path_out1='TEST_Thunmpy_08-07-2019/T_unm_add_40m_from20m_jour_NDBI_qual.bsq'
path_out2='TEST_Thunmpy_08-07-2019/Delta_T_final_20m_from40m_jour_NDBI_qual.bsq'
path_out3='TEST_Thunmpy_08-07-2019/T_unmixed_final_20m_from40m_jour_NDBI_qual.bsq'

Thunmcorr.quality_correction(path_index, path_temperature, path_unm, iscale, max_thre, min_thre, path_out, path_out1, path_out2, path_out3)

############## correction ATPRK ############


path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_fit = 'TEST_Thunmpy_08-07-2019/Fit_NDBI_T_080628.txt'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
path_unm = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from100m_jourNDBI.bsq'
iscale=5
scc=100
block_size = 5
sill=7
ran=1000
path_out='TEST_Thunmpy_08-07-2019/Delta_T_100m_from20m_jour_NDBI_atprk.bsq'
path_out1='TEST_Thunmpy_08-07-2019/T_unm_add_100m_from20m_jour_NDBI_atprk.bsq'
path_out2='TEST_Thunmpy_08-07-2019/Delta_T_final_20m_from100m_jour_NDBI_atprk.bsq'
path_out3='TEST_Thunmpy_08-07-2019/T_unmixed_final_20m_from100m_jour_NDBI_atprk.bsq'
path_plot= 'TEST_Thunmpy_08-07-2019/Gamma_c_NDBI_100m.png'

Thunmcorr.correction_ATPRK(path_index, path_fit, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot)


############## correction AATPRK ############


path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
path_slope='TEST_Thunmpy_08-07-2019/Slope_100m_NDBI.bsq'
path_intercept='TEST_Thunmpy_08-07-2019/Intercept_100m_NDBI.bsq'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
path_unm = 'TEST_Thunmpy_08-07-2019/T_unm_20m_from100m_jourNDBI_aatprk.bsq'
iscale=5
scc=100
block_size = 5
sill=7
ran=1000
path_out='TEST_Thunmpy_08-07-2019/Delta_T_100m_from20m_jour_NDBI_aatprk.bsq'
path_out1='TEST_Thunmpy_08-07-2019/T_unm_add_100m_from20m_jour_NDBI_aatprk.bsq'
path_out2='TEST_Thunmpy_08-07-2019/Delta_T_final_20m_from100m_jour_NDBI_aatprk.bsq'
path_out3='TEST_Thunmpy_08-07-2019/T_unmixed_final_20m_from100m_jour_NDBI_aatprk.bsq'
path_plot= 'TEST_Thunmpy_08-07-2019/Gamma_c_NDBI_100m_aatprk.png'

Thunmcorr.correction_AATPRK(path_index, path_slope, path_intercept, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot)


'''
##################################### STATS ##########################
######################################################################
######################################################################
'''

from Thunmpy import Thunmstats

path_unm = 'TEST_Thunmpy_08-07-2019/T_unmixed_final_20m_from100m_jour_NDBI_aatprk.bsq'
path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_20m_bruite.bsq'
min_T = 285
path_file = "TEST_Thunmpy_08-07-2019/Output.txt"
path_file1 = "TEST_Thunmpy_08-07-2019/Output_ttest_.txt"
path_file2 = "TEST_Thunmpy_08-07-2019/Output_corr.txt"
plot =1
path_plot = 'TEST_Thunmpy_08-07-2019/Hist_T20m_080628_jour.png'
path_plot1 = 'TEST_Thunmpy_08-07-2019/Hist_Tunm_100m_NDBI_080628_jour.png'
path_plot2 = 'TEST_Thunmpy_08-07-2019/Hist_delta_100m_NDBI_080628_jour.png'

Thunmstats.unm_stats(path_temperature, path_unm, min_T, path_file, path_file1, path_file2, plot, path_plot, path_plot1, path_plot2)
